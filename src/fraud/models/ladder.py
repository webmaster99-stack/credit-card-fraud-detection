"""The model ladder (docs/plan.md, Phase 3): each rung must beat the previous on the primary metric
(recall at precision >= min_precision) to earn its complexity.

Every estimator exposes `fit(X, y)` and a uniform `fraud_score(estimator, X)` ranking score, so
`tune.py` and `train.py` can drive all seven rungs through one harness. Isolation Forest ignores `y`
in `fit` (unsupervised); its score sign is flipped to match "higher = more fraud-like".
"""

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from xgboost import XGBClassifier


class FraudEstimator(Protocol):
    def fit(self, X: pd.DataFrame, y: NDArray[np.int_]) -> "FraudEstimator": ...


class DummyLegitClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Step 0 floor: always predicts legitimate. Never flags anything."""

    def fit(self, X: pd.DataFrame, y: NDArray[np.int_]) -> "DummyLegitClassifier":
        return self

    def fraud_score(self, X: pd.DataFrame) -> NDArray[np.float64]:
        return np.zeros(len(X), dtype=np.float64)


class AmountRuleHeuristic(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Step 0 floor: score is the transaction amount's percentile rank within the fitted data.

    A pure amount rule, deliberately ignorant of every other feature. Ranks on `log_amt` (already in
    the transformed feature frame): log and standard-scaling are both monotonic, so the percentile
    rank is identical to ranking on the raw amount.
    """

    def __init__(self, amount_column: str = "log_amt") -> None:
        self.amount_column = amount_column

    def fit(self, X: pd.DataFrame, y: NDArray[np.int_]) -> "AmountRuleHeuristic":
        self.reference_ = np.sort(X[self.amount_column].to_numpy(dtype=np.float64))
        return self

    def fraud_score(self, X: pd.DataFrame) -> NDArray[np.float64]:
        amounts = X[self.amount_column].to_numpy(dtype=np.float64)
        return np.searchsorted(self.reference_, amounts, side="right") / len(self.reference_)


class SplineInteractionLogisticRegression(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    """Step 2: logistic regression on top of spline-expanded amount/hour plus pairwise interactions
    of the numeric columns, to see how far a linear model can go before reaching for trees.
    """

    def __init__(
        self,
        C: float = 1.0,
        n_knots: int = 5,
        numeric_columns: tuple[str, ...] = ("log_amt", "hour", "distance_km"),
        seed: int = 42,
    ) -> None:
        self.C = C
        self.n_knots = n_knots
        self.numeric_columns = numeric_columns
        self.seed = seed

    def _expand(self, X: pd.DataFrame, fit: bool) -> NDArray[np.float64]:
        present = [c for c in self.numeric_columns if c in X.columns]
        numeric = X[present].to_numpy(dtype=np.float64)
        if fit:
            self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3, include_bias=False)
            splined = self.spline_.fit_transform(numeric)
        else:
            splined = self.spline_.transform(numeric)
        rest = X.drop(columns=present).to_numpy(dtype=np.float64)
        interactions = (numeric[:, :, None] * numeric[:, None, :]).reshape(len(X), -1)
        return np.concatenate([rest, splined, interactions], axis=1)

    def fit(self, X: pd.DataFrame, y: NDArray[np.int_]) -> "SplineInteractionLogisticRegression":
        expanded = self._expand(X, fit=True)
        self.model_ = LogisticRegression(
            C=self.C, class_weight="balanced", max_iter=1000, random_state=self.seed
        )
        self.model_.fit(expanded, y)
        return self

    def fraud_score(self, X: pd.DataFrame) -> NDArray[np.float64]:
        proba: NDArray[np.float64] = self.model_.predict_proba(self._expand(X, fit=False))[:, 1]
        return proba


def fraud_score(estimator: Any, X: pd.DataFrame) -> NDArray[np.float64]:
    """Ranking score, higher = more fraud-like. Uses the model's own hook when it has one."""
    if hasattr(estimator, "fraud_score"):
        score: NDArray[np.float64] = estimator.fraud_score(X)
        return score
    if isinstance(estimator, IsolationForest):
        # decision_function: higher = more normal. Flip so higher = more anomalous (fraud-like).
        anomaly: NDArray[np.float64] = -estimator.decision_function(X)
        return anomaly
    proba: NDArray[np.float64] = estimator.predict_proba(X)[:, 1]
    return proba


def _build_logreg(p: dict[str, Any], seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=p.get("C", 1.0), class_weight="balanced", max_iter=1000, random_state=seed
    )


def _build_spline_logreg(p: dict[str, Any], seed: int) -> SplineInteractionLogisticRegression:
    return SplineInteractionLogisticRegression(
        C=p.get("C", 1.0), n_knots=p.get("n_knots", 5), seed=seed
    )


def _build_random_forest(p: dict[str, Any], seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", None),
        min_samples_leaf=p.get("min_samples_leaf", 1),
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def _build_lightgbm(p: dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=p.get("n_estimators", 300),
        num_leaves=p.get("num_leaves", 31),
        learning_rate=p.get("learning_rate", 0.05),
        min_child_samples=p.get("min_child_samples", 20),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        scale_pos_weight=p.get("scale_pos_weight", 1.0),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _build_xgboost(p: dict[str, Any], seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 6),
        learning_rate=p.get("learning_rate", 0.05),
        min_child_weight=p.get("min_child_weight", 1),
        subsample=p.get("subsample", 0.8),
        colsample_bytree=p.get("colsample_bytree", 0.8),
        scale_pos_weight=p.get("scale_pos_weight", 1.0),
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
        eval_metric="aucpr",
    )


def _build_isolation_forest(p: dict[str, Any], seed: int) -> IsolationForest:
    return IsolationForest(
        n_estimators=p.get("n_estimators", 200),
        contamination=p.get("contamination", 0.005),
        max_samples=p.get("max_samples", "auto"),
        n_jobs=-1,
        random_state=seed,
    )


def _space_logreg(trial: optuna.Trial) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def _space_spline_logreg(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "n_knots": trial.suggest_int("n_knots", 3, 8),
    }


def _space_random_forest(trial: optuna.Trial) -> dict[str, Any]:
    # n_estimators is capped at 300 (not 500+): measured at ~410s/fit for 300 trees on the full
    # train split, by far the ladder's most expensive rung, so params.yaml's tune.n_trials_by_step
    # also cuts this step's trial budget hardest.
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
    }


def _space_lightgbm(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 200.0, log=True),
    }


def _space_xgboost(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 200.0, log=True),
    }


def _space_isolation_forest(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "contamination": trial.suggest_float("contamination", 0.001, 0.02, log=True),
    }


@dataclass(frozen=True)
class LadderStep:
    id: int
    key: str
    name: str
    build: Any
    space: Any | None = None  # optuna.Trial -> dict, or None if not tunable
    unsupervised: bool = False


LADDER: list[LadderStep] = [
    LadderStep(0, "dummy", "Dummy (always legit)", lambda p, s: DummyLegitClassifier()),
    LadderStep(0, "amount_rule", "Amount-rule heuristic", lambda p, s: AmountRuleHeuristic()),
    LadderStep(1, "logreg", "Logistic regression (L2, scaled)", _build_logreg, space=_space_logreg),
    LadderStep(
        2,
        "logreg_interact",
        "Logistic regression + splines/interactions",
        _build_spline_logreg,
        space=_space_spline_logreg,
    ),
    LadderStep(
        3, "random_forest", "Random forest", _build_random_forest, space=_space_random_forest
    ),
    LadderStep(4, "lightgbm", "LightGBM", _build_lightgbm, space=_space_lightgbm),
    LadderStep(5, "xgboost", "XGBoost", _build_xgboost, space=_space_xgboost),
    LadderStep(
        6,
        "isolation_forest",
        "Isolation Forest (unsupervised)",
        _build_isolation_forest,
        space=_space_isolation_forest,
        unsupervised=True,
    ),
]

STEPS_BY_KEY: dict[str, LadderStep] = {step.key: step for step in LADDER}


def get_step(key: str) -> LadderStep:
    if key not in STEPS_BY_KEY:
        raise ValueError(f"Unknown ladder step {key!r}; expected one of {list(STEPS_BY_KEY)}.")
    return STEPS_BY_KEY[key]


def build_estimator(key: str, hyperparams: dict[str, Any], seed: int) -> Any:
    return get_step(key).build(hyperparams, seed)
