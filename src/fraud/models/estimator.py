"""The deployable model: a ladder estimator plus calibration and a decision threshold, composed with
the feature pipeline into one scikit-learn `Pipeline` (CLAUDE.md model lineage: "the deployable unit
is one scikit-learn Pipeline (features + model)").

`fit` trains only the base estimator (on train, via the outer Pipeline's normal `fit`); `calibrate`
and `set_threshold` are called afterwards, explicitly, on the validation split, per the Phase 3
protocol (tune on train -> calibrate on validation -> pick threshold on validation).
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin

from fraud.models.calibrate import CalibrationMethod, apply_calibrator, fit_calibrator
from fraud.models.ladder import fraud_score


class CalibratedThresholdClassifier(BaseEstimator, ClassifierMixin):  # type: ignore[misc]
    def __init__(self, estimator: Any, calibration_method: CalibrationMethod = "sigmoid") -> None:
        self.estimator = estimator
        self.calibration_method = calibration_method

    def fit(self, X: pd.DataFrame, y: NDArray[np.int_]) -> "CalibratedThresholdClassifier":
        self.estimator.fit(X, y)
        self.calibrator_: Any = None
        self.threshold_: float = 0.5
        self.classes_ = np.array([0, 1])
        return self

    def calibrate(self, X_valid: pd.DataFrame, y_valid: NDArray[np.int_]) -> None:
        raw = fraud_score(self.estimator, X_valid)
        self.calibrator_ = fit_calibrator(self.calibration_method, raw, y_valid)

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    def predict_proba(self, X: pd.DataFrame) -> NDArray[np.float64]:
        raw = fraud_score(self.estimator, X)
        fraud_p = apply_calibrator(self.calibration_method, self.calibrator_, raw)
        return np.column_stack([1 - fraud_p, fraud_p])

    def predict(self, X: pd.DataFrame) -> NDArray[np.int_]:
        flags: NDArray[np.int_] = (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)
        return flags
