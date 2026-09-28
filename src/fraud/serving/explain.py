"""Per-prediction reasons for the demo and API: SHAP values folded back onto the input features.

The pipeline one-hot encodes and scales, so raw SHAP columns like `category_shopping_net` or a
scaled `log_amt` mean little to a reader. Contributions are summed per source feature (all `state_*`
columns become "state") and shown next to the unscaled value the caller actually submitted.
SHAP explains the model's raw score (log-odds, before calibration); the sign and ranking carry over
to the calibrated probability, the magnitude is not a probability.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud.features.pipeline import CATEGORICAL

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# feature -> (label shown to the reader, formatter of the unscaled value)
_SPECS: dict[str, tuple[str, Callable[[Any], str]]] = {
    "log_amt": ("Amount", lambda v: f"${np.expm1(float(v)):,.2f}"),
    "category": ("Merchant category", str),
    "hour": ("Hour of day", lambda v: f"{int(v):02d}:00"),
    "weekday": ("Day of week", lambda v: WEEKDAYS[int(v)]),
    "is_night": ("Night-time purchase", lambda v: "yes" if float(v) else "no"),
    "age": ("Cardholder age", lambda v: f"{float(v):.0f}"),
    "gender": ("Gender", str),
    "log_city_pop": ("City population", lambda v: f"{np.expm1(float(v)):,.0f}"),
    "state": ("Home state", str),
    "distance_km": ("Cardholder-to-merchant distance", lambda v: f"{float(v):,.0f} km"),
}


@dataclass(frozen=True)
class Reason:
    feature: str
    label: str
    value: str
    contribution: float
    direction: str  # "raises" or "lowers"

    @property
    def text(self) -> str:
        return f"{self.label} = {self.value} {self.direction} the fraud score"


def source_feature(column: str) -> str:
    """The input feature an encoded column comes from (`state_TX` -> `state`)."""
    for name in CATEGORICAL:
        if column.startswith(f"{name}_"):
            return name
    return column


def make_reason(feature: str, contribution: float, raw_value: Any) -> Reason:
    label, fmt = _SPECS.get(feature, (feature.replace("_", " ").capitalize(), lambda v: f"{v:.3g}"))
    return Reason(
        feature=feature,
        label=label,
        value=fmt(raw_value),
        contribution=float(contribution),
        direction="raises" if contribution > 0 else "lowers",
    )


def build_reasons(
    shap_values: np.ndarray, encoded_columns: list[str], raw: pd.DataFrame, top_k: int
) -> list[list[Reason]]:
    """Top-k reasons per row from SHAP values over the encoded columns and the unscaled features."""
    grouped = pd.DataFrame(shap_values, columns=encoded_columns).T.groupby(source_feature).sum().T
    out: list[list[Reason]] = []
    for i in range(len(grouped)):
        row = grouped.iloc[i]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(top_k)
        out.append(
            [make_reason(str(name), value, raw.iloc[i][str(name)]) for name, value in top.items()]
        )
    return out


def reasons_frame(reasons: list[Reason]) -> pd.DataFrame:
    """Chart-ready rows (label with value, signed contribution, effect), largest effect first."""
    return pd.DataFrame(
        {
            "reason": [f"{r.label}: {r.value}" for r in reasons],
            "contribution": [r.contribution for r in reasons],
            "effect": [f"{r.direction} fraud score" for r in reasons],
        }
    )
