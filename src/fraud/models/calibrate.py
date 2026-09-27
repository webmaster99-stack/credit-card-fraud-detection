"""Probability calibration (Phase 3 protocol step 2): map a ranking score to a calibrated fraud
probability, fit on the validation split (the model itself was already fit and tuned on train).
"""

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CalibrationMethod = Literal["sigmoid", "isotonic", "none"]


def fit_calibrator(
    method: CalibrationMethod, raw_score: NDArray[np.float64], y: NDArray[np.int_]
) -> Any:
    if method == "none":
        return None
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(raw_score, y)
        return calibrator
    if method == "sigmoid":
        # Platt scaling: a 1-feature logistic regression of the label on the raw score.
        calibrator = LogisticRegression()
        calibrator.fit(raw_score.reshape(-1, 1), y)
        return calibrator
    raise ValueError(f"Unknown calibration method {method!r}; expected sigmoid, isotonic or none.")


def apply_calibrator(
    method: CalibrationMethod, calibrator: Any, raw_score: NDArray[np.float64]
) -> NDArray[np.float64]:
    if method == "none" or calibrator is None:
        return raw_score
    if method == "isotonic":
        out: NDArray[np.float64] = calibrator.predict(raw_score)
        return out
    if method == "sigmoid":
        proba: NDArray[np.float64] = calibrator.predict_proba(raw_score.reshape(-1, 1))[:, 1]
        return proba
    raise ValueError(f"Unknown calibration method {method!r}; expected sigmoid, isotonic or none.")
