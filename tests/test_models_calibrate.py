import numpy as np
import pytest

from fraud.models.calibrate import apply_calibrator, fit_calibrator


@pytest.fixture
def separable_scores() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=500) < 0.1).astype(int)
    raw = np.clip(y * 3 + rng.normal(0, 1, size=500), -5, 8)
    return raw, y


def test_none_is_a_passthrough(separable_scores: tuple[np.ndarray, np.ndarray]) -> None:
    raw, y = separable_scores
    calibrator = fit_calibrator("none", raw, y)
    assert calibrator is None
    np.testing.assert_array_equal(apply_calibrator("none", calibrator, raw), raw)


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrated_output_is_a_probability_and_monotonic(
    method: str, separable_scores: tuple[np.ndarray, np.ndarray]
) -> None:
    raw, y = separable_scores
    calibrator = fit_calibrator(method, raw, y)  # type: ignore[arg-type]
    calibrated = apply_calibrator(method, calibrator, raw)  # type: ignore[arg-type]
    assert ((calibrated >= 0) & (calibrated <= 1)).all()
    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= -1e-9)  # monotonic non-decreasing in raw score


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown calibration method"):
        fit_calibrator("bogus", np.array([0.1, 0.2]), np.array([0, 1]))  # type: ignore[arg-type]
