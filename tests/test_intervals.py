import numpy as np
import pandas as pd
import pytest

from src.intervals import calibrate, conformal_margins, interval_report


def _forecasts(n: int, half_width: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target_date = pd.date_range("2023-01-01", periods=n, freq="D")
    y_true = 50 + rng.normal(0, 10, n)
    return pd.DataFrame({
        "origin": target_date - pd.Timedelta(days=1),
        "target_date": target_date,
        "horizon": 1,
        "ars": "ARS Norte",
        "y_true": y_true,
        "y_low": 50 - half_width,
        "y_high": 50 + half_width,
    })


def test_conformal_calibration_widens_intervals_that_are_too_narrow():
    validation = _forecasts(700, half_width=2.0)
    margins = conformal_margins(validation)
    assert margins.loc[1] > 0
    calibrated = calibrate(validation, margins)
    coverage = interval_report(calibrated, (pd.Timestamp("2023-01-01"), None)).loc["Total", "cobertura %"]
    assert coverage == pytest.approx(80, abs=2)


def test_conformal_calibration_narrows_intervals_that_are_too_wide():
    validation = _forecasts(700, half_width=40.0)
    assert conformal_margins(validation).loc[1] < 0
