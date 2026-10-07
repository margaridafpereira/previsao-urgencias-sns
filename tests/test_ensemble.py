import pandas as pd
import pytest

from src.ensemble import combine


def _forecasts(predictions: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "origin": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "horizon": [1, 1],
        "ars": ["ARS Norte", "ARS Norte"],
        "y_true": [50.0, 60.0],
        "y_pred": predictions,
    })


def test_combine_weights_the_two_forecasts():
    combined = combine(_forecasts([40.0, 70.0]), _forecasts([60.0, 50.0]), weight=0.75)
    assert combined["y_pred"].tolist() == pytest.approx([45.0, 65.0])
    assert combined["y_true"].tolist() == [50.0, 60.0]
