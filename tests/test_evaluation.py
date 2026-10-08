import pandas as pd
import pytest

from src.baselines import last_value, seasonal_naive
from src.evaluation import TARGET_REGIONS, score


def _observed(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=len(values), freq="D")
    return pd.DataFrame({region: values for region in TARGET_REGIONS}, index=index)


def test_seasonal_naive_copies_same_weekday_of_previous_week():
    observed = _observed([float(day) for day in range(21)])
    forecasts = seasonal_naive(observed)
    row = forecasts[(forecasts["origin"] == "2025-01-10") & (forecasts["horizon"] == 3)].iloc[0]
    # origem dia 9 (0-based), alvo dia 12, copia o dia 5
    assert row["target_date"] == pd.Timestamp("2025-01-13")
    assert row["y_pred"] == 5.0
    assert row["y_true"] == 12.0


def test_last_value_uses_only_data_up_to_origin_and_fills_gaps():
    values = [10.0, 20.0, None, 40.0]
    forecasts = last_value(_observed(values))
    at_gap = forecasts[(forecasts["origin"] == "2025-01-03") & (forecasts["horizon"] == 1)].iloc[0]
    assert at_gap["y_pred"] == 20.0
    assert at_gap["y_true"] == 40.0


def test_score_ignores_days_without_observed_value():
    forecasts = pd.DataFrame({
        "origin": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "target_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "horizon": [1, 1],
        "ars": [TARGET_REGIONS[0]] * 2,
        "y_true": [50.0, None],
        "y_pred": [40.0, 99.0],
    })
    table = score(forecasts, (pd.Timestamp("2025-01-01"), None))
    assert table.loc[TARGET_REGIONS[0], 1] == pytest.approx(10.0)


def test_seasonal_naive_beyond_one_week_uses_two_weeks_back():
    observed = _observed([float(day) for day in range(30)])
    forecasts = seasonal_naive(observed)
    row = forecasts[(forecasts["origin"] == "2025-01-15") & (forecasts["horizon"] == 10)].iloc[0]
    # origem dia 14 (0-based), alvo dia 24, copia o dia 10: o mesmo dia da semana mais recente já conhecido
    assert row["y_pred"] == 10.0
    assert (forecasts["horizon"].max(), forecasts["horizon"].min()) == (14, 1)


def test_score_reports_short_and_long_horizons_separately():
    forecasts = pd.DataFrame({
        "origin": pd.to_datetime(["2025-01-01"] * 2),
        "target_date": pd.to_datetime(["2025-01-02", "2025-01-11"]),
        "horizon": [1, 10],
        "ars": [TARGET_REGIONS[0]] * 2,
        "y_true": [50.0, 50.0],
        "y_pred": [48.0, 40.0],
    })
    table = score(forecasts, (pd.Timestamp("2025-01-01"), None))
    assert table.loc[TARGET_REGIONS[0], "1-7"] == pytest.approx(2.0)
    assert table.loc[TARGET_REGIONS[0], "8-14"] == pytest.approx(10.0)
