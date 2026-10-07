import pandas as pd
import pytest

from src.data import RAW_DIR, load_daily
from src.evaluation import TARGET_REGIONS
from src.nhits import FUTURE, long_frame, pad_future

REQUIRED = ["monitorizacao-sazonal-csh", "atividade-sindrome-gripal-csh", "meteo"]
needs_data = pytest.mark.skipif(
    not all((RAW_DIR / f"{name}.parquet").exists() for name in REQUIRED),
    reason="dados ainda não descarregados",
)


@needs_data
def test_long_frame_has_no_missing_values_and_one_row_per_day_and_region():
    frame, past = long_frame(load_daily(), "wait_minutes")
    assert not frame.isna().any().any()
    assert not frame.duplicated(["unique_id", "ds"]).any()
    assert set(frame["unique_id"]) == set(TARGET_REGIONS)
    assert "episodes" in past


@needs_data
def test_pad_future_adds_days_with_the_known_calendar():
    frame, _ = long_frame(load_daily(), "wait_minutes")
    end = frame["ds"].max() + pd.Timedelta(days=6)
    padded = pad_future(frame, end)
    added = padded[padded["ds"] > frame["ds"].max()]
    assert len(added) == 6 * len(TARGET_REGIONS)
    christmas = pad_future(frame[frame["ds"] <= "2024-12-20"], pd.Timestamp("2024-12-26"))
    assert christmas.loc[christmas["ds"] == "2024-12-25", "is_holiday"].eq(1).all()
    assert set(FUTURE) <= set(padded.columns)
