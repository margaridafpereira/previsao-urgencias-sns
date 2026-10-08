import numpy as np
import pandas as pd

from src.data import REGIONS
from src.regions import active_regions, region_status


def _daily(days: int, missing: dict[str, slice]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    frame = pd.DataFrame(
        [(day, region, 60.0) for day in dates for region in REGIONS], columns=["periodo", "ars", "wait_minutes"]
    )
    for region, days_slice in missing.items():
        gone = frame["periodo"].isin(dates[days_slice]) & (frame["ars"] == region)
        frame.loc[gone, "wait_minutes"] = np.nan
    return frame


def test_region_without_recent_data_is_excluded():
    daily = _daily(200, {"ARS Algarve": slice(150, None)})  # parou há 50 dias
    assert "ARS Algarve" not in active_regions(daily)
    assert len(active_regions(daily)) == len(REGIONS) - 1


def test_region_returns_once_it_has_enough_recent_data():
    stopped = _daily(200, {"ARS Algarve": slice(0, 120)})  # voltou há 80 dias, 89 % de cobertura
    assert "ARS Algarve" in active_regions(stopped)
    just_back = _daily(200, {"ARS Algarve": slice(0, 170)})  # voltou há 30 dias
    assert "ARS Algarve" not in active_regions(just_back)


def test_publication_delay_common_to_all_regions_excludes_none():
    daily = _daily(200, {region: slice(195, None) for region in REGIONS})  # ninguém publicou nos últimos 5 dias
    status = region_status(daily)
    assert status["active"].all()
    assert (status["days_behind"] == 0).all()


def test_models_cover_the_evaluation_regions_plus_any_region_that_returns():
    from src.evaluation import TARGET_REGIONS
    from src.predict import model_regions

    gone = _daily(200, {"ARS Algarve": slice(150, None)})
    assert model_regions(gone, "wait_minutes") == TARGET_REGIONS
    back = _daily(200, {"ARS Algarve": slice(0, 120)})
    assert "ARS Algarve" in model_regions(back, "wait_minutes")
    assert model_regions(back, "wait_minutes") == list(REGIONS)
