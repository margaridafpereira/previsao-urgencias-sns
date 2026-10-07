import pytest

from src.data import RAW_DIR, REGIONS, load_daily

REQUIRED = ["monitorizacao-sazonal-csh", "atividade-sindrome-gripal-csh", "meteo"]


@pytest.mark.skipif(
    not all((RAW_DIR / f"{name}.parquet").exists() for name in REQUIRED),
    reason="dados ainda não descarregados",
)
def test_load_daily_has_one_row_per_day_and_region():
    daily = load_daily()
    assert not daily.duplicated(["periodo", "ars"]).any()
    assert set(daily["ars"]) == set(REGIONS)
    days = daily["periodo"].nunique()
    assert len(daily) == days * len(REGIONS)
    assert {"wait_minutes", "episodes", "respiratory_pct", "temperature_2m_mean"} <= set(daily.columns)
