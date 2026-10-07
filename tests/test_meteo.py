import pandas as pd
import pytest

from src.ingest.meteo import RAW_PATH, REGIONS, VARIABLES
from src.ingest.sns import RAW_DIR


def test_regions_match_sns_region_names():
    sns_daily = RAW_DIR / "monitorizacao-sazonal-csh.parquet"
    if not sns_daily.exists():
        pytest.skip("dados do SNS ainda não descarregados")
    sns_regions = set(pd.read_parquet(sns_daily, columns=["ars"])["ars"].unique())
    assert set(REGIONS) <= sns_regions


@pytest.mark.skipif(not RAW_PATH.exists(), reason="dados ainda não descarregados")
def test_meteo_dataset_has_one_row_per_region_and_day():
    df = pd.read_parquet(RAW_PATH)
    assert {"periodo", "ars", *VARIABLES} <= set(df.columns)
    assert not df.duplicated(["periodo", "ars"]).any()
    assert set(df["ars"]) == set(REGIONS)


@pytest.mark.skipif(not RAW_PATH.exists(), reason="dados ainda não descarregados")
def test_meteo_values_are_plausible_for_portugal():
    df = pd.read_parquet(RAW_PATH)
    assert df["temperature_2m_max"].between(-10, 50).all()
    assert (df["precipitation_sum"].dropna() >= 0).all()
