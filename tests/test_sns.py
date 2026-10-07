import struct

import pandas as pd
import pytest

from src.ingest.sns import RAW_DIR, normalize, parse_wkb_point


def test_parse_wkb_point_reads_lon_lat():
    wkb = bytes([1]) + struct.pack("<Idd", 1, -8.715083, 39.232503)
    assert parse_wkb_point(wkb) == pytest.approx((-8.715083, 39.232503))


def test_parse_wkb_point_ignores_missing_values():
    assert parse_wkb_point(None) is None
    assert parse_wkb_point(b"") is None


def test_normalize_removes_duplicates_and_sorts():
    df = pd.DataFrame({"periodo": ["2024-01-02", "2024-01-01", "2024-01-01"], "valor": [2.0, 1.0, 1.0]})
    result = normalize(df, "periodo")
    assert result["valor"].tolist() == [1.0, 2.0]


DAILY = RAW_DIR / "monitorizacao-sazonal-csh.parquet"

@pytest.mark.skipif(not DAILY.exists(), reason="dados ainda não descarregados")
def test_daily_dataset_has_expected_schema_and_no_duplicate_keys():
    df = pd.read_parquet(DAILY)
    assert {"periodo", "ars", "indicador", "valor"} <= set(df.columns)
    assert not df.duplicated(["periodo", "ars", "indicador"]).any()
    assert df["indicador"].str.startswith("Tempo médio de espera").any()
