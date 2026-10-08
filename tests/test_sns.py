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


def test_log_publication_appends_one_row_per_check(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    import src.ingest.sns as sns

    log = tmp_path / "publication_log.csv"
    monkeypatch.setattr(sns, "PUBLICATION_LOG", log)
    sns.log_publication("monitorizacao-sazonal-csh", "2026-10-03", datetime(2026, 10, 7, 8, 0, tzinfo=timezone.utc))
    sns.log_publication("monitorizacao-sazonal-csh", "2026-10-03", datetime(2026, 10, 8, 8, 0, tzinfo=timezone.utc))
    rows = pd.read_csv(log)
    assert list(rows.columns) == ["checked_at", "dataset_id", "last_day"]
    assert rows["checked_at"].tolist() == ["2026-10-07T08:00:00Z", "2026-10-08T08:00:00Z"]
    assert (rows["last_day"] == "2026-10-03").all()
