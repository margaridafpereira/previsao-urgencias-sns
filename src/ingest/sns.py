"""Descarrega os datasets do Portal da Transparência do SNS para data/raw/.

    python -m src.ingest.sns

Faz sempre o download completo de cada dataset (ver decisão D4 em docs/decisions.md):
são poucos MB e o SNS corrige dias passados, por isso uma recolha incremental
perderia essas correções.
"""

from __future__ import annotations

import io
import struct
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets"

# dataset_id -> coluna de data
DATASETS: dict[str, str] = {
    "monitorizacao-sazonal-csh": "periodo",
    "atividade-sindrome-gripal-csh": "periodo",
    "atendimentos-em-urgencia-triagem-manchester": "tempo",
    "atendimentos-por-tipo-de-urgencia-hospitalar-link": "tempo",
}

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TIMEOUT_SECONDS = 120


def download_dataset(dataset_id: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{dataset_id}/exports/parquet"
    response = requests.get(url, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return pd.read_parquet(io.BytesIO(response.content))


def parse_wkb_point(value: bytes | None) -> tuple[float, float] | None:

    if not isinstance(value, (bytes, bytearray)) or len(value) != 21:
        return None
    order = "<" if value[0] == 1 else ">"
    geometry_type, lon, lat = struct.unpack(f"{order}Idd", value[1:])
    return (lon, lat) if geometry_type == 1 else None


def normalize(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """Converte a coluna de data, remove duplicados exatos e ordena."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column]).dt.tz_localize(None).dt.normalize()
    if "localizacao_geografica" in df.columns:
        points = df.pop("localizacao_geografica").map(parse_wkb_point)
        df["lon"] = points.map(lambda p: p[0] if p else None)
        df["lat"] = points.map(lambda p: p[1] if p else None)
    df = df.drop_duplicates()
    return df.sort_values(date_column).reset_index(drop=True)


def save(df: pd.DataFrame, dataset_id: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{dataset_id}.parquet"
    df.to_parquet(path, index=False)
    return path


def main() -> int:
    failures = 0
    for dataset_id, date_column in DATASETS.items():
        try:
            df = normalize(download_dataset(dataset_id), date_column)
            path = save(df, dataset_id)
        except Exception as error: 
            print(f"ERRO   {dataset_id}: {error}", file=sys.stderr)
            failures += 1
            continue
        first, last = df[date_column].min().date(), df[date_column].max().date()
        print(f"OK     {dataset_id}: {len(df):>6} linhas, {first} -> {last} ({path.name})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
