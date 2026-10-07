"""Descarrega o histórico meteorológico diário do Open-Meteo para data/raw/meteo.parquet.

    python -m src.ingest.meteo

Uma coordenada por região (ver docs/data-sources.md). Download completo em cada
execução, tal como nos dados do SNS (decisão D4).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = date(2016, 11, 1)  # primeiro dia dos dados diários do SNS
VARIABLES = ["temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"]

# Os nomes coincidem com a coluna `ars` do SNS, para o join ser direto.
REGIONS: dict[str, tuple[float, float]] = {
    "ARS Norte": (41.1496, -8.6110),                  # Porto
    "ARS Centro": (40.2033, -8.4103),                 # Coimbra
    "ARS Lisboa e Vale do Tejo": (38.7223, -9.1393),  # Lisboa
    "ARS Alentejo": (38.5714, -7.9135),               # Évora
    "ARS Algarve": (37.0194, -7.9304),                # Faro
}

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "meteo.parquet"
TIMEOUT_SECONDS = 60


def download_region(region: str, lat: float, lon: float, end_date: date) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(VARIABLES),
        "timezone": "Europe/Lisbon",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    daily = response.json()["daily"]
    df = pd.DataFrame(daily).rename(columns={"time": "periodo"})
    df["periodo"] = pd.to_datetime(df["periodo"])
    df.insert(1, "ars", region)
    # Os últimos dias podem ainda não estar disponíveis e vêm a null.
    return df.dropna(subset=VARIABLES, how="all")


def main() -> int:
    end_date = date.today() - timedelta(days=1)
    frames = []
    for region, (lat, lon) in REGIONS.items():
        try:
            df = download_region(region, lat, lon, end_date)
        except Exception as error:
            print(f"ERRO   {region}: {error}", file=sys.stderr)
            return 1
        frames.append(df)
        print(f"OK     {region}: {len(df):>5} dias, {df['periodo'].min().date()} -> {df['periodo'].max().date()}")

    result = pd.concat(frames).sort_values(["periodo", "ars"]).reset_index(drop=True)
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(RAW_PATH, index=False)
    print(f"Gravado {RAW_PATH.name}: {len(result)} linhas")
    return 0


if __name__ == "__main__":
    sys.exit(main())
