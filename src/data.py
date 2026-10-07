"""Carrega os dados brutos de data/raw/ numa tabela diária: uma linha por (periodo, ars)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Prefixo do indicador no SNS -> nome curto da coluna
INDICATORS = {
    "Tempo médio de espera": "wait_minutes",
    "Número estimado de episódios": "episodes",
    "Taxa diária de atendimentos urgentes com prioridade verde ou azul": "non_urgent_rate",
    "Taxa diária de atendimentos urgentes com internamento": "admission_pct",
    "Taxa de episódios de urgência com diagnóstico de infeção respiratória": "respiratory_pct",
}

REGIONS = [
    "ARS Norte",
    "ARS Centro",
    "ARS Lisboa e Vale do Tejo",
    "ARS Alentejo",
    "ARS Algarve",
    "Portugal Continental",
]


def _short_name(indicator: str) -> str | None:
    return next((short for prefix, short in INDICATORS.items() if indicator.startswith(prefix)), None)


def load_daily(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Junta os indicadores diários do SNS e a meteorologia.

    Os dois datasets diários do SNS repetem três indicadores com valores idênticos;
    do dataset da gripe só se aproveita a taxa de infeção respiratória.
    Os dias sem dados ficam como linhas com NaN, para que as lacunas sejam visíveis.
    """
    sns = pd.concat(
        [
            pd.read_parquet(raw_dir / "monitorizacao-sazonal-csh.parquet"),
            pd.read_parquet(raw_dir / "atividade-sindrome-gripal-csh.parquet"),
        ]
    )
    sns["indicator"] = sns["indicador"].map(_short_name)
    sns = sns.dropna(subset=["indicator"]).drop_duplicates(["periodo", "ars", "indicator"])
    wide = sns.pivot(index=["periodo", "ars"], columns="indicator", values="valor")

    days = pd.date_range(sns["periodo"].min(), sns["periodo"].max(), freq="D")
    full_index = pd.MultiIndex.from_product([days, REGIONS], names=["periodo", "ars"])
    daily = wide.reindex(full_index).reset_index()
    daily.columns.name = None

    meteo = pd.read_parquet(raw_dir / "meteo.parquet")
    return daily.merge(meteo, on=["periodo", "ars"], how="left")
