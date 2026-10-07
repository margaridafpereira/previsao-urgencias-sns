"""Protocolo de avaliação comum a todos os modelos (decisão D8 em docs/decisions.md).

Uma previsão é feita na origem `t` (último dia com dados conhecidos) para os dias
`t + h`, com `h` de 1 a 7. Todos os modelos devolvem uma tabela com as colunas
FORECAST_COLUMNS e são pontuados pela mesma função `score()`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import REGIONS

TARGETS = ["wait_minutes", "episodes"]
HORIZONS = range(1, 8)

# O Algarve fica de fora como alvo (D7).
TARGET_REGIONS = [region for region in REGIONS if region != "ARS Algarve"]

# Os períodos referem-se à data prevista (target_date), não à origem.
VALIDATION = (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))
TEST = (pd.Timestamp("2025-01-01"), None)

FORECAST_COLUMNS = ["origin", "target_date", "horizon", "ars", "y_true", "y_pred"]

# Previsões gravadas por cada modelo, para combinar modelos sem os voltar a treinar.
FORECASTS_DIR = Path(__file__).resolve().parents[1] / "data" / "forecasts"


def in_period(dates: pd.Series, period: tuple[pd.Timestamp, pd.Timestamp | None]) -> pd.Series:
    start, end = period
    mask = dates >= start
    return mask if end is None else mask & (dates <= end)


def actuals(daily: pd.DataFrame, target: str) -> pd.DataFrame:
    """Valores observados do alvo, uma coluna por região-alvo e uma linha por dia."""
    wide = daily.pivot(index="periodo", columns="ars", values=target)[TARGET_REGIONS]
    return wide.asfreq("D")


def score(forecasts: pd.DataFrame, period: tuple[pd.Timestamp, pd.Timestamp | None]) -> pd.DataFrame:
    """MAE por região e horizonte, só nos dias do período com valor observado.

    Devolve uma linha por região mais a linha "Média" e uma coluna por horizonte
    mais a coluna "1-7".
    """
    scored = forecasts[in_period(forecasts["target_date"], period)].dropna(subset=["y_true"])
    if scored["y_pred"].isna().any():
        raise ValueError("há previsões em falta para dias com valor observado")
    error = (scored["y_true"] - scored["y_pred"]).abs()
    table = error.groupby([scored["ars"], scored["horizon"]]).mean().unstack("horizon")
    table = table.reindex(TARGET_REGIONS)
    table["1-7"] = error.groupby(scored["ars"]).mean()
    table.loc["Média"] = table.mean()
    table.columns.name = "horizon"
    return table


def forecast_path(model: str, target: str, test: bool) -> Path:
    return FORECASTS_DIR / f"{model}_{target}_{'test' if test else 'validation'}.parquet"


def save_forecasts(forecasts: pd.DataFrame, model: str, target: str, test: bool) -> None:
    path = forecast_path(model, target, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    forecasts[FORECAST_COLUMNS].to_parquet(path, index=False)


def load_forecasts(model: str, target: str, test: bool) -> pd.DataFrame:
    return pd.read_parquet(forecast_path(model, target, test))
