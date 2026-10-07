"""Que regiões têm dados suficientes para serem previstas (D19, substitui D7).

A avaliação dos modelos usa sempre as mesmas regiões (evaluation.TARGET_REGIONS),
para que as comparações se mantenham válidas. A previsão publicada usa esta regra,
para que uma região entre ou saia sozinha quando o SNS retoma ou deixa de publicar.
"""

from __future__ import annotations

import pandas as pd

from src.data import REGIONS

WINDOW_DAYS = 90
MIN_COVERAGE = 0.8
MAX_STALENESS_DAYS = 14


def region_status(daily: pd.DataFrame, target: str = "wait_minutes") -> pd.DataFrame:
    """Cobertura e atraso de cada região, medidos em relação ao último dia publicado.

    O atraso é contado a partir do último dia com dados em qualquer região,
    não a partir de hoje: o atraso normal de publicação do SNS afeta todas as
    regiões por igual e não deve excluir nenhuma.
    """
    latest = daily.loc[daily[target].notna(), "periodo"].max()
    window = daily[daily["periodo"] > latest - pd.Timedelta(days=WINDOW_DAYS)]
    rows = []
    for region in REGIONS:
        series = window.loc[window["ars"] == region].set_index("periodo")[target]
        observed = series.dropna()
        last = observed.index.max() if len(observed) else pd.NaT
        coverage = len(observed) / WINDOW_DAYS
        staleness = (latest - last).days if pd.notna(last) else None
        active = coverage >= MIN_COVERAGE and staleness is not None and staleness <= MAX_STALENESS_DAYS
        rows.append({"ars": region, "coverage": coverage, "last_value": last, "days_behind": staleness, "active": active})
    return pd.DataFrame(rows).set_index("ars")


def active_regions(daily: pd.DataFrame, target: str = "wait_minutes") -> list[str]:
    status = region_status(daily, target)
    return status.index[status["active"]].tolist()
