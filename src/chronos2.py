"""Chronos-2 (Amazon, 2025): modelo fundacional pré-treinado, usado sem treino (zero-shot) (D17).

    python -m src.chronos2               # validação
    python -m src.chronos2 --test        # teste

O modelo não é treinado nem afinado com os dados do SNS: em cada origem recebe
os últimos CONTEXT_DAYS dias da série, as covariáveis passadas (gripe,
internamentos, meteorologia, episódios) e o calendário dos 7 dias previstos
(conhecido de antemão), e devolve a previsão e os quantis 10 % e 90 %.

Cada par (origem, região) é tratado como uma série independente, o que permite
prever centenas de origens de uma vez.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from src.baselines import seasonal_naive
from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGETS, TARGET_REGIONS, TEST, VALIDATION, actuals, in_period, score
from src.features import COVID, _wide, calendar
from src.tracking import record

MODEL_ID = "amazon/chronos-2"
N_HORIZONS = len(HORIZONS)
CONTEXT_DAYS = 512
ORIGINS_PER_CALL = 60
FUTURE = ["is_holiday", "after_holiday", "christmas_window"]


def load_pipeline():
    from chronos import Chronos2Pipeline

    torch.set_num_threads(max(1, torch.get_num_threads()))
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")


def build_inputs(
    daily: pd.DataFrame, target: str, regions: list[str] = TARGET_REGIONS
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Tabela diária por região (alvo + covariáveis passadas) e calendário futuro."""
    past = ["respiratory_pct", "admission_pct", "temperature_2m_mean", "precipitation_sum"]
    if target != "episodes":
        past.append("episodes")
    columns = {"target": _wide(daily, target, regions), **{name: _wide(daily, name, regions) for name in past}}
    frame = pd.concat({name: wide.stack(future_stack=True) for name, wide in columns.items()}, axis=1)
    frame.index.names = ["timestamp", "ars"]
    frame = frame.reset_index()
    frame[past] = frame[past].fillna(0)

    days = pd.date_range(frame["timestamp"].min(), frame["timestamp"].max() + pd.Timedelta(days=N_HORIZONS))
    cal = calendar(days)[FUTURE].astype(float)
    cal["is_covid"] = ((days >= COVID[0]) & (days <= COVID[1])).astype(float)
    return frame, cal, past


def forecast_origins(
    daily: pd.DataFrame,
    target: str,
    origins: pd.DatetimeIndex,
    pipeline=None,
    regions: list[str] = TARGET_REGIONS,
) -> pd.DataFrame:
    """Previsões para as origens dadas, com o quantil 10 % e 90 % (y_low, y_high)."""
    pipeline = load_pipeline() if pipeline is None else pipeline
    frame, cal, past = build_inputs(daily, target, regions)
    by_region = {region: group.set_index("timestamp").sort_index() for region, group in frame.groupby("ars")}
    future_columns = FUTURE + ["is_covid"]

    results = []
    started = time.time()
    for chunk_start in range(0, len(origins), ORIGINS_PER_CALL):
        chunk = origins[chunk_start : chunk_start + ORIGINS_PER_CALL]
        contexts, futures = [], []
        for origin in chunk:
            context_days = pd.date_range(origin - pd.Timedelta(days=CONTEXT_DAYS - 1), origin)
            future_days = pd.date_range(origin + pd.Timedelta(days=1), periods=N_HORIZONS)
            for region in regions:
                item = f"{region}|{origin.date()}"
                context = by_region[region].reindex(context_days)[["target", *past]].ffill().bfill()
                context = context.join(cal[future_columns])
                contexts.append(context.assign(item_id=item).rename_axis("timestamp").reset_index())
                future = cal.loc[future_days, future_columns].assign(item_id=item)
                futures.append(future.rename_axis("timestamp").reset_index())
        prediction = pipeline.predict_df(
            pd.concat(contexts, ignore_index=True),
            future_df=pd.concat(futures, ignore_index=True),
            prediction_length=N_HORIZONS,
            quantile_levels=[0.1, 0.5, 0.9],
            freq="D",
        )
        results.append(prediction)
        done = chunk_start + len(chunk)
        print(f"  {target}: {done}/{len(origins)} origens ({time.time() - started:.0f}s)", flush=True)

    prediction = pd.concat(results, ignore_index=True)
    item = prediction["item_id"].str.split("|", expand=True)
    forecasts = pd.DataFrame({
        "ars": item[0],
        "origin": pd.to_datetime(item[1]),
        "target_date": prediction["timestamp"],
        "y_pred": prediction["0.5"],
        "y_low": prediction["0.1"],
        "y_high": prediction["0.9"],
    })
    forecasts["horizon"] = (forecasts["target_date"] - forecasts["origin"]).dt.days
    truth = actuals(daily, target).stack().rename("y_true")
    truth.index.names = ["target_date", "ars"]
    forecasts = forecasts.join(truth, on=["target_date", "ars"])
    return forecasts[FORECAST_COLUMNS + ["y_low", "y_high"]]


def forecast(daily: pd.DataFrame, target: str, period, pipeline=None) -> pd.DataFrame:
    dates = _wide(daily, target).index
    start, end = period
    end = dates[-1] if end is None else end
    origins = dates[(dates >= start - pd.Timedelta(days=N_HORIZONS)) & (dates < end)]
    return forecast_origins(daily, target, origins, pipeline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="avaliar no teste em vez da validação")
    args = parser.parse_args()

    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    pipeline = load_pipeline()
    period, period_name = (TEST, "teste") if args.test else (VALIDATION, "validação")
    for target in TARGETS:
        naive = score(seasonal_naive(actuals(daily, target)), period)
        forecasts = forecast(daily, target, period, pipeline)
        record(forecasts, "chronos2", target, args.test, {"model_id": MODEL_ID, "context_days": CONTEXT_DAYS})
        table = score(forecasts, period)
        print(f"\n{target} | chronos2 (zero-shot) | {period_name}")
        print(table)
        gain = 1 - table.loc["Média", "1-7"] / naive.loc["Média", "1-7"]
        print(f"MAE médio 1-7: {table.loc['Média', '1-7']:.2f} vs. seasonal naive {naive.loc['Média', '1-7']:.2f} ({gain:+.1%})")
        scored = forecasts[in_period(forecasts["target_date"], period)].dropna(subset=["y_true"])
        coverage = scored["y_true"].between(scored["y_low"], scored["y_high"]).mean()
        width = (scored["y_high"] - scored["y_low"]).mean()
        print(f"Intervalo 10-90 %: cobertura {coverage:.1%}, largura média {width:.1f}")


if __name__ == "__main__":
    main()
