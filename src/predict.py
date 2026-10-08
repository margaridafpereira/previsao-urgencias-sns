"""Previsão diária publicada: média dos 4 modelos (D18), 14 dias, com intervalo de 80 % (D22).

    python -m src.predict

1. A origem é o último dia publicado pelo SNS (o atraso de publicação é visível, D21).
2. Cada modelo é treinado com todos os dados até à origem e prevê os 14 dias seguintes:
   LightGBM, LSTM, SARIMAX e Chronos-2 (este sem treino). A previsão é a média dos quatro.
3. O intervalo vem dos quantis do LightGBM, alargados pelas margens conformal da
   validação (D14, data/interval_margins.json), e contém sempre a previsão.
4. Os modelos são treinados para as regiões com dados recentes (D19), mais as cinco
   da avaliação; só se publicam as regiões com dados recentes. Uma região que volte a
   ter dados (por exemplo o Algarve) entra sozinha.

Grava data/predictions/latest.parquet e uma cópia por dia de execução, para mais
tarde comparar as previsões com o que o SNS publicar.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from src import chronos2, lstm
from src.data import load_daily
from src.data import REGIONS
from src.evaluation import HORIZONS, TARGETS, TARGET_REGIONS
from src.features import _wide, calendar, feature_columns
from src.intervals import QUANTILES, load_margins
from src.lgbm import NUM_ROUNDS, PARAMS, build_tables
from src.regions import active_regions
from src.statistical import EXOG, _sarimax_forecasts

PREDICTIONS_DIR = Path(__file__).resolve().parents[1] / "data" / "predictions"
MODELS = ["lightgbm", "lstm", "sarimax", "chronos2"]
N_HORIZONS = len(HORIZONS)


def _lightgbm(tables: dict[int, pd.DataFrame], origin: pd.Timestamp, params: dict) -> pd.DataFrame:
    """Um modelo por horizonte, treinado com tudo o que se conhecia na origem."""
    frames = []
    for horizon, table in tables.items():
        features = feature_columns(table)
        train = table[(table["target_date"] <= origin) & table["y_rel"].notna()]
        model = lgb.train(params, lgb.Dataset(train[features], train["y_rel"]), NUM_ROUNDS)
        rows = table[table["origin"] == origin]
        frames.append(pd.DataFrame({
            "ars": rows["ars"].astype(str).to_numpy(),
            "horizon": horizon,
            "y": rows["base"].to_numpy() + model.predict(rows[features]),
        }))
    return pd.concat(frames, ignore_index=True)


def _lstm(daily: pd.DataFrame, target: str, origin: pd.Timestamp, regions: list[str]) -> pd.DataFrame:
    data = lstm.Dataset(daily, target, regions)
    index = data.dates.get_loc(origin)
    model = lstm.train(data, np.arange(data.first_valid_origin(), index - N_HORIZONS + 1))
    x, future, region, _, base = data.samples(np.array([index]))
    with torch.no_grad():
        predicted = model(x, future, region).numpy()
    return pd.DataFrame([
        {"ars": regions[r], "horizon": h + 1, "y": base[r] * (1 + predicted[r, h])}
        for r in range(len(regions)) for h in range(N_HORIZONS)
    ])


def _sarimax(daily: pd.DataFrame, target: str, origin: pd.Timestamp, regions: list[str]) -> pd.DataFrame:
    y = _wide(daily, target, regions)
    days = pd.date_range(y.index[0], origin + pd.Timedelta(days=N_HORIZONS), freq="D")
    exog = calendar(days)[EXOG].astype(float)
    rows = []
    for region in regions:
        predicted = _sarimax_forecasts(y[region].asfreq("D"), exog, pd.DatetimeIndex([origin]))[0]
        rows += [{"ars": region, "horizon": h + 1, "y": predicted[h]} for h in range(N_HORIZONS)]
    return pd.DataFrame(rows)


def _chronos2(daily: pd.DataFrame, target: str, origin: pd.Timestamp, pipeline, regions: list[str]) -> pd.DataFrame:
    result = chronos2.forecast_origins(daily, target, pd.DatetimeIndex([origin]), pipeline, regions)
    return result.rename(columns={"y_pred": "y"})[["ars", "horizon", "y"]]


def model_regions(daily: pd.DataFrame, target: str) -> list[str]:
    """As cinco regiões da avaliação mais as que tenham dados recentes, pela ordem de REGIONS."""
    active = set(active_regions(daily, target))
    return [region for region in REGIONS if region in TARGET_REGIONS or region in active]


def predict_target(daily: pd.DataFrame, target: str, pipeline, margins: pd.Series) -> pd.DataFrame:
    regions = model_regions(daily, target)
    origin = daily.loc[daily["ars"].isin(regions) & daily[target].notna(), "periodo"].max()
    tables = build_tables(daily, target, regions)
    timings = {}

    def timed(name, function):
        started = time.time()
        result = function()
        timings[name] = time.time() - started
        return result.set_index(["ars", "horizon"])["y"].rename(name)

    per_model = pd.concat([
        timed("lightgbm", lambda: _lightgbm(tables, origin, PARAMS)),
        timed("lstm", lambda: _lstm(daily, target, origin, regions)),
        timed("sarimax", lambda: _sarimax(daily, target, origin, regions)),
        timed("chronos2", lambda: _chronos2(daily, target, origin, pipeline, regions)),
    ], axis=1)
    bounds = pd.concat([
        timed(column, lambda alpha=alpha: _lightgbm(tables, origin, {**PARAMS, "objective": "quantile", "alpha": alpha}))
        for column, alpha in QUANTILES.items()
    ], axis=1)

    result = per_model.join(bounds).reset_index()
    result["y_pred"] = result[MODELS].mean(axis=1)
    margin = result["horizon"].map(margins)
    low = result[["y_low", "y_high"]].min(axis=1) - margin
    high = result[["y_low", "y_high"]].max(axis=1) + margin
    # O intervalo vem do LightGBM e a previsão da média: garantir que a contém.
    result["y_low"] = np.minimum(low, result["y_pred"])
    result["y_high"] = np.maximum(high, result["y_pred"])
    result["target"] = target
    result["origin"] = origin
    result["target_date"] = origin + pd.to_timedelta(result["horizon"], unit="D")
    print(f"{target}: origem {origin.date()}, " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items()), flush=True)
    return result


def main() -> int:
    run_at = datetime.now(timezone.utc)
    daily = load_daily()
    margins = load_margins()
    pipeline = chronos2.load_pipeline()

    frames = []
    for target in TARGETS:
        result = predict_target(daily, target, pipeline, margins[target])
        frames.append(result[result["ars"].isin(active_regions(daily, target))])

    predictions = pd.concat(frames, ignore_index=True)
    predictions["run_at"] = run_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    run_day = pd.Timestamp(run_at.date())
    # Dias anteriores a hoje que o SNS ainda não publicou: estimativas, não previsões (D21).
    predictions["kind"] = np.where(predictions["target_date"] < run_day, "not_yet_published", "forecast")
    predictions["delay_days"] = (run_day - predictions["origin"]).dt.days
    columns = ["run_at", "target", "ars", "origin", "delay_days", "target_date", "horizon", "kind",
               "y_pred", "y_low", "y_high", *MODELS]
    predictions = predictions[columns].sort_values(["target", "ars", "target_date"]).reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_DIR / "latest.parquet", index=False)
    predictions.to_parquet(PREDICTIONS_DIR / f"{run_at.date().isoformat()}.parquet", index=False)

    wait = predictions[(predictions["target"] == "wait_minutes") & (predictions["kind"] == "forecast")]
    print(f"\nÚltimo dia publicado pelo SNS: {predictions['origin'].max().date()} ({predictions['delay_days'].max()} dias de atraso)")
    print(wait.pivot(index="target_date", columns="ars", values="y_pred").round(0).head(7).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
