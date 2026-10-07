"""Modelos estatísticos clássicos, uma série de cada vez (D16).

    python -m src.statistical                # validação
    python -m src.statistical --test         # teste

- ets: Holt-Winters aditivo com tendência amortecida e sazonalidade semanal.
- sarimax: SARIMA(1,0,1)(1,1,1)_7 com o calendário dos dias previstos como
  variáveis exógenas (feriado, pós-feriado, Natal), conhecido de antemão.

São os modelos mais usados na literatura sobre urgências (docs/literature.md).
Cada série é ajustada nos últimos TRAIN_YEARS anos, de 3 em 3 meses; entre
reajustes, os parâmetros ficam fixos e o modelo só é atualizado com os dados
novos até cada origem.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.baselines import seasonal_naive
from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGETS, TARGET_REGIONS, TEST, VALIDATION, actuals, score
from src.features import _wide, calendar
from src.tracking import record

N_HORIZONS = len(HORIZONS)
TRAIN_YEARS = 3
REFIT_MONTHS = 3
EXOG = ["is_holiday", "after_holiday", "christmas_window"]


def _ets_forecasts(history: pd.Series, origins: pd.DatetimeIndex) -> np.ndarray:
    window = pd.Timedelta(days=365 * TRAIN_YEARS)
    fitted = ExponentialSmoothing(
        history[origins[0] - window : origins[0]], trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
    ).fit()
    params = {
        "smoothing_level": fitted.params["smoothing_level"],
        "smoothing_trend": fitted.params["smoothing_trend"],
        "smoothing_seasonal": fitted.params["smoothing_seasonal"],
        "damping_trend": fitted.params["damping_trend"],
    }
    result = []
    for origin in origins:
        model = ExponentialSmoothing(
            history[origin - window : origin], trend="add", damped_trend=True, seasonal="add", seasonal_periods=7
        )
        result.append(model.fit(**params, optimized=False).forecast(N_HORIZONS).to_numpy())
    return np.array(result)


def _sarimax_forecasts(history: pd.Series, exog: pd.DataFrame, origins: pd.DatetimeIndex) -> np.ndarray:
    window = pd.Timedelta(days=365 * TRAIN_YEARS)
    train = history[origins[0] - window : origins[0]]
    fitted = SARIMAX(
        train, exog=exog.loc[train.index], order=(1, 0, 1), seasonal_order=(1, 1, 1, 7)
    ).fit(disp=False)
    result = []
    for origin in origins:
        current = history[origin - window : origin]
        updated = fitted.apply(current, exog=exog.loc[current.index])
        future = pd.date_range(origin + pd.Timedelta(days=1), periods=N_HORIZONS, freq="D")
        result.append(np.asarray(updated.forecast(N_HORIZONS, exog=exog.loc[future])))
    return np.array(result)


def forecast(daily: pd.DataFrame, target: str, period, model: str) -> pd.DataFrame:
    y = _wide(daily, target)  # lacunas preenchidas com o último valor (só passado)
    start, end = period
    end = y.index[-1] if end is None else end
    # Calendário conhecido de antemão: estende-se 7 dias para lá do fim dos dados.
    all_days = pd.date_range(y.index[0], y.index[-1] + pd.Timedelta(days=N_HORIZONS), freq="D")
    exog = calendar(all_days)[EXOG].astype(float)

    origins = y.index[(y.index >= start - pd.Timedelta(days=N_HORIZONS)) & (y.index < end)]
    months = origins.to_period("M")
    blocks = (months - months.min()).map(lambda offset: offset.n // REFIT_MONTHS)

    frames = []
    for region in TARGET_REGIONS:
        history = y[region].asfreq("D")
        for block in np.unique(blocks):
            block_origins = origins[blocks == block]
            started = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if model == "ets":
                    predicted = _ets_forecasts(history, block_origins)
                else:
                    predicted = _sarimax_forecasts(history, exog, block_origins)
            for h in range(N_HORIZONS):
                frames.append(pd.DataFrame({
                    "origin": block_origins,
                    "target_date": block_origins + pd.Timedelta(days=h + 1),
                    "horizon": h + 1,
                    "ars": region,
                    "y_pred": predicted[:, h],
                }))
        print(f"  {model} {target} {region}: {time.time() - started:.0f}s no último bloco", flush=True)

    forecasts = pd.concat(frames, ignore_index=True)
    truth = actuals(daily, target).stack().rename("y_true")
    truth.index.names = ["target_date", "ars"]
    forecasts = forecasts.join(truth, on=["target_date", "ars"])
    return forecasts[FORECAST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="avaliar no teste em vez da validação")
    parser.add_argument("--models", nargs="+", default=["ets", "sarimax"], choices=["ets", "sarimax"])
    args = parser.parse_args()

    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    period, period_name = (TEST, "teste") if args.test else (VALIDATION, "validação")
    for target in TARGETS:
        naive = score(seasonal_naive(actuals(daily, target)), period)
        for model in args.models:
            forecasts = forecast(daily, target, period, model)
            record(forecasts, model, target, args.test, {"train_years": TRAIN_YEARS, "refit_months": REFIT_MONTHS})
            table = score(forecasts, period)
            print(f"\n{target} | {model} | {period_name}")
            print(table)
            gain = 1 - table.loc["Média", "1-7"] / naive.loc["Média", "1-7"]
            print(f"MAE médio 1-7: {table.loc['Média', '1-7']:.2f} vs. seasonal naive {naive.loc['Média', '1-7']:.2f} ({gain:+.1%})")


if __name__ == "__main__":
    main()
