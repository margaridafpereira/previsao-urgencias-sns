"""N-HiTS (biblioteca neuralforecast): uma rede moderna de previsão já pronta (D12).

    python -m src.nhits                # validação
    python -m src.nhits --test         # teste

Usa as mesmas entradas que a LSTM (D11): as últimas INPUT_SIZE observações, as
covariáveis passadas e o calendário dos dias previstos. A normalização é feita
pela própria rede, janela a janela (scaler "robust"), o que trata a mudança de nível.

Walk-forward pelo `cross_validation` do neuralforecast: uma origem por dia,
com um novo treino a cada REFIT_DAYS origens.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NHITS

from src.baselines import seasonal_naive
from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGETS, TARGET_REGIONS, TEST, VALIDATION, actuals, save_forecasts, score
from src.features import COVID, _wide, calendar

N_HORIZONS = len(HORIZONS)
INPUT_SIZE = 56
REFIT_DAYS = 91  # cerca de 3 meses, como a LSTM
FUTURE = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "after_holiday", "christmas_window", "is_covid"]


def future_calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    cal = calendar(dates)
    cal["dow_sin"] = np.sin(2 * np.pi * cal["dow"] / 7)
    cal["dow_cos"] = np.cos(2 * np.pi * cal["dow"] / 7)
    cal["doy_sin"] = np.sin(2 * np.pi * cal["day_of_year"] / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * cal["day_of_year"] / 365.25)
    cal["is_covid"] = ((dates >= COVID[0]) & (dates <= COVID[1])).astype(int)
    return cal[FUTURE]


def long_frame(daily: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    """Formato do neuralforecast: unique_id, ds, y e covariáveis, sem NaN."""
    columns = {"y": _wide(daily, target)}
    past = ["respiratory_pct", "admission_pct", "temperature_2m_mean", "precipitation_sum"]
    if target != "episodes":
        past.append("episodes")
    for name in past:
        columns[name] = _wide(daily, name)
    frame = pd.concat({name: wide.stack(future_stack=True) for name, wide in columns.items()}, axis=1)
    frame.index.names = ["ds", "unique_id"]
    frame = frame.reset_index()

    dates = pd.DatetimeIndex(frame["ds"].unique()).sort_values()
    frame = frame.join(future_calendar(dates), on="ds")
    # Só ficam NaN no início do histórico, antes de cada série começar.
    frame[past] = frame[past].fillna(0)
    frame = frame.dropna(subset=["y"])
    return frame[["unique_id", "ds", "y", *past, *FUTURE]], past


def pad_future(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Acrescenta dias até `end` para que as últimas origens tenham os 7 horizontes.

    Os valores acrescentados só servem de alvo (nunca entram no treino nem na
    janela de entrada) e não são pontuados, porque score() usa os valores observados.
    """
    last = frame["ds"].max()
    if last >= end:
        return frame
    extra_dates = pd.date_range(last + pd.Timedelta(days=1), end, freq="D")
    cal = future_calendar(extra_dates)
    tail = frame[frame["ds"] == last].drop(columns=FUTURE + ["ds"])
    extra = tail.merge(pd.DataFrame({"ds": extra_dates}), how="cross").join(cal, on="ds")
    return pd.concat([frame, extra[frame.columns]], ignore_index=True)


def make_model(past: list[str]) -> NHITS:
    return NHITS(
        h=N_HORIZONS,
        input_size=INPUT_SIZE,
        futr_exog_list=FUTURE,
        hist_exog_list=past,
        loss=MAE(),
        scaler_type="robust",
        max_steps=1000,
        learning_rate=1e-3,
        early_stop_patience_steps=5,
        val_check_steps=50,
        batch_size=32,
        windows_batch_size=256,
        random_seed=42,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )


def forecast(daily: pd.DataFrame, target: str, period) -> pd.DataFrame:
    frame, past = long_frame(daily, target)
    start, end = period
    end = frame["ds"].max() if end is None else end
    # Origens de (start - 7) a (end - 1); o frame termina em end + 7 - 1 para cobrir todos os horizontes.
    frame = pad_future(frame, end + pd.Timedelta(days=N_HORIZONS - 1))
    frame = frame[frame["ds"] <= end + pd.Timedelta(days=N_HORIZONS - 1)]
    first_origin = start - pd.Timedelta(days=N_HORIZONS)
    n_windows = (end - pd.Timedelta(days=1) - first_origin).days + 1

    nf = NeuralForecast(models=[make_model(past)], freq="D")
    started = time.time()
    cv = nf.cross_validation(
        df=frame, n_windows=n_windows, step_size=1, val_size=56, refit=REFIT_DAYS, h=N_HORIZONS
    )
    print(f"  {target}: {n_windows} origens em {time.time() - started:.0f}s", flush=True)

    cv = cv.rename(columns={"unique_id": "ars", "cutoff": "origin", "ds": "target_date", "NHITS": "y_pred"})
    cv["horizon"] = (cv["target_date"] - cv["origin"]).dt.days
    truth = actuals(daily, target).stack().rename("y_true")
    truth.index.names = ["target_date", "ars"]
    cv = cv.drop(columns="y").join(truth, on=["target_date", "ars"])
    return cv[cv["ars"].isin(TARGET_REGIONS)][FORECAST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="avaliar no teste em vez da validação")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    logging.getLogger("lightning").setLevel(logging.ERROR)
    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    period, period_name = (TEST, "teste") if args.test else (VALIDATION, "validação")
    for target in TARGETS:
        naive = score(seasonal_naive(actuals(daily, target)), period)
        forecasts = forecast(daily, target, period)
        save_forecasts(forecasts, "nhits", target, args.test)
        table = score(forecasts, period)
        print(f"\n{target} | nhits | {period_name}")
        print(table)
        gain = 1 - table.loc["Média", "1-7"] / naive.loc["Média", "1-7"]
        print(f"MAE médio 1-7: {table.loc['Média', '1-7']:.2f} vs. seasonal naive {naive.loc['Média', '1-7']:.2f} ({gain:+.1%})")


if __name__ == "__main__":
    main()
