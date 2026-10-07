"""Baselines ingénuas, a fasquia que todos os modelos têm de bater (D2).

    python -m src.baselines

- last_value: o valor de ontem (da origem) repete-se nos 7 dias seguintes.
- seasonal_naive: cada dia repete o mesmo dia da semana anterior,
  ou seja `t + h` recebe o valor de `t + h - 7`, que é sempre conhecido na origem.

As lacunas do histórico são preenchidas com o último valor conhecido antes de
servirem de entrada, para que haja sempre previsão (só usa informação passada).
"""

from __future__ import annotations

import pandas as pd

from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGETS, TEST, VALIDATION, actuals, in_period, score
from src.tracking import record


def _forecast(observed: pd.DataFrame, lag_for_horizon) -> pd.DataFrame:
    """Gera previsões para todas as origens e horizontes.

    `lag_for_horizon(h)` diz quantos dias antes da origem está o valor que se copia.
    """
    history = observed.ffill()
    frames = []
    for horizon in HORIZONS:
        predicted = history.shift(lag_for_horizon(horizon))  
        frame = predicted.stack().rename("y_pred").reset_index()
        frame.columns = ["origin", "ars", "y_pred"]
        frame["horizon"] = horizon
        frame["target_date"] = frame["origin"] + pd.Timedelta(days=horizon)
        frames.append(frame)
    forecasts = pd.concat(frames, ignore_index=True)
    truth = observed.stack().rename("y_true").reset_index()
    truth.columns = ["target_date", "ars", "y_true"]
    forecasts = forecasts.merge(truth, on=["target_date", "ars"], how="left")
    return forecasts[FORECAST_COLUMNS]


def last_value(observed: pd.DataFrame) -> pd.DataFrame:
    return _forecast(observed, lambda horizon: 0)


def seasonal_naive(observed: pd.DataFrame) -> pd.DataFrame:
    return _forecast(observed, lambda horizon: 7 - horizon)


BASELINES = {"last_value": last_value, "seasonal_naive": seasonal_naive}


def main() -> None:
    daily = load_daily()
    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    for target in TARGETS:
        observed = actuals(daily, target)
        for name, baseline in BASELINES.items():
            forecasts = baseline(observed)
            for test, period_name, period in [(False, "validação", VALIDATION), (True, "teste", TEST)]:
                print(f"\n{target} | {name} | {period_name}: MAE por região e horizonte")
                print(score(forecasts, period))
                record(forecasts[in_period(forecasts["target_date"], period)], name, target, test)


if __name__ == "__main__":
    main()
