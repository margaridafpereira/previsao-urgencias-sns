"""LightGBM global (as cinco séries juntas), um modelo por horizonte.

    python -m src.lgbm                 # validação: histórico completo vs. desde 2022 (D6)
    python -m src.lgbm --test          # teste, com a configuração escolhida na validação

Walk-forward com reajuste mensal: as previsões de cada mês usam um modelo treinado
só com dias cujo valor já era conhecido na primeira origem desse mês.
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import pandas as pd

from src.baselines import seasonal_naive
from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGET_REGIONS, TARGETS, TEST, VALIDATION, actuals, in_period, score
from src.features import build_features, feature_columns
from src.tracking import record

PARAMS = {
    "objective": "l1",  # otimiza diretamente o MAE
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 42,
}
NUM_ROUNDS = 400
TRAIN_STARTS = {"full_history": pd.Timestamp("2016-11-01"), "since_2022": pd.Timestamp("2022-01-01")}


def forecast_horizon(
    table: pd.DataFrame,
    period,
    train_start: pd.Timestamp,
    params: dict | None = None,
    num_rounds: int = NUM_ROUNDS,
    refit_months: int = 1,
) -> pd.DataFrame:
    """Previsões walk-forward para um horizonte, com um novo treino a cada `refit_months` meses."""
    params = PARAMS if params is None else params
    features = feature_columns(table)
    to_predict = table[in_period(table["target_date"], period)]
    months = to_predict["target_date"].dt.to_period("M")
    block = (months - months.min()).apply(lambda offset: offset.n // refit_months)
    frames = []
    for _, rows in to_predict.groupby(block):
        first_origin = rows["origin"].min()
        train = table[
            (table["target_date"] <= first_origin)
            & (table["origin"] >= train_start)
            & table["y_rel"].notna()
        ]
        model = lgb.train(params, lgb.Dataset(train[features], train["y_rel"]), num_rounds)
        rows = rows.copy()
        rows["y_pred"] = rows["base"] + model.predict(rows[features])
        frames.append(rows)
    return pd.concat(frames)


def build_tables(
    daily: pd.DataFrame, target: str, regions: list[str] = TARGET_REGIONS
) -> dict[int, pd.DataFrame]:
    return {horizon: build_features(daily, target, horizon, regions) for horizon in HORIZONS}


def forecast(
    daily: pd.DataFrame,
    target: str,
    period,
    train_start: pd.Timestamp,
    tables: dict[int, pd.DataFrame] | None = None,
    **kwargs,
) -> pd.DataFrame:
    tables = build_tables(daily, target) if tables is None else tables
    frames = []
    for horizon, table in tables.items():
        result = forecast_horizon(table, period, train_start, **kwargs)
        result["horizon"] = horizon
        frames.append(result)
    forecasts = pd.concat(frames, ignore_index=True)
    forecasts["ars"] = forecasts["ars"].astype(str)
    return forecasts[FORECAST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="avaliar no teste em vez da validação")
    parser.add_argument("--train-start", choices=TRAIN_STARTS, default="full_history")
    args = parser.parse_args()

    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    period, period_name = (TEST, "teste") if args.test else (VALIDATION, "validação")
    starts = [args.train_start] if args.test else list(TRAIN_STARTS)

    for target in TARGETS:
        naive = score(seasonal_naive(actuals(daily, target)), period)
        print(f"\n{target} | seasonal_naive | {period_name}")
        print(naive)
        for start in starts:
            forecasts = forecast(daily, target, period, TRAIN_STARTS[start])
            if start == "full_history":
                record(forecasts, "lightgbm", target, args.test)
            table = score(forecasts, period)
            print(f"\n{target} | lightgbm ({start}) | {period_name}")
            print(table)
            gain = 1 - table.loc["Média", "1-7"] / naive.loc["Média", "1-7"]
            print(f"MAE médio 1-7: {table.loc['Média', '1-7']:.2f} vs. {naive.loc['Média', '1-7']:.2f} ({gain:+.1%})")


if __name__ == "__main__":
    main()
