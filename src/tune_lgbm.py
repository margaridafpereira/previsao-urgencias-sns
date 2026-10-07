"""Pesquisa aleatória dos hiperparâmetros do LightGBM no período de validação (D10).

    python -m src.tune_lgbm            # 24 configurações, registadas no MLflow
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Para ser mais rápida, a pesquisa retreina de 3 em 3 meses em vez de todos os meses
e otimiza só o tempo de espera, o alvo principal. A melhor configuração é depois
confirmada com o reajuste mensal, nos dois alvos.
"""

from __future__ import annotations

import argparse
import random

import mlflow
import pandas as pd

from src.data import load_daily
from src.evaluation import VALIDATION, score
from src.lgbm import NUM_ROUNDS, PARAMS, TRAIN_STARTS, build_tables, forecast

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "lightgbm-tuning"
TARGET = "wait_minutes"
REFIT_MONTHS = 3

SEARCH_SPACE = {
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [7, 15, 31, 63],
    "min_data_in_leaf": [20, 40, 80, 160],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.7, 0.8, 1.0],
    "lambda_l2": [0.0, 1.0, 10.0],
    "num_rounds": [200, 400, 800],
}


def sample_configs(n_trials: int, seed: int = 0) -> list[dict]:
    """A configuração atual primeiro, depois `n_trials - 1` amostras aleatórias sem repetições."""
    rng = random.Random(seed)
    current = {key: PARAMS.get(key, 0.0) for key in SEARCH_SPACE if key != "num_rounds"}
    configs = [{**current, "num_rounds": NUM_ROUNDS}]
    while len(configs) < n_trials:
        config = {key: rng.choice(values) for key, values in SEARCH_SPACE.items()}
        if config not in configs:
            configs.append(config)
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=24)
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    daily = load_daily()
    tables = build_tables(daily, TARGET)

    results = []
    for trial, config in enumerate(sample_configs(args.trials)):
        config = dict(config)
        num_rounds = config.pop("num_rounds")
        params = {**PARAMS, **config}
        forecasts = forecast(
            daily, TARGET, VALIDATION, TRAIN_STARTS["full_history"],
            tables=tables, params=params, num_rounds=num_rounds, refit_months=REFIT_MONTHS,
        )
        table = score(forecasts, VALIDATION)
        mae = table.loc["Média", "1-7"]
        with mlflow.start_run(run_name=f"trial-{trial:02d}"):
            mlflow.log_params({**config, "num_rounds": num_rounds, "target": TARGET, "refit_months": REFIT_MONTHS})
            mlflow.log_metric("val_mae_1_7", mae)
            for horizon in (1, 7):
                mlflow.log_metric(f"val_mae_h{horizon}", table.loc["Média", horizon])
        results.append({**config, "num_rounds": num_rounds, "val_mae_1_7": mae})
        print(f"trial {trial:02d}: MAE {mae:.3f}  {config}, num_rounds={num_rounds}", flush=True)

    ranking = pd.DataFrame(results).sort_values("val_mae_1_7")
    pd.set_option("display.width", 160, "display.max_columns", None)
    print("\nMelhores configurações:")
    print(ranking.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
