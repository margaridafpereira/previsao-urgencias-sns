"""Registo de todas as avaliações no MLflow, numa única experiência (D15).

    python -m src.tracking             # regista de novo todas as previsões gravadas em data/forecasts/
    mlflow ui --backend-store-uri sqlite:///mlflow.db

Cada modelo chama `record()` no fim da avaliação: grava as previsões
(data/forecasts/) e cria uma execução no MLflow com o MAE por horizonte e por região.
"""

from __future__ import annotations

import os

import pandas as pd

from src.evaluation import FORECASTS_DIR, TARGETS, TEST, VALIDATION, save_forecasts, score

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "model-comparison"


def log_evaluation(forecasts: pd.DataFrame, model: str, target: str, test: bool, params: dict | None = None) -> None:
    # Importado só aqui: a previsão diária (src/predict.py) não precisa do MLflow instalado.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    period = TEST if test else VALIDATION
    period_name = "test" if test else "validation"
    table = score(forecasts, period)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"{model}-{target}-{period_name}"):
        mlflow.set_tags({"model": model, "target": target, "period": period_name})
        if params:
            mlflow.log_params(params)
        mlflow.log_metric("mae_1_7", table.loc["Média", "1-7"])
        if "8-14" in table.columns:
            mlflow.log_metric("mae_8_14", table.loc["Média", "8-14"])
        for horizon in [column for column in table.columns if isinstance(column, int)]:
            mlflow.log_metric(f"mae_h{horizon}", table.loc["Média", horizon])
        for region, value in table["1-7"].drop("Média").items():
            mlflow.log_metric(f"mae_{region.replace('ARS ', '').replace(' ', '_')}", value)


def record(forecasts: pd.DataFrame, model: str, target: str, test: bool, params: dict | None = None) -> None:
    """Grava as previsões e regista a avaliação no MLflow."""
    save_forecasts(forecasts, model, target, test)
    log_evaluation(forecasts, model, target, test, params)


def main() -> None:
    for path in sorted(FORECASTS_DIR.glob("*.parquet")):
        # Nome do ficheiro: <modelo>_<alvo>_<período>; o modelo pode ter "_" (seasonal_naive).
        target = next((t for t in TARGETS if f"_{t}_" in path.stem), None)
        if target is None or "intervals" in path.stem:
            continue
        model, period_name = path.stem.split(f"_{target}_")
        log_evaluation(pd.read_parquet(path), model, target, period_name == "test")
        print(f"registado {path.name}")


if __name__ == "__main__":
    main()
