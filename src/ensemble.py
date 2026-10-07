"""Média ponderada das previsões do LightGBM e da LSTM (D13).

    python -m src.ensemble

Lê as previsões gravadas em data/forecasts/ (correr antes `python -m src.lgbm`
e `python -m src.lstm`, com e sem --test). O peso é escolhido na validação e
aplicado sem alterações ao teste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import FORECAST_COLUMNS, TARGETS, TEST, VALIDATION, load_forecasts, score

MODELS = ("lightgbm", "lstm")
WEIGHTS = np.round(np.arange(0, 1.01, 0.1), 1)  # peso do LightGBM; o resto vai para a LSTM
KEYS = ["origin", "target_date", "horizon", "ars"]


def combine(first: pd.DataFrame, second: pd.DataFrame, weight: float) -> pd.DataFrame:
    """weight * first + (1 - weight) * second, nas previsões que os dois modelos têm em comum."""
    merged = first.merge(second[KEYS + ["y_pred"]], on=KEYS, suffixes=("", "_second"), validate="one_to_one")
    merged["y_pred"] = weight * merged["y_pred"] + (1 - weight) * merged["y_pred_second"]
    return merged[FORECAST_COLUMNS]


def mae(forecasts: pd.DataFrame, period) -> float:
    return score(forecasts, period).loc["Média", "1-7"]


def main() -> None:
    pd.set_option("display.precision", 2, "display.width", 140, "display.max_columns", None)
    for target in TARGETS:
        validation = {model: load_forecasts(model, target, test=False) for model in MODELS}
        test = {model: load_forecasts(model, target, test=True) for model in MODELS}

        curve = pd.Series(
            {weight: mae(combine(*validation.values(), weight), VALIDATION) for weight in WEIGHTS},
            name="MAE validação",
        )
        best = curve.idxmin()
        print(f"\n{target}: MAE na validação por peso do LightGBM")
        print(curve.to_frame().T.to_string())

        rows = {
            "lightgbm": test["lightgbm"],
            "lstm": test["lstm"],
            "média 50/50": combine(*test.values(), 0.5),
            f"ensemble (peso {best})": combine(*test.values(), best),
        }
        summary = pd.DataFrame({
            name: score(forecasts, TEST).loc["Média", [1, 7, "1-7"]] for name, forecasts in rows.items()
        }).T
        print(f"\n{target}: teste, MAE médio por horizonte")
        print(summary.to_string())


if __name__ == "__main__":
    main()
