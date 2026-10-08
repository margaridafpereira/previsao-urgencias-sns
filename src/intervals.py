"""Intervalos de previsão a 80 % para o LightGBM (D14).

    python -m src.intervals

1. Regressão por quantis: dois LightGBM extra por horizonte, para os quantis 10 % e 90 %,
   com as mesmas features e o mesmo walk-forward do modelo pontual (D9).
2. Calibração conformal (CQR): na validação mede-se quanto os intervalos falham
   e alarga-se (ou estreita-se) cada horizonte por essa margem, aplicada sem
   alterações ao teste. Assim a cobertura fica perto dos 80 % prometidos.

Grava as previsões com `y_low` e `y_high` em data/forecasts/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_daily
from src.evaluation import FORECASTS_DIR, TARGETS, TEST, VALIDATION, in_period
from src.lgbm import PARAMS, TRAIN_STARTS, build_tables, forecast

COVERAGE = 0.8
QUANTILES = {"y_low": (1 - COVERAGE) / 2, "y_high": 1 - (1 - COVERAGE) / 2}
KEYS = ["origin", "target_date", "horizon", "ars"]

# Margens conformal por alvo e horizonte, usadas pela previsão diária (src/predict.py).
MARGINS_PATH = Path(__file__).resolve().parents[1] / "data" / "interval_margins.json"


def save_margins(margins: dict[str, pd.Series]) -> None:
    payload = {target: {str(h): round(float(m), 3) for h, m in series.items()} for target, series in margins.items()}
    MARGINS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_margins() -> dict[str, pd.Series]:
    payload = json.loads(MARGINS_PATH.read_text(encoding="utf-8"))
    return {target: pd.Series({int(h): m for h, m in values.items()}, name="margin") for target, values in payload.items()}


def quantile_forecasts(daily: pd.DataFrame, target: str, period, tables) -> pd.DataFrame:
    """Previsões dos quantis inferior e superior, numa só tabela."""
    result = None
    for column, alpha in QUANTILES.items():
        params = {**PARAMS, "objective": "quantile", "alpha": alpha}
        part = forecast(daily, target, period, TRAIN_STARTS["full_history"], tables=tables, params=params)
        part = part.rename(columns={"y_pred": column})
        result = part if result is None else result.merge(part[KEYS + [column]], on=KEYS)
    # Os dois modelos são independentes; garantir que o inferior não fica acima do superior.
    low, high = result[["y_low", "y_high"]].min(axis=1), result[["y_low", "y_high"]].max(axis=1)
    result["y_low"], result["y_high"] = low, high
    return result


def conformal_margins(validation: pd.DataFrame) -> pd.Series:
    """Margem por horizonte para que a cobertura na validação seja COVERAGE (CQR)."""
    scored = validation[in_period(validation["target_date"], VALIDATION)].dropna(subset=["y_true"])
    errors = np.maximum(scored["y_low"] - scored["y_true"], scored["y_true"] - scored["y_high"])

    def margin(group: pd.Series) -> float:
        n = len(group)
        return float(np.quantile(group, min(1.0, COVERAGE * (1 + 1 / n))))

    return errors.groupby(scored["horizon"]).apply(margin).rename("margin")


def calibrate(forecasts: pd.DataFrame, margins: pd.Series) -> pd.DataFrame:
    result = forecasts.copy()
    margin = result["horizon"].map(margins)
    result["y_low"] = result["y_low"] - margin
    result["y_high"] = result["y_high"] + margin
    return result


def interval_report(forecasts: pd.DataFrame, period) -> pd.DataFrame:
    """Cobertura (% de dias dentro do intervalo) e largura média, por região e no total."""
    scored = forecasts[in_period(forecasts["target_date"], period)].dropna(subset=["y_true"])
    inside = scored["y_true"].between(scored["y_low"], scored["y_high"])
    width = scored["y_high"] - scored["y_low"]
    report = pd.DataFrame({
        "cobertura %": inside.groupby(scored["ars"]).mean() * 100,
        "largura": width.groupby(scored["ars"]).mean(),
    })
    report.loc["Total"] = [inside.mean() * 100, width.mean()]
    by_horizon = pd.DataFrame({
        "cobertura %": inside.groupby(scored["horizon"]).mean() * 100,
        "largura": width.groupby(scored["horizon"]).mean(),
    })
    by_horizon.index = [f"h={h}" for h in by_horizon.index]
    return pd.concat([report, by_horizon])


def main() -> None:
    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    all_margins = {}
    for target in TARGETS:
        tables = build_tables(daily, target)
        validation = quantile_forecasts(daily, target, VALIDATION, tables)
        test = quantile_forecasts(daily, target, TEST, tables)
        margins = conformal_margins(validation)
        all_margins[target] = margins
        calibrated = calibrate(test, margins)
        calibrated.to_parquet(FORECASTS_DIR / f"lightgbm_intervals_{target}_test.parquet", index=False)

        print(f"\n{target}: margem conformal por horizonte (da validação)")
        print(margins.to_frame().T.to_string())
        print(f"\n{target}: teste, intervalo de {COVERAGE:.0%}, antes e depois da calibração")
        print(pd.concat(
            {"quantis": interval_report(test, TEST), "calibrado": interval_report(calibrated, TEST)}, axis=1
        ).to_string())

    save_margins(all_margins)


if __name__ == "__main__":
    main()
