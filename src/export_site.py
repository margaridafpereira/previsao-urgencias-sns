"""Prepara os dados da página pública (GitHub Pages) a partir da última previsão (D23).

    python -m src.export_site                # site/data/forecast.json, a partir de data/predictions/latest.parquet
    python -m src.export_site --accuracy     # site/data/accuracy.json, a partir das previsões de teste (D18)

A página (site/index.html) é estática: só lê estes dois ficheiros JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data import REGIONS, load_daily
from src.evaluation import FORECASTS_DIR, TARGETS, TEST, in_period
from src.predict import MODELS, PREDICTIONS_DIR
from src.regions import region_status

SITE_DATA = Path(__file__).resolve().parents[1] / "site" / "data"
HISTORY_DAYS = 42
NORMAL_YEARS = (2023, 2024, 2025)  # regime pós-COVID (D6)
NORMAL_WINDOW_DAYS = 7

LABELS = {
    "wait_minutes": {"label": "Tempo médio de espera", "unit": "min", "decimals": 0},
    "episodes": {"label": "Episódios de urgência por dia", "unit": "episódios", "decimals": 0},
}


def _normal(series: pd.Series, dates: pd.DatetimeIndex) -> list[float | None]:
    """Média do mesmo período do ano (± 7 dias) em 2023-2025: o 'normal para a época'."""
    values = []
    for date in dates:
        window = []
        for year in NORMAL_YEARS:
            try:
                center = date.replace(year=year)
            except ValueError:  # 29 de fevereiro
                center = date.replace(year=year, day=28)
            window.append(series[center - pd.Timedelta(days=NORMAL_WINDOW_DAYS) : center + pd.Timedelta(days=NORMAL_WINDOW_DAYS)])
        joined = pd.concat(window).dropna()
        values.append(round(float(joined.mean()), 1) if len(joined) else None)
    return values


def _records(frame: pd.DataFrame, columns: dict[str, str]) -> list[dict]:
    out = []
    for row in frame.itertuples(index=False):
        record = {}
        for key, column in columns.items():
            value = getattr(row, column)
            if isinstance(value, pd.Timestamp):
                value = value.date().isoformat()
            elif isinstance(value, float):
                value = None if pd.isna(value) else round(value, 1)
            record[key] = value
        out.append(record)
    return out


def build_forecast_json(predictions: pd.DataFrame, daily: pd.DataFrame) -> dict:
    run_at = predictions["run_at"].iloc[0]
    last_official = predictions["origin"].max()
    payload = {
        "run_at": run_at,
        "last_official_day": last_official.date().isoformat(),
        "delay_days": int(predictions["delay_days"].max()),
        "targets": LABELS,
        "regions": REGIONS,
        "models": MODELS,
        "series": {},
        "inactive": {},
    }
    for target in TARGETS:
        status = region_status(daily, target)
        last_any = daily.dropna(subset=[target]).groupby("ars")["periodo"].max()
        payload["inactive"][target] = [
            {"ars": region, "last_value": last_any[region].date().isoformat() if region in last_any else None}
            for region in status.index[~status["active"]]
        ]
        wide = daily.pivot(index="periodo", columns="ars", values=target).asfreq("D")
        payload["series"][target] = {}
        for region, part in predictions[predictions["target"] == target].groupby("ars"):
            part = part.sort_values("target_date")
            history = wide[region].loc[last_official - pd.Timedelta(days=HISTORY_DAYS - 1) : last_official]
            observed = pd.DataFrame({"date": history.index, "value": history.to_numpy()})
            part = part.assign(normal=_normal(wide[region], pd.DatetimeIndex(part["target_date"])))
            payload["series"][target][region] = {
                "observed": _records(observed, {"date": "date", "value": "value"}),
                "forecast": _records(part, {
                    "date": "target_date", "kind": "kind", "horizon": "horizon",
                    "y": "y_pred", "low": "y_low", "high": "y_high", "normal": "normal",
                    **{model: model for model in MODELS},
                }),
            }
    return payload


def build_accuracy_json() -> dict:
    """Erro médio da média dos 4 modelos (D18) no teste, por região e horizonte."""
    keys = ["origin", "target_date", "horizon", "ars"]
    payload = {"period": f"{TEST[0].date().isoformat()} em diante", "targets": {}}
    for target in TARGETS:
        merged = None
        for model in MODELS:
            frame = pd.read_parquet(FORECASTS_DIR / f"{model}_{target}_test.parquet")
            frame = frame.rename(columns={"y_pred": model})
            merged = frame if merged is None else merged.merge(frame[keys + [model]], on=keys)
        merged = merged[in_period(merged["target_date"], TEST)].dropna(subset=["y_true"])
        error = (merged[MODELS].mean(axis=1) - merged["y_true"]).abs()
        by_region = {}
        for region, group in error.groupby(merged["ars"]):
            horizons = merged.loc[group.index, "horizon"]
            by_region[region] = {
                "1-7": round(float(group[horizons <= 7].mean()), 1),
                "8-14": round(float(group[horizons > 7].mean()), 1),
                "by_horizon": {int(h): round(float(v), 1) for h, v in group.groupby(horizons).mean().items()},
            }
        payload["targets"][target] = by_region
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", action="store_true", help="gera site/data/accuracy.json")
    args = parser.parse_args()
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    if args.accuracy:
        path, payload = SITE_DATA / "accuracy.json", build_accuracy_json()
    else:
        predictions = pd.read_parquet(PREDICTIONS_DIR / "latest.parquet")
        path, payload = SITE_DATA / "forecast.json", build_forecast_json(predictions, load_daily())
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Gravado {path}")


if __name__ == "__main__":
    main()
