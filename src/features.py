"""Features para os modelos tabulares (LightGBM), uma tabela por horizonte.

Cada linha é uma (origem `t`, região) e prevê o dia `t + h`. Só se usa
informação conhecida na origem, exceto o calendário do dia previsto, que é
conhecido de antemão (decisão D9 em docs/decisions.md).

O alvo e os lags são expressos em relação a `base`, a média dos últimos 28 dias,
para que as árvores não tenham de extrapolar níveis nunca vistos (o tempo de
espera subiu cerca de 30 % depois da COVID).
"""

from __future__ import annotations

import holidays
import pandas as pd

from src.evaluation import TARGET_REGIONS

COVID = (pd.Timestamp("2020-03-01"), pd.Timestamp("2021-03-31"))  # D6
BASE_WINDOW = 28
N_LAGS = 14
WEATHER = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"]


def _wide(daily: pd.DataFrame, column: str) -> pd.DataFrame:
    """Uma coluna por região, um dia por linha, lacunas preenchidas com o último valor."""
    wide = daily.pivot(index="periodo", columns="ars", values=column).asfreq("D").ffill()
    if column in WEATHER:
        # Portugal Continental não tem estação própria (D5): usa a média das regiões.
        wide["Portugal Continental"] = wide.drop(columns="Portugal Continental", errors="ignore").mean(axis=1)
    return wide[TARGET_REGIONS]


def _long(wide: pd.DataFrame, name: str) -> pd.Series:
    return wide.stack(future_stack=True).rename(name)


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendário de cada dia: conhecido de antemão, por isso pode ser do dia previsto."""
    pt_holidays = holidays.PT(years=range(dates.min().year - 1, dates.max().year + 2))
    is_holiday = pd.Series([day in pt_holidays for day in dates.date], index=dates)
    day_before = pd.Series([(day - pd.Timedelta(days=1)).date() in pt_holidays for day in dates], index=dates)
    month_day = dates.month * 100 + dates.day
    return pd.DataFrame(
        {
            "dow": dates.dayofweek,
            "day_of_year": dates.dayofyear,
            "is_holiday": is_holiday.astype(int),
            "after_holiday": day_before.astype(int),
            "christmas_window": ((month_day >= 1224) | (month_day <= 105)).astype(int),
        },
        index=dates,
    )


def build_features(daily: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """Tabela de treino/previsão para um horizonte.

    Colunas fixas: origin, target_date, ars, base, y_true, y_rel (= y_true - base).
    As restantes são features.
    """
    y = _wide(daily, target)
    observed = daily.pivot(index="periodo", columns="ars", values=target).asfreq("D")[TARGET_REGIONS]
    base = y.rolling(BASE_WINDOW, min_periods=7).mean()

    columns: dict[str, pd.Series] = {"base": _long(base, "base")}
    for lag in range(N_LAGS):
        columns[f"lag_{lag}"] = _long(y.shift(lag) - base, f"lag_{lag}")
    for weeks in (1, 2):
        lag = 7 * weeks - horizon  # mesmo dia da semana do dia previsto, há `weeks` semanas
        columns[f"same_weekday_{weeks}w"] = _long(y.shift(lag) - base, f"same_weekday_{weeks}w")
    columns["mean_7"] = _long(y.rolling(7).mean() - base, "mean_7")
    columns["std_28"] = _long(y.rolling(BASE_WINDOW).std(), "std_28")

    if target != "episodes":
        episodes = _wide(daily, "episodes")
        columns["episodes_ratio"] = _long(episodes / episodes.rolling(BASE_WINDOW).mean(), "episodes_ratio")
    respiratory = _wide(daily, "respiratory_pct")
    columns["respiratory_pct"] = _long(respiratory, "respiratory_pct")
    columns["respiratory_change_7"] = _long(respiratory - respiratory.shift(7), "respiratory_change_7")
    columns["admission_pct"] = _long(_wide(daily, "admission_pct"), "admission_pct")
    for variable in WEATHER:
        columns[variable] = _long(_wide(daily, variable).rolling(3).mean(), variable)

    table = pd.concat(columns.values(), axis=1)
    table.index.names = ["origin", "ars"]
    table = table.reset_index()
    table["target_date"] = table["origin"] + pd.Timedelta(days=horizon)
    table["is_covid"] = table["origin"].between(*COVID).astype(int)

    cal = calendar(pd.DatetimeIndex(table["target_date"].unique()).sort_values())
    table = table.join(cal, on="target_date")

    truth = _long(observed, "y_true")
    truth.index.names = ["target_date", "ars"]
    table = table.join(truth, on=["target_date", "ars"])
    table["y_rel"] = table["y_true"] - table["base"]
    table["ars"] = pd.Categorical(table["ars"], categories=TARGET_REGIONS)
    return table.dropna(subset=["base"]).reset_index(drop=True)


ID_COLUMNS = ["origin", "target_date", "y_true", "y_rel"]


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in ID_COLUMNS]
