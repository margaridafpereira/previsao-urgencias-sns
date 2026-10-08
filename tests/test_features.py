import pandas as pd
import pytest

from src.data import RAW_DIR, load_daily
from src.features import build_features, calendar, feature_columns

REQUIRED = ["monitorizacao-sazonal-csh", "atividade-sindrome-gripal-csh", "meteo"]
needs_data = pytest.mark.skipif(
    not all((RAW_DIR / f"{name}.parquet").exists() for name in REQUIRED),
    reason="dados ainda não descarregados",
)


def test_calendar_flags_holidays_and_christmas_window():
    cal = calendar(pd.DatetimeIndex(["2024-12-25", "2024-12-26", "2025-01-06", "2025-04-25"]))
    assert cal["is_holiday"].tolist() == [1, 0, 0, 1]
    assert cal["after_holiday"].tolist() == [0, 1, 0, 0]
    assert cal["christmas_window"].tolist() == [1, 1, 0, 0]


@needs_data
def test_features_do_not_use_data_after_the_origin():
    daily = load_daily()
    origin = pd.Timestamp("2024-01-15")
    tampered = daily.copy()
    after = tampered["periodo"] > origin
    numeric = tampered.select_dtypes("number").columns
    tampered.loc[after, numeric] = tampered.loc[after, numeric] * 10

    for horizon in (1, 7, 8, 14):
        original = build_features(daily, "wait_minutes", horizon)
        changed = build_features(tampered, "wait_minutes", horizon)
        columns = feature_columns(original)
        at_origin = lambda table: table.loc[table["origin"] == origin, columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(at_origin(original), at_origin(changed))
