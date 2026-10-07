import numpy as np
import pandas as pd
import pytest
import torch

from src.data import RAW_DIR, load_daily
from src.lstm import WINDOW, Dataset

REQUIRED = ["monitorizacao-sazonal-csh", "atividade-sindrome-gripal-csh", "meteo"]
needs_data = pytest.mark.skipif(
    not all((RAW_DIR / f"{name}.parquet").exists() for name in REQUIRED),
    reason="dados ainda não descarregados",
)


@needs_data
def test_inputs_do_not_use_data_after_the_origin():
    daily = load_daily()
    origin = pd.Timestamp("2024-01-15")
    tampered = daily.copy()
    after = tampered["periodo"] > origin
    numeric = tampered.select_dtypes("number").columns
    tampered.loc[after, numeric] = tampered.loc[after, numeric] * 10

    original, changed = Dataset(daily, "wait_minutes"), Dataset(tampered, "wait_minutes")
    index = np.array([original.dates.get_loc(origin)])
    x, future, _, y, _ = original.samples(index)
    x2, future2, _, y2, _ = changed.samples(index)
    assert x.shape[1] == WINDOW
    assert torch.equal(x, x2)
    assert torch.equal(future, future2)  # o calendário é conhecido de antemão
    assert not torch.allclose(y, y2, equal_nan=True)  # o alvo, esse, muda
