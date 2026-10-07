"""LSTM escrita de raiz em PyTorch: um modelo global que prevê os 7 horizontes de uma vez (D11).

    python -m src.lstm                 # validação
    python -m src.lstm --test          # teste

Para cada (origem `t`, região) a rede lê as últimas WINDOW semanas de dados diários
até `t` e o calendário dos 7 dias seguintes, e devolve o desvio relativo de cada
dia em relação a `base` (a média dos últimos 28 dias), tal como o LightGBM (D9).

Walk-forward com reajuste trimestral (treinar uma rede é mais lento do que uma árvore).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from src.baselines import seasonal_naive
from src.data import load_daily
from src.evaluation import FORECAST_COLUMNS, HORIZONS, TARGETS, TARGET_REGIONS, TEST, VALIDATION, actuals, save_forecasts, score
from src.features import BASE_WINDOW, COVID, _wide, calendar

WINDOW = 56
N_HORIZONS = len(HORIZONS)
REFIT_MONTHS = 3
HIDDEN = 64
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 6
LEARNING_RATE = 1e-3
EARLY_STOP_FRACTION = 0.1  # as origens mais recentes do treino servem para parar cedo


class Dataset:
    """Arrays por dia e região, mais o calendário, prontos a cortar em janelas."""

    def __init__(self, daily: pd.DataFrame, target: str):
        y = _wide(daily, target)
        self.dates = y.index
        self.observed = daily.pivot(index="periodo", columns="ars", values=target).asfreq("D")[TARGET_REGIONS]
        self.observed = self.observed.reindex(self.dates).to_numpy()
        self.y = y.to_numpy()
        self.base = y.rolling(BASE_WINDOW, min_periods=7).mean().to_numpy()

        # Covariáveis diárias (dia, região, canal), já em escalas próximas de 0-1.
        channels = []
        if target != "episodes":
            episodes = _wide(daily, "episodes")
            channels.append((episodes / episodes.rolling(BASE_WINDOW).mean() - 1).to_numpy())
        channels.append(_wide(daily, "respiratory_pct").to_numpy() / 10)
        channels.append(_wide(daily, "admission_pct").to_numpy() / 10)
        channels.append((_wide(daily, "temperature_2m_mean").to_numpy() - 15) / 8)
        channels.append(np.log1p(_wide(daily, "precipitation_sum").to_numpy()) / 3)
        cal = calendar(self.dates)
        per_day = np.stack(
            [
                np.sin(2 * np.pi * cal["dow"] / 7),
                np.cos(2 * np.pi * cal["dow"] / 7),
                np.sin(2 * np.pi * cal["day_of_year"] / 365.25),
                np.cos(2 * np.pi * cal["day_of_year"] / 365.25),
                cal["is_holiday"],
                cal["after_holiday"],
                cal["christmas_window"],
                ((self.dates >= COVID[0]) & (self.dates <= COVID[1])).astype(float),
            ],
            axis=1,
        )
        n_regions = len(TARGET_REGIONS)
        channels += [np.repeat(per_day[:, [k]], n_regions, axis=1) for k in range(per_day.shape[1])]
        self.covariates = np.nan_to_num(np.stack(channels, axis=2)).astype(np.float32)

        # Calendário dos dias previstos: dia da semana (one-hot), feriado, pós-feriado, Natal.
        dow = np.eye(7)[cal["dow"].to_numpy()]
        self.future_calendar = np.concatenate(
            [dow, cal[["is_holiday", "after_holiday", "christmas_window"]].to_numpy()], axis=1
        ).astype(np.float32)

    @property
    def n_channels(self) -> int:
        return 1 + self.covariates.shape[2]

    def first_valid_origin(self) -> int:
        return WINDOW + BASE_WINDOW

    def samples(self, origins: np.ndarray):
        """Entradas e alvos para as origens dadas (índices de dia), todas as regiões.

        Devolve x (N, WINDOW, canais), future (N, 7 * 10), region (N,), y (N, 7), base (N,).
        """
        n_regions = len(TARGET_REGIONS)
        base = self.base[origins]  # (O, R)
        windows = sliding_window_view(self.y, WINDOW, axis=0)[origins - WINDOW + 1]  # (O, R, W)
        y_in = windows / base[:, :, None] - 1
        cov = sliding_window_view(self.covariates, WINDOW, axis=0)[origins - WINDOW + 1]  # (O, R, C, W)
        x = np.concatenate([y_in[:, :, None, :], cov], axis=2).transpose(0, 1, 3, 2)  # (O, R, W, C)

        padded = np.vstack([self.observed, np.full((N_HORIZONS, n_regions), np.nan)])
        future_y = sliding_window_view(padded, N_HORIZONS, axis=0)[origins + 1]  # (O, R, 7)
        target = future_y / base[:, :, None] - 1

        padded_cal = np.vstack([self.future_calendar, np.zeros((N_HORIZONS, self.future_calendar.shape[1]))])
        future = sliding_window_view(padded_cal, N_HORIZONS, axis=0)[origins + 1]  # (O, 10, 7)
        future = np.repeat(future.reshape(len(origins), 1, -1), n_regions, axis=1)

        region = np.broadcast_to(np.arange(n_regions), (len(origins), n_regions))
        flat = lambda array: array.reshape(len(origins) * n_regions, *array.shape[2:])
        return (
            torch.tensor(np.nan_to_num(flat(x)), dtype=torch.float32),
            torch.tensor(flat(future), dtype=torch.float32),
            torch.tensor(flat(region).copy(), dtype=torch.long),
            torch.tensor(flat(target), dtype=torch.float32),
            flat(base),
        )


class Net(nn.Module):
    def __init__(self, n_channels: int, n_future: int, n_regions: int):
        super().__init__()
        self.region = nn.Embedding(n_regions, 4)
        self.lstm = nn.LSTM(n_channels + 4, HIDDEN, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + n_future + 4, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, N_HORIZONS),
        )

    def forward(self, x, future, region):
        emb = self.region(region)
        steps = torch.cat([x, emb[:, None, :].expand(-1, x.shape[1], -1)], dim=2)
        _, (hidden, _) = self.lstm(steps)
        return self.head(torch.cat([hidden[-1], future, emb], dim=1))


def masked_l1(prediction, target):
    mask = ~torch.isnan(target)
    return (prediction[mask] - target[mask]).abs().mean()


def train(data: Dataset, origins: np.ndarray, seed: int = 42) -> Net:
    torch.manual_seed(seed)
    n_stop = max(1, int(len(origins) * EARLY_STOP_FRACTION))
    fit_x, fit_f, fit_r, fit_y, _ = data.samples(origins[:-n_stop])
    stop_x, stop_f, stop_r, stop_y, _ = data.samples(origins[-n_stop:])

    model = Net(data.n_channels, fit_f.shape[1], len(TARGET_REGIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    generator = torch.Generator().manual_seed(seed)
    for _ in range(MAX_EPOCHS):
        model.train()
        for batch in torch.randperm(len(fit_x), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = masked_l1(model(fit_x[batch], fit_f[batch], fit_r[batch]), fit_y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_loss = masked_l1(model(stop_x, stop_f, stop_r), stop_y).item()
        if stop_loss < best_loss:
            best_loss, best_state, bad_epochs = stop_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def forecast(daily: pd.DataFrame, target: str, period) -> pd.DataFrame:
    data = Dataset(daily, target)
    dates = data.dates
    start, end = period
    end = dates[-1] if end is None else end
    # Origens cujas previsões caem no período, agrupadas em blocos de REFIT_MONTHS meses.
    origin_idx = np.flatnonzero((dates >= start - pd.Timedelta(days=N_HORIZONS)) & (dates <= end))
    months = pd.PeriodIndex(dates[origin_idx], freq="M")
    block = (months - months.min()).map(lambda offset: offset.n // REFIT_MONTHS)

    frames = []
    for block_id in np.unique(block):
        to_predict = origin_idx[block == block_id]
        first = to_predict[0]
        # Treino: só origens cujos 7 dias previstos já eram conhecidos na primeira origem do bloco.
        train_origins = np.arange(data.first_valid_origin(), first - N_HORIZONS + 1)
        started = time.time()
        model = train(data, train_origins)
        x, future, region, y_rel, base = data.samples(to_predict)
        with torch.no_grad():
            predicted = model(x, future, region).numpy()
        n_regions = len(TARGET_REGIONS)
        origins = np.repeat(dates[to_predict], n_regions)
        for h in range(N_HORIZONS):
            frames.append(pd.DataFrame({
                "origin": origins,
                "target_date": origins + pd.Timedelta(days=h + 1),
                "horizon": h + 1,
                "ars": np.tile(TARGET_REGIONS, len(to_predict)),
                "y_true": base * (1 + y_rel[:, h].numpy()),
                "y_pred": base * (1 + predicted[:, h]),
            }))
        print(f"  bloco {dates[first].date()}: treino {time.time() - started:.0f}s", flush=True)
    return pd.concat(frames, ignore_index=True)[FORECAST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="avaliar no teste em vez da validação")
    args = parser.parse_args()

    pd.set_option("display.precision", 1, "display.width", 140, "display.max_columns", None)
    daily = load_daily()
    period, period_name = (TEST, "teste") if args.test else (VALIDATION, "validação")
    for target in TARGETS:
        naive = score(seasonal_naive(actuals(daily, target)), period)
        forecasts = forecast(daily, target, period)
        save_forecasts(forecasts, "lstm", target, args.test)
        table = score(forecasts, period)
        print(f"\n{target} | lstm | {period_name}")
        print(table)
        gain = 1 - table.loc["Média", "1-7"] / naive.loc["Média", "1-7"]
        print(f"MAE médio 1-7: {table.loc['Média', '1-7']:.2f} vs. seasonal naive {naive.loc['Média', '1-7']:.2f} ({gain:+.1%})")


if __name__ == "__main__":
    main()
