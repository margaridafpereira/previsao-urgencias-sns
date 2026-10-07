# SNS Emergency Department Forecasting

Forecasts the **average waiting time** and the **number of emergency episodes** in Portuguese public hospitals (SNS), per region of mainland Portugal, up to 7 days ahead.

**Who it is for:**
- **Citizens**: know whether the coming days will be a peak.
- **Hospital management**: anticipate flu and heatwave peaks.

**Constraint:** zero cost. Public data only, free models and free tooling.

## Documentation

| Document | Content |
|---|---|
| [docs/data-sources.md](docs/data-sources.md) | Data sources, endpoints and fields |
| [docs/decisions.md](docs/decisions.md) | Decision log and the reasoning behind each decision |

## Project layout

```
src/ingest/        data collection scripts
src/data.py        loads the raw data into one daily table (date × region)
src/evaluation.py  evaluation protocol shared by every model (D8)
src/baselines.py   naive baselines, the bar every model has to beat
src/features.py    features for the tabular models, one table per horizon
src/lgbm.py        LightGBM, one global model per horizon, monthly walk-forward refit
src/tune_lgbm.py   random hyperparameter search on validation, logged to MLflow (mlflow.db)
src/lstm.py        LSTM in PyTorch, one global model for all 7 horizons, 3-monthly walk-forward refit
src/nhits.py       N-HiTS from neuralforecast, same protocol
src/ensemble.py    weighted average of saved LightGBM and LSTM forecasts (data/forecasts/)
src/intervals.py   80 % prediction intervals: quantile LightGBM + conformal calibration
notebooks/         analysis notebooks (01-eda: exploratory data analysis and key findings)
data/raw/          downloaded datasets (Parquet, versioned in Git)
tests/             tests
.github/workflows/ scheduled daily data collection
```

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python -m src.ingest.sns        # download the SNS datasets into data/raw/
python -m src.ingest.meteo      # download daily weather per region into data/raw/meteo.parquet
python -m src.baselines         # score the naive baselines (MAE per region and horizon)
python -m src.lgbm              # LightGBM on validation (a few minutes); --test for the test period
python -m src.lstm              # LSTM on validation (~7 min); --test for the test period
python -m src.nhits             # N-HiTS on validation (~15 min); --test for the test period
python -m src.ensemble          # combine saved forecasts (run lgbm and lstm, with and without --test, first)
python -m src.intervals         # prediction intervals (~25 min)
python -m src.tune_lgbm         # hyperparameter search (~30 min); browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db
python -m pytest -q             # run the tests
```

Run the commands from the repository root.

## Results so far

Test period (2025-01-01 onwards), mean absolute error over the five series, horizons 1 to 7 days. Lower is better.

| Model | Waiting time (min) | Episodes per day |
|---|---|---|
| Seasonal naive (same weekday last week) | 7.63 | 278 |
| LightGBM (D9) | 5.92 | 219 |
| LSTM (D11) | 6.25 | 216 |
| N-HiTS (D12) | 6.66 | 249 |
| **Average of LightGBM and LSTM** (D13) | **5.88** | **199** |

Chosen models: LightGBM alone for waiting time (the average gains only 0.7 %), and the 50/50 average for episodes (−8 %).

## Roadmap

| # | Step | Status |
|---|---|---|
| 1 | Ingest SNS and weather data, daily GitHub Action | done |
| 2 | Exploratory data analysis ([notebooks/01-eda.ipynb](notebooks/01-eda.ipynb)) | done |
| 3 | Evaluation protocol and naive baselines (D8) | done |
| 4 | Features: lags, rolling means, calendar, holidays, COVID flag, flu, temperature (D9) | done |
| 5 | LightGBM: −22 % MAE vs. seasonal naive on the test period (D9) | done |
| 5b | Tune LightGBM hyperparameters, runs in MLflow: no meaningful gain, defaults kept (D10) | done |
| 6 | LSTM written from scratch in PyTorch: beats naive, does not beat LightGBM (D11) | done |
| 7 | N-HiTS (neuralforecast): beats naive, last of the three models; TFT skipped (D12) | done |
| 8 | Ensemble of LightGBM and LSTM, final comparison table (D13) | done |
| 8b | 80 % prediction intervals: quantile LightGBM with conformal calibration (D14) | next |
| 9 | Daily retraining and forecast in GitHub Actions, demo on Hugging Face Spaces | |

## Stack (all free)

| Need | Tool |
|---|---|
| Language | Python 3.12+ |
| Data | pandas, Parquet |
| Classical ML | scikit-learn, LightGBM/XGBoost |
| Deep learning | PyTorch (LSTM; then N-HiTS/TFT via `neuralforecast` or `darts`) |
| GPU training | Kaggle Notebooks / Google Colab |
| Experiment tracking | MLflow (local) |
| Automation | GitHub Actions (daily collection and retraining) |
| Demo | Hugging Face Spaces (Gradio) |
