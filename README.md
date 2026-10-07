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
python -m pytest -q             # run the tests
```

Run the commands from the repository root.

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
