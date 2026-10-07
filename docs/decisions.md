# Decision log

Each entry records the context, the decision and its consequences. Entries are never deleted: if a decision changes, a new entry supersedes the old one.

---

## D1: Forecast per region and per day, not per hospital and per hour

**Context.** The original idea was to forecast each hospital's waiting time for the next few hours. Checking the sources showed that:
- real-time waiting times per hospital (`tempos.min-saude.pt`) exist only in a Power BI report, with no open API;
- the Transparency Portal has **10 years of daily data per region**, including the average time from triage to the first medical observation.

**Decision.** The target is the **daily** average waiting time, for **6 series** (5 regions plus mainland Portugal), with a **1 to 7 day** horizon.

**Consequences.**
- (+) Long history from day one, without months of data collection.
- (+) Official, stable data, with no scraping.
- (−) Coarser than "hospital X in two hours".
- Monthly per-hospital forecasting (datasets 1.3 and 1.4 in [data-sources.md](data-sources.md)) remains an optional extension.

---

## D2: Naive baseline first, then classical ML, then deep learning

**Context.** Without a baseline there is no way to tell whether deep learning adds anything.

**Decision.** Models are compared in this order: seasonal naive (same weekday of the previous week) → LightGBM with engineered features → LSTM written from scratch in PyTorch → a modern deep learning forecaster (N-HiTS or TFT). Every model is evaluated on the same walk-forward test split and must beat the previous one.

**Consequences.** The result is an honest comparison table, which is the most valuable artefact for the portfolio. If deep learning does not win on a series this small, that is a valid finding.

---

## D3: Data lives in the repository, as Parquet

**Context.** Zero-cost constraint and a small volume (tens of thousands of rows, a few MB).

**Decision.** Raw data goes to `data/raw/` and processed data to `data/processed/`, in Parquet, versioned in Git. A GitHub Action refreshes it daily.

**Consequences.** No database and no paid storage. Revisit if the volume exceeds about 100 MB, for example by moving to Hugging Face Datasets (also free).

---

## D4: Full download on every run, not incremental

**Context.** The four SNS datasets total about 4 MB as Parquet, a full download takes seconds, and the SNS can revise values for past days.

**Decision.** Every run downloads each dataset in full through `exports/parquet` and replaces the file in `data/raw/`.

**Consequences.**
- (+) Simpler code, and retroactive SNS corrections are picked up automatically.
- (+) The daily `git diff` shows whether past values were revised, which says something about data quality.
- (−) Each daily commit rewrites files of about 1.8 MB, so after a year the Git history may reach a few hundred MB. If that becomes a problem, commit weekly instead or move to Hugging Face Datasets.

The same rule applies to the weather data (`data/raw/meteo.parquet`), which Open-Meteo also revises for recent days.

---

## D5: One weather point per region, none for mainland Portugal

**Context.** The SNS targets are regional aggregates, while weather is local. The options were a single city per region, an average over several points, or a population-weighted average.

**Decision.** Use one city per region: the main urban centre, where most emergency visits happen (Porto, Coimbra, Lisboa, Évora, Faro). Do not download weather for the `Portugal Continental` series.

**Consequences.**
- (+) 5 API calls, simple to understand and to join on `ars`.
- (−) Inland and mountain areas (Trás-os-Montes, Beira Interior) are not represented. Temperature anomalies tend to be regional, so the error should be small; revisit if weather features turn out to matter a lot.
- When features are built, the `Portugal Continental` series will either get a population-weighted average of the five regions or be modelled without weather features.

---

## D6: Keep the COVID period, flag it, and evaluate only on recent data

**Context.** From March 2020 to early 2021 emergency episodes dropped by about 40 % and waiting times dropped with them. Since 2022 waiting times are about 30 % higher than in 2017–2019. Neither period reflects the current regime. Dropping the COVID months would break the lag features (yesterday, last week) around the gap.

**Decision.**
- Keep all data, and add an `is_covid` feature for 2020-03-01 to 2021-03-31.
- Evaluate (validation and test) only on data from 2023 onwards.
- As an experiment, also train on 2022 onwards only, and keep whichever performs better on the same test period.

**Consequences.** The models see the full history, including seasonality from 10 winters, but are judged only on the current regime.

---

## D7: The Algarve is not a forecasting target for now

**Context.** The Algarve waiting-time series has large gaps every few years, extreme outliers (up to 378 minutes), and only 25 days of data in 2026 (the last on 2026-06-14). Even a naive forecast has an MAE of about 30 minutes there, against 6–13 in the other regions.

**Decision.** Forecast five series: Norte, Centro, Lisboa e Vale do Tejo, Alentejo and mainland Portugal. Keep the Algarve in the raw data and in the analysis. Its episode counts, which are complete, can still be used as a feature.

**Consequences.** The published forecast will not cover the Algarve. Revisit if the SNS resumes publishing its waiting time consistently; the daily data collection will show it.

---

## D8: Evaluation protocol: forecast origin, horizons, splits and metric

**Context.** D2 requires every model to be evaluated on the same walk-forward split, and D6 says to evaluate only on data from 2023 onwards. Without one shared definition, each model would end up scored slightly differently.

**Decision.**
- **Forecast origin:** on day `t`, a model may use only data up to and including `t`, and it forecasts `t + 1` to `t + 7` (horizons 1 to 7). Every day is an origin, so this is walk-forward by construction.
- **Targets:** `wait_minutes` (primary) and `episodes` (secondary), for the five series from D7.
- **Periods**, by the date being forecast: validation from 2023-01-01 to 2024-12-31, used to choose features and hyperparameters; test from 2025-01-01 onwards, used only to compare finished models.
- **Metric:** MAE per region and horizon, averaged over the five series. Days with no observed value (for example the 2025-06-24 to 2025-07-04 gap) are not scored.
- **Gaps in the inputs** are forward-filled with the last known value, so every model always produces a forecast.
- **Code:** `src/evaluation.py` (constants and `score()`). Every model returns the same forecast table (`origin`, `target_date`, `horizon`, `ars`, `y_true`, `y_pred`).

**Consequences.**
- (+) Models are compared like for like, per horizon.
- (−) Forecasts are made as if the data for day `t` were already available on day `t`. In practice the SNS publishes with a delay of a few days. The real horizon is therefore longer, and this has to be handled when the forecast is put into production.

**Baselines** (`python -m src.baselines`), MAE in minutes, average over the five series:

| Model | Validation, h=1 | Validation, 1–7 | Test, h=1 | Test, 1–7 |
|---|---|---|---|---|
| last value (yesterday) | 9.3 | 10.8 | 7.6 | 9.0 |
| **seasonal naive** (same weekday last week) | 9.6 | **9.6** | 7.6 | **7.6** |

The seasonal naive is the bar to beat. Per region on the test period: Norte 5.7, Centro 7.9, Lisboa e Vale do Tejo 12.2, Alentejo 6.2, mainland Portugal 6.1.

---

## D9: LightGBM design: one global model per horizon, relative target, no future weather

**Context.** This is the first model to try to beat the seasonal naive (D2, D8). Several choices are not obvious: one model per series or one for all of them, how to cope with the post-COVID level shift (trees cannot predict values outside the range they were trained on), and which information may be used at forecast time.

**Decision.**
- **One global model** for all five series, with the region as a categorical feature, and **one model per horizon** (7 models). Each horizon has its own lag features (for example the same weekday last week is `t + h - 7`).
- **Relative target:** the model predicts `y(t + h) − base`, where `base` is the mean of the last 28 days. The lags are expressed relative to `base` too. The level comes from recent data, and the trees learn only the deviations.
- **Features** (`src/features.py`), all known at the origin `t`:
  - 14 daily lags, the same weekday 1 and 2 weeks back, the 7-day mean and the 28-day standard deviation;
  - the episodes relative to their 28-day mean;
  - the respiratory infection share and its 7-day change, and the admission share;
  - weather as 3-day means up to `t`; mainland Portugal uses the average of the regions (D5);
  - `is_covid` (D6).
- **Calendar of the forecast day** (weekday, day of the year, holiday, day after a holiday, 24 Dec to 5 Jan window): this is the only future information allowed, because it is known in advance.
- **No weather for the forecast day.** Using the observed weather would be an unfair advantage, because in production only a weather forecast exists. Open-Meteo's historical forecasts could be used later to test this properly.
- **Loss:** L1, which optimises the MAE directly.
- **Walk-forward with a monthly refit:** each month is forecast by a model trained only on days whose value was known at that month's first origin.
- **Training start:** the full history beat training on 2022 onwards on the validation period (7.48 vs. 7.85 minutes; 238 vs. 257 episodes). The full history is kept, which closes the experiment from D6.

**Consequences.**
- (+) No leakage: a test (`tests/test_features.py`) changes every value after the origin and checks that the features at the origin stay the same.
- (−) Evaluation takes a few minutes, because it fits about 7 × 24 models per period.
- Hyperparameters are not tuned yet (`src/lgbm.py`, `PARAMS`). Tuning on the validation period is the obvious next improvement.

**Results on the test period** (`python -m src.lgbm --test`, from 2025-01-01, MAE averaged over the five series):

| Target | Seasonal naive | LightGBM, h=1 | LightGBM, h=7 | LightGBM, 1–7 | Gain |
|---|---|---|---|---|---|
| Waiting time (min) | 7.6 | 5.3 | 6.2 | **5.9** | −22 % |
| Episodes | 278 | 179 | 237 | **219** | −21 % |

Waiting time per region, horizons 1–7: Norte 4.2, Centro 6.4, Lisboa e Vale do Tejo 9.6, Alentejo 4.8, mainland Portugal 4.5. The gain on the test period is about the same as on validation (−22 %), so the model is not overfitted to the validation years.

---

## D10: Keep the default LightGBM hyperparameters

**Context.** The D9 hyperparameters (`src/lgbm.py`, `PARAMS`) were reasonable defaults that had never been tuned. Tuning them was the obvious next improvement.

**Decision.** A random search over 24 configurations was run on the validation period (`python -m src.tune_lgbm`, logged to MLflow in `mlflow.db`). It searched the learning rate, the number of leaves, the minimum data per leaf, the feature and bagging fractions, the L2 penalty and the number of rounds. To keep it fast, the search optimised the waiting time only and refitted every 3 months instead of every month.

The best configuration reached 7.456 minutes, against 7.469 for the current one. That is a difference of 0.01 minutes, well within the noise. Most configurations scored between 7.46 and 7.55. Only the clearly underfitted ones were worse, for example a learning rate of 0.01 with 200 rounds scored 8.2. The current hyperparameters are kept.

**Consequences.**
- The model is not sensitive to its hyperparameters. The gains came from the features and from the relative target (D9), so further improvements should come from new information (for example weather forecasts, school holidays, or the Algarve episodes), not from tuning.
- The search script and the MLflow log stay in the repository, so the search can be repeated after new features are added. `mlflow.db` is not versioned. To browse the runs: `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

---

## D11: LSTM design: one global sequence model, all 7 horizons at once

**Context.** This is step 3 of D2: an LSTM written from scratch in PyTorch, to check whether a sequence model beats LightGBM on these series.

**Decision.**
- **Input:** for each origin `t` and region, the last 56 days up to `t`. There are 14 channels per day:
  - the target relative to `base` (the 28-day mean at `t`, as in D9);
  - the episodes relative to their 28-day mean (waiting-time model only);
  - the respiratory infection and admission shares;
  - temperature and precipitation;
  - weekday and day of the year as sine and cosine, holiday, day after a holiday, the Christmas window and `is_covid`.
- **Known future:** the calendar of the 7 forecast days (weekday one-hot, holiday, day after a holiday, Christmas window) goes straight to the output head.
- **Architecture:** a region embedding (size 4) is joined to every step. One LSTM layer with 64 units feeds an MLP head that outputs the 7 horizons at once. The loss is L1, with days that have no observed value masked out.
- **Training:** Adam, learning rate 1e-3, batches of 256. Up to 60 epochs, with early stopping on the most recent 10 % of the training origins (patience 6).
- **Walk-forward with a refit every 3 months**, not every month, because training a network takes about 25 seconds. For LightGBM the refit frequency made no difference (D10: 7.469 with a 3-month refit, 7.48 with a monthly one).
- A test (`tests/test_lstm.py`) checks that the inputs at an origin do not change when every later value is changed.

**Results** (MAE averaged over the five series, horizons 1–7, one training seed):

| Target | Period | Seasonal naive | LightGBM | LSTM |
|---|---|---|---|---|
| Waiting time (min) | validation | 9.64 | **7.48** | 7.62 |
| Waiting time (min) | test | 7.63 | **5.92** | 6.25 |
| Episodes | validation | 307 | 238 | **235** |
| Episodes | test | 278 | 219 | **216** |

**Consequences.**
- The LSTM clearly beats the seasonal naive (−18 % to −24 %), but it does not beat LightGBM. On the waiting time, the primary target, it is about 5 % worse on the test period. On episodes it is about 1 % better, which is within the noise of a single seed. With about 3,500 days per series this is the expected outcome, and it is the honest finding D2 asked for.
- The LSTM loses more at horizon 1 (6.3 → 5.8 vs. 5.3 for LightGBM on the test period) and less at horizon 7. Its errors may therefore differ enough from LightGBM's for an average of the two to help.
- Next: N-HiTS or TFT (step 7 in the README), then the comparison and possibly an ensemble.

---

## D12: N-HiTS from neuralforecast does not beat LightGBM; LightGBM stays the main model

**Context.** This is step 4 of D2: a modern, off-the-shelf deep learning forecaster. N-HiTS was chosen over TFT because it is much faster to train on a CPU, and walk-forward evaluation needs about 9 refits per period and target.

**Decision.**
- **Setup:** `src/nhits.py`, `neuralforecast` 3.2.2. The model sees the same information as the LSTM (D11):
  - the last 56 days of the target;
  - past covariates: respiratory share, admissions, temperature, precipitation, and the episodes for the waiting-time model;
  - the known calendar of the forecast days;
  - `is_covid`.
- **Settings:** L1 loss, `robust` scaling per window (handles the level shift), up to 1,000 steps with early stopping on the last 56 days of each training set.
- **Walk-forward** through `NeuralForecast.cross_validation`: one origin per day, refit every 91 origins (about 3 months, as for the LSTM). For the last origins of the test period, the forecast days beyond the data are padded with the known calendar. They are never used as input or scored.
- The default architecture is used and was not tuned.

**Results** (MAE averaged over the five series, horizons 1–7):

| Target | Period | Seasonal naive | LightGBM | LSTM | N-HiTS |
|---|---|---|---|---|---|
| Waiting time (min) | validation | 9.64 | **7.48** | 7.62 | 8.17 |
| Waiting time (min) | test | 7.63 | **5.92** | 6.25 | 6.66 |
| Episodes | validation | 307 | 238 | **235** | 273 |
| Episodes | test | 278 | 219 | **216** | 249 |

**Consequences.**
- N-HiTS beats the seasonal naive (−10 % to −15 %) but comes last of the three models on both targets and both periods. Its likely handicaps are a network sized for long series, about 3,500 days per series, and per-window scaling that hides the level from the model, while LightGBM and the LSTM receive it explicitly through `base`.
- **LightGBM stays the main model** for the waiting time. It is the most accurate, the fastest to train (seconds) and the easiest to explain. For episodes, the LSTM and LightGBM are tied.
- TFT is not tried. It is heavier than N-HiTS and has the same data limitation, so it is unlikely to change the conclusion. Revisit if the data changes, for example with per-hospital series.
- Next: average the forecasts of LightGBM and the LSTM (they err differently, D11) and close the comparison table.
