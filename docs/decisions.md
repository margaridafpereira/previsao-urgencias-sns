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

## D7: The Algarve is not a forecasting target for now (superseded by D19)

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

---

## D13: Average LightGBM and LSTM: use it for episodes, keep LightGBM alone for waiting time

**Context.** LightGBM and the LSTM make different errors (D11): the LSTM loses more at horizon 1, LightGBM more at horizon 7. Averaging models with different errors often beats each of them.

**Decision.**
- **Method:** `src/ensemble.py` combines the saved forecasts (`data/forecasts/`, written by `src/lgbm.py` and `src/lstm.py`) as `w × LightGBM + (1 − w) × LSTM`. The weight `w` is chosen on the validation period from 0 to 1 in steps of 0.1, and then applied unchanged to the test period.
- **Waiting time:** the best weight on validation is 0.6, giving 7.36 against 7.48 for LightGBM alone. On the test period the gain almost vanishes: 5.88 against 5.92 (−0.7 %). This is not worth running and maintaining a second model, so **LightGBM alone** stays the waiting-time model.
- **Episodes:** the best weight is 0.5, giving 219.7 against 237.9 on validation. On the test period it scores **198.6 against 218.9 for LightGBM and 215.9 for the LSTM (−8 %)**, and it is better at every horizon (h=1: 170 vs. 179; h=7: 213 vs. 231). For episodes, **the 50/50 average** is the best model.

**Results on the test period** (MAE averaged over the five series, horizons 1–7):

| Model | Waiting time (min) | Episodes |
|---|---|---|
| Seasonal naive | 7.63 | 278 |
| LightGBM | 5.92 | 219 |
| LSTM | 6.25 | 216 |
| N-HiTS | 6.66 | 249 |
| LightGBM + LSTM | **5.88** (w = 0.6) | **199** (w = 0.5) |

**Consequences.**
- In production, the waiting-time forecast needs only LightGBM. The episode forecast needs both models, so the daily job also has to train the LSTM (about 30 seconds on a CPU). If that turns out to be a burden, falling back to LightGBM costs about 10 % in accuracy on episodes.

---

## D14: 80 % prediction intervals: quantile LightGBM with conformal calibration

**Context.** A single number ("72 minutes tomorrow") does not say how sure the forecast is. Hospital managers need a range to plan for ("between 63 and 84"). Recent forecasting work also evaluates probabilistic forecasts, not only point forecasts.

**Decision.**
- **Quantile regression:** two extra LightGBM models per horizon, with the quantile loss at 10 % and 90 %. They use the same features and the same monthly walk-forward as the point model (D9). The interval adapts to conditions, so it is wider in winter and when flu is rising.
- **Conformal calibration (CQR):** on the validation period, measure how far the actual values fall outside the interval. Widen (or narrow) each horizon by the margin that gives 80 % coverage, then apply that margin unchanged to the test period.
- **Code:** `src/intervals.py`. The output is written to `data/forecasts/lightgbm_intervals_<target>_test.parquet`.

**Results on the test period** (coverage: share of days whose actual value falls inside the interval; width: average high − low):

| Target | Raw quantiles: coverage / width | Calibrated: coverage / width |
|---|---|---|
| Waiting time | 75.8 % / 17.3 min | **84.8 % / 20.9 min** |
| Episodes | 76.6 % / 670 | **81.2 % / 716** |

**Consequences.**
- The raw quantiles are slightly too narrow (about 76 % for a promised 80 %), as quantile regression usually is. Calibration fixes this.
- For the waiting time, calibration overshoots on the test period (85 %). The validation years (2023–2024) were more volatile than 2025–2026, so the margin learnt there is a little too generous. That is the safe side for planning. Recalibrating on recent data in production would bring it closer to 80 %.
- Calibrated coverage per region ranges from 80 % (Lisboa e Vale do Tejo) to 88 % (Alentejo, Norte). The average width is 21 minutes for an average wait of about 70 minutes.
- The published forecast will show the calibrated interval.

---

## D15: Every evaluation is logged to MLflow

**Context.** MLflow was first used only for the LightGBM hyperparameter search (D10). The other models were reported only in the terminal and in this log, which made them hard to compare.

**Decision.**
- `src/tracking.py`, `record()`: every model (baselines, LightGBM, LSTM, N-HiTS, ETS, SARIMAX, ensemble) calls it at the end of an evaluation.
- It saves the forecasts to `data/forecasts/` and creates a run in the `model-comparison` experiment of `mlflow.db`, with tags `model`, `target` and `period`.
- It logs the MAE for horizons 1–7, per horizon and per region.
- `python -m src.tracking` logs again every forecast already saved.

**Consequences.**
- All models can be compared in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`, **Model training** mode, experiment `model-comparison`).
- `mlflow.db` and `data/forecasts/` are not versioned, because they are reproducible from the code. The numbers that matter are kept in this log and in the README.

---

## D16: Classical statistical models: SARIMAX is a strong reference

**Context.** ARIMA and exponential smoothing are the most used models in the ED forecasting literature ([literature.md](literature.md)). The comparison had no classical statistical model, which any reviewer would expect.

**Decision.**
- `src/statistical.py` fits one model per series with `statsmodels`. `statsforecast` was rejected because it requires pandas < 3.
- **ETS:** additive Holt-Winters with a damped trend and weekly seasonality.
- **SARIMAX:** SARIMA(1,0,1)(1,1,1)₇, with the calendar of the forecast days as exogenous variables (holiday, day after a holiday, Christmas window). The calendar is known in advance.
- **Fitting:** each model is fitted on the last 3 years of data, every 3 months. Between refits the parameters are fixed and the state is updated with the new data up to each origin. The orders were not tuned.

**Results** (MAE averaged over the five series, horizons 1–7):

| Target | Period | Seasonal naive | ETS | SARIMAX | LightGBM |
|---|---|---|---|---|---|
| Waiting time (min) | validation | 9.64 | 8.13 | 7.63 | **7.48** |
| Waiting time (min) | test | 7.63 | 6.87 | 6.05 | **5.92** |
| Episodes | validation | 307 | 273 | 249 | **238** |
| Episodes | test | 278 | 249 | 232 | **219** |

**Consequences.**
- **SARIMAX, a classical model with three calendar variables and no flu or weather data, is only 2 % behind LightGBM on the waiting time** (6.05 vs. 5.92 on the test period). It beats the LSTM (6.25) and N-HiTS (6.66). This matches the literature: well-specified classical models are hard to beat on short daily series.
- LightGBM's advantage therefore comes from the extra information (flu, admissions, episodes, weather) and from pooling the five series. It is real but small for the waiting time, and clearer for episodes (−5 %).
- ETS beats the naive baseline but is the weakest model. It has no calendar variables, and holidays and the Christmas window matter (EDA).
- SARIMAX makes different errors from LightGBM, so it is a natural candidate for the ensemble.

---

## D17: Zero-shot Chronos-2 matches the trained LightGBM

**Context.** Time series foundation models are the 2025–2026 state of the art ([literature.md](literature.md)). The question is whether a model pre-trained on millions of other series, and never trained on SNS data, can match models trained on 10 years of it.

**Decision.**
- **Setup:** `src/chronos2.py`, `amazon/chronos-2` through `chronos-forecasting` 2.3.2, on a CPU, with no training or fine-tuning.
- **Inputs at each origin and region:**
  - the last 512 days of the target;
  - past covariates: respiratory share, admissions, temperature, precipitation, and the episodes for the waiting-time model;
  - known-future covariates for the 7 forecast days: holiday, day after a holiday, Christmas window and `is_covid`.
- **Outputs:** the median as the point forecast, plus the 10 % and 90 % quantiles.
- Each (origin, region) pair is a separate series, so 60 origins are forecast per call. The test period takes about 45 minutes for both targets.

**Results on the test period** (MAE averaged over the five series):

| Target | Seasonal naive | SARIMAX | LightGBM | **Chronos-2 (zero-shot)** |
|---|---|---|---|---|
| Waiting time (min), h=1 | 7.63 | – | 5.26 | **5.21** |
| Waiting time (min), 1–7 | 7.63 | 6.05 | **5.92** | 5.95 |
| Episodes, 1–7 | 278 | 232 | 219 | **217** |

Its own 10–90 % interval, with no calibration, covers 76.5 % (waiting time, 17 minutes wide) and 79.4 % (episodes) of the test days. The calibrated LightGBM intervals (D14) cover 84.8 % and 81.2 %.

**Consequences.**
- **A model that has never seen SNS data ties with LightGBM, the best trained model**, and beats the LSTM, N-HiTS and SARIMAX. This is the most notable finding of the comparison. It also means LightGBM is close to what these inputs allow.
- **Ensembles look promising, but are not decided yet.** Equal-weight averages, fixed in advance, give on the test period:
  - LightGBM + Chronos-2: 5.77 min and 206 episodes;
  - LightGBM + LSTM + Chronos-2: 5.74 min and **194** episodes, against 199 for the D13 ensemble.

  Picking the best combination by looking at the test period would be selection on the test set. Chronos-2 is therefore being run on the validation period, and the combination will be chosen there (D18).
- **Production cost:** Chronos-2 needs no training, but it is a 120M-parameter model. It takes about 2–3 seconds per forecast day on a CPU, which is fine for one forecast a day in GitHub Actions. The first run downloads the model, which takes about 1–2 minutes.

---

## D19: Which regions are forecast is decided by a data rule (supersedes D7)

**Context.** D7 removed the Algarve by hand, because its waiting-time series has gaps and stopped in June 2026. If the SNS resumed publishing, the project would keep ignoring it until someone changed the code. If another region stopped, the project would keep publishing forecasts with no recent data behind them.

**Decision.**
- `src/regions.py`, `active_regions()`: a region is forecast if, over the last 90 days, it has a waiting time on **at least 80 % of the days**, and its last value is **at most 14 days** older than the latest day published for any region.
- Staleness is measured against the latest published day, not against today. The usual publication delay of the SNS (about 4 days) affects every region equally and must not exclude any of them.
- **Evaluation keeps the fixed set of five series** (`evaluation.TARGET_REGIONS`), so that model comparisons stay valid. The rule applies to the published forecast.
- Status on 2026-10-07: Norte, Centro, Lisboa e Vale do Tejo, Alentejo and mainland Portugal are active (100 % coverage). The Algarve is inactive (0 % coverage over the last 90 days).

**Consequences.**
- If the Algarve resumes publishing, it is forecast again after about 72 days of data (80 % of 90), with no code change. The global models already learnt its history from 2017–2025.
- An inactive region is shown as "no recent waiting-time data", never with an invented or borrowed value (compare snsmonitor.pt in [data-sources.md](data-sources.md)).

**Validation results** (added 2026-10-08): on 2023–2024, Chronos-2 scores 7.52 minutes for the waiting time (LightGBM 7.48) and 248 for episodes (LightGBM 238). Its 10–90 % interval covers 77.0 % and 77.9 %. For the waiting time it ties with LightGBM in both periods. For episodes it is weaker on validation than on test.

---

## D18: Final model: equal-weight average of the models, chosen on validation

**Context.** D13 and D17 showed that averaging models with different errors helps. To avoid selecting on the test period, every equal-weight combination of LightGBM, LSTM, SARIMAX and Chronos-2 (15 in total) was ranked on the validation period. The test period was only used to confirm the ranking.

**Results** (MAE for horizons 1–7, averaged over the five series; best five on validation plus references):

| Combination | Waiting time, validation | Waiting time, test | Episodes, validation | Episodes, test |
|---|---|---|---|---|
| **LightGBM + LSTM + SARIMAX + Chronos-2** | **7.20** (best) | **5.69** | 220.3 | 195.2 |
| LightGBM + LSTM + SARIMAX | 7.25 | 5.75 | **218.4** (best) | 196.5 |
| LightGBM + LSTM + Chronos-2 | 7.23 | 5.74 | 219.2 | **194.2** |
| LightGBM + Chronos-2 | 7.32 | 5.77 | 232.9 | 206.4 |
| LightGBM + LSTM (D13) | 7.37 | 5.90 | 219.7 | 198.6 |
| LightGBM alone | 7.48 | 5.92 | 237.9 | 218.9 |
| Chronos-2 alone | 7.52 | 5.95 | 248.4 | 216.9 |

**Findings.**
- Every combination of two or more different models beats the best single model on both targets. The validation ranking matches the test ranking closely, so the gain is not luck.
- **Waiting time:** the four-model average is best on validation and is confirmed on test: **5.69 minutes, −4 % against LightGBM**.
- **Episodes:** the top combinations all contain the LSTM, and they are within 1 % of each other on validation (218–220). Without the LSTM, the best is 233. **The four-model average reaches 195 on test, −11 % against LightGBM.**
- The **four-model equal-weight average** is within 1 % of the best combination for both targets. It is therefore chosen as the single configuration for both. It has no tuned weights, which keeps it robust.

**Decision.** The best research result is the **four-model average: 5.69 minutes and 195 episodes on the test period.** The production setup (four models, or a simpler subset) is decided at deployment; see the trade-off below.

**Trade-off for production.** LightGBM + Chronos-2 needs only one trained model, because Chronos-2 is zero-shot. It loses 1.4 % on the waiting time (5.77 vs. 5.69) but 6 % on episodes (206 vs. 195). Keeping the LSTM recovers most of the episode gain.

---

## D20: Measure the SNS publication delay, and keep the daily collection lightweight

**Context.** D8 assumes that the data for day `t` is available on day `t`. In practice it is not:
- On 2026-10-07 and again on 2026-10-08 (10:22 UTC), the latest day published was 2026-10-03, so the delay was 4–5 days.
- The catalogue showed the flu dataset as modified on 2026-10-06 with data up to 2026-10-03. This suggests the SNS publishes in batches, not every day.
- snsmonitor.pt shows the same delay ([data-sources.md](data-sources.md)).

The delay decides which horizons the published forecast really needs. If the latest data is from day `t` and today is `t + d`, then "today to 6 days ahead" means horizons `d` to `d + 6`.

**Decision.**
- `src/ingest/sns.py` appends one row per collection run and dataset to `data/raw/publication_log.csv` (`checked_at`, `dataset_id`, `last_day`). The daily GitHub Action commits it with the data. After a few weeks it shows the real delay and the publication pattern.
- The daily collection no longer installs the modelling stack (PyTorch, neuralforecast, Chronos, MLflow). It uses `requirements-ingest.txt` (requests, pandas, pyarrow, pytest) and runs only the ingestion tests. This makes it faster and less likely to break.

**Consequences.**
- How the forecast handles the delay is decided once the log has a few weeks of data, or earlier if the forecast horizon is extended (see the roadmap).

---

## D21: Keep the 7-day horizon, state the SNS delay openly, and keep 14 days as an option

**Context.** With a publication delay of about 5 days (D20), the models forecast 7 days from the latest published day. On 2026-10-08, with data up to 2026-10-03, they cover 4 to 10 October, so only today and the next 2 days are still in the future. Extending the models to 14 days would cover a full week ahead, but every model would have to be adapted and evaluated again.

**Decision.**
- **Keep the 7-day horizon for now.** The published forecast shows the days from the latest published day up to `t + 7`, each with its date. Days already past are shown as "estimates of days the SNS has not published yet", not as forecasts.
- **State the delay clearly** next to the forecast: "Latest official SNS data: 3 October 2026 (published with a delay of about 5 days)". The date comes from the data, not from a fixed number.
- **Option to extend to 14 days later:** the models are written for a configurable horizon (`evaluation.HORIZONS`). Extending means re-running the evaluation of the four models in D18, which takes about 3–4 hours of CPU. This is done when the publication log (D20) confirms the delay is structural, or if users need more days ahead.

**Consequences.**
- On a typical day the forecast covers today plus about 2 days. Its usefulness depends on how often and how late the SNS publishes, which the log will show.
- The published accuracy stays the one measured for horizons 1–7 (D18). It is not reduced to the subset of horizons that are still in the future.
