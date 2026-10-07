# Literature review

What the research says about forecasting emergency department (ED) demand and waiting times, and how this project relates to it. Reviewed on 2026-10-07. Paywalled papers were read from their abstracts and published summaries only; this is noted where it applies.

## 1. Which models work for ED forecasting

| Finding | Source | What this project does |
|---|---|---|
| Machine learning models generally beat classical time series methods. ARIMA, SVM and random forests are the most used; gradient-boosted machines are recommended among hybrid methods. 53 methods reviewed. | Updated systematic review, *BMC Emergency Medicine* 2025 [1] (abstract and summaries only) | LightGBM is the main model (D9). ARIMA and ETS were added as the classical reference (D16). |
| For ED occupancy, DeepAR, N-BEATS, TFT and LightGBM all beat the classical benchmarks, by up to 15 %. **LightGBM was the best model**, and the univariate LightGBM beat the multivariate deep learning models. | Tuominen et al., *International Journal of Forecasting* 2024 [2] | Same result here: LightGBM beats LSTM (D11) and N-HiTS (D12). |
| Recent ED arrival studies compare a seasonal naive baseline, ETS, ridge regression, LightGBM and hybrid TCN or LSTM networks, with feature engineering. | *BMC Medical Informatics and Decision Making* 2024 [3]; *Healthcare* 2026 [4] | Same comparison set: seasonal naive (D8), ETS (D16), LightGBM, LSTM. |
| A LightGBM model won the M5 competition, the largest public forecasting competition, and all 50 top methods were machine learning. | M5 competition [5] | Supports a global gradient-boosting model across related series (D9). |

## 2. Which predictors help

| Predictor | Evidence | Status here |
|---|---|---|
| Day of the week, month, public holidays | Consistently useful [1, 6] | Used (D9) |
| School holidays | Used in several studies [6] | **Not yet used** |
| Temperature and precipitation | Improve accuracy in most studies [1, 6] | Used, up to the forecast origin only (D9). Weather *forecasts* for the forecast days are not used yet. |
| Influenza and respiratory activity | Used in seasonal studies | Used: SNS respiratory infection share (D9) |
| Internet search volume (e.g. Google Trends) | Improves arrival forecasts in some studies [7] | Not used |

## 3. State of the art in 2025–2026: foundation models

Pre-trained time series foundation models (Chronos-2, TimesFM 2.5, Moirai, TiRex) forecast series they have never seen, without training ("zero-shot") [8, 9].

- **Chronos-2** (Amazon, October 2025) accepts past and known-future covariates. On fev-bench it outperforms other models by a wide margin on tasks with covariates [8].
- On a public energy benchmark, zero-shot Chronos-2 beat a tuned production XGBoost pipeline [8].

None of the ED forecasting studies above test foundation models. This project compares zero-shot Chronos-2 with the trained LightGBM (D17).

## 4. Evaluation practice

| Practice | Status here |
|---|---|
| Seasonal naive baseline | Yes (D8) |
| Rolling-origin (walk-forward) evaluation, no leakage | Yes (D8), with leakage tests |
| Error per forecast horizon | Yes (D8) |
| Probabilistic forecasts (intervals, coverage) | Yes: quantile LightGBM with conformal calibration (D14), and the Chronos-2 quantiles |
| Hierarchical consistency (regions add up to the national total) | **Not yet**: candidate improvement for episodes |

## 5. Gaps this project addresses

1. Most ED studies forecast **arrivals at one hospital**. Few forecast **waiting times**, and few use national open data. This project forecasts both waiting time and episodes for every SNS region from public data.
2. Foundation models have not been tested on ED data in the studies found. This project tests one.
3. Many studies report only point forecasts. This project reports calibrated intervals.

## References

1. *Prognostic models for predicting patient arrivals in emergency departments: an updated systematic review and research agenda.* BMC Emergency Medicine, 2025. https://link.springer.com/article/10.1186/s12873-025-01250-8
2. Tuominen et al. *Forecasting emergency department occupancy with advanced machine learning models and multivariable input.* International Journal of Forecasting, 2024. https://www.sciencedirect.com/science/article/pii/S0169207023001346
3. *Enhanced forecasting of emergency department patient arrivals using feature engineering approach and machine learning.* BMC Medical Informatics and Decision Making, 2024. https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-024-02788-6
4. *Predicting Emergency Department Patient Arrivals at Hospitals Using Machine Learning Techniques.* Healthcare, 2026. https://www.mdpi.com/2227-9032/14/9/1191
5. Makridakis et al. *The M5 competition.* Summary: https://medium.com/analytics-vidhya/predicting-the-future-with-learnings-from-the-m5-competition-d54e84ca3d0d
6. Jiang et al. *A systematic review of the modelling of patient arrivals in emergency departments.* Quantitative Imaging in Medicine and Surgery. https://qims.amegroups.org/article/view/102522/html
7. *Accurate Forecasting of Emergency Department Arrivals With Internet Search Index and Machine Learning Models.* JMIR Medical Informatics. https://www.sciencedirect.com/org/science/article/pii/S2291969422001776
8. Ansari et al. *Chronos-2: From Univariate to Universal Forecasting.* 2025. https://arxiv.org/html/2510.15821v1
9. *A Survey of Deep Learning and Foundation Models for Time Series Forecasting.* 2024. https://arxiv.org/pdf/2401.13912
