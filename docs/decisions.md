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
