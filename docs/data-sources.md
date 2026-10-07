# Data sources

All sources are public, free and need no authentication. Last verified against the live APIs on 2026-10-07.

## 1. SNS Transparency Portal (primary source)

- **Platform:** OpenDataSoft, REST API v2.1.
- **Base URL:** `https://transparencia.sns.gov.pt/api/explore/v2.1/catalog/datasets/<dataset_id>`
- **Full download:** `<base>/exports/parquet` (or `csv`). This is what `src/ingest/sns.py` uses.
- **Queries:** `<base>/records`, at most 100 rows per request, with ODSQL in `where`, `select`, `group_by` and `order_by`.
- **Geolocation:** the Parquet export returns `localizacao_geografica` as a **WKB point** (binary), not as lat/lon. The ingestion script decodes it into `lat` and `lon` columns.

### 1.1 `monitorizacao-sazonal-csh`: daily, per region (the forecasting target)

Hospital care activity, seasonal monitoring.

| Field | Type | Notes |
|---|---|---|
| `periodo` | date | **Daily**, from 2016-11-01 to the present |
| `ars` | text | `ARS Norte`, `ARS Centro`, `ARS Lisboa e Vale do Tejo`, `ARS Alentejo`, `ARS Algarve`, `Portugal Continental` |
| `indicador` | text | See below |
| `valor` | double | |
| `unidade` | text | e.g. `minuto`, `%`, `epis. urg./100.000 resid.` |

**Indicators:**
- `Tempo médio de espera entre a triagem e a primeira observação médica (rede de urgência hospitalar)`: average minutes from triage to the first medical observation. **This is the primary target.**
- `Número estimado de episódios de urgência`: estimated number of emergency episodes. **This is the secondary target (demand).**
- `Taxa diária de atendimentos urgentes com prioridade verde ou azul`: non-urgent (green/blue triage) visits per 100,000 residents.
- `Taxa diária de atendimentos urgentes com internamento`: share of visits that led to a hospital admission (%).

**Volume:** about 2,000–2,200 rows per year for the waiting-time indicator (6 regions × 365 days). Some days are missing.

**Average waiting time per year (minutes, all regions):**
2017: 50 · 2019: 53 · 2020: 43 (COVID) · 2022: 64 · 2023: 70 · 2024: 71 · 2025: 66.
2020 is an anomaly and needs specific handling.

**Caveat:** the dataset metadata reports `modified: 2019-01-21`, but the data is still updated daily. Do not rely on that field to detect new data.

### 1.2 `atividade-sindrome-gripal-csh`: daily, per region (feature: flu)

Same schema as 1.1 (`periodo`, `ars`, `indicador`, `valor`, `unidade`), with about 84,000 rows. Three of its four indicators (waiting time, episodes, non-urgent rate) repeat 1.1 with identical values. The only new one is:

- `Taxa de episódios de urgência com diagnóstico de infeção respiratória`: share of episodes diagnosed with a respiratory infection (%). This is the flu signal.

### Data quality (daily datasets)

- All regions share an 11-day gap, from 2025-06-24 to 2025-07-04.
- The **Algarve waiting time** has large gaps (153 days missing in 2017, 147 in 2019, 100 in 2024, 78 in 2025, and only 25 of 276 days present in 2026, the last one on 2026-06-14) and extreme outliers (up to 378 minutes). The Algarve episode counts are complete. See D7 in [decisions.md](decisions.md).
- `src/data.py` (`load_daily()`) merges both datasets and the weather into one row per `periodo` × `ars`, keeping missing days as rows with NaN.

### 1.3 `atendimentos-em-urgencia-triagem-manchester`: monthly, per hospital

| Field | Notes |
|---|---|
| `tempo` | **Monthly**, from 2013-01 onwards, published about 2 months late |
| `regiao`, `instituicao` | 78 institutions |
| `lat`, `lon` | Decoded from the WKB geolocation |
| `no_de_atendimentos_em_urgencia_su_triagem_manchester_{vermelha,laranja,amarela,verde,azul,branca}` | Visit counts per Manchester triage colour |
| `no_de_atendimentos_em_sem_triagem_manchester` | Visits without triage |

Useful for exploratory analysis and for monthly per-hospital forecasting. Not suitable for daily forecasting.

### 1.4 `atendimentos-por-tipo-de-urgencia-hospitalar-link`: monthly, per hospital

Contains `urgencias_geral`, `urgencias_pediatricas`, `urgencia_obstetricia`, `urgencia_psiquiatrica` and `total_urgencias`, per institution and month.

## 2. Weather: Open-Meteo

- **Historical:** `https://archive-api.open-meteo.com/v1/archive`, daily data since 1940, available up to the previous day. Free for non-commercial use, no API key. This is what `src/ingest/meteo.py` uses.
- **Forecast:** `https://api.open-meteo.com/v1/forecast`, 7–16 days ahead. Not collected yet; it will be needed at prediction time, when weather features have to come from a forecast.
- **Output:** `data/raw/meteo.parquet`, one row per `periodo` × `ars`, from 2016-11-01. About 18,000 rows and 130 KB.
- **Variables:** `temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean` (°C), `precipitation_sum` (mm). Days are aligned to `Europe/Lisbon`.

One coordinate represents each region (see D5 in [decisions.md](decisions.md)). The `ars` values match the SNS dataset exactly, so the two join directly.

| `ars` | City | Lat | Lon |
|---|---|---|---|
| ARS Norte | Porto | 41.1496 | −8.6110 |
| ARS Centro | Coimbra | 40.2033 | −8.4103 |
| ARS Lisboa e Vale do Tejo | Lisboa | 38.7223 | −9.1393 |
| ARS Alentejo | Évora | 38.5714 | −7.9135 |
| ARS Algarve | Faro | 37.0194 | −7.9304 |

## 3. Calendar

- **National holidays:** the Python library `holidays` (`holidays.PT()`), no network calls.
- **School holidays:** the calendar published yearly by DGEstE (entered manually).

## Evaluated and rejected

| Source | Why not |
|---|---|
| `tempos.min-saude.pt` (real-time waiting times per hospital) | It is an embedded Power BI report. Its only JSON API (`/api/institutions`) answers 400 to direct calls and contains no waiting times. Scraping Power BI would be fragile and legally questionable. See D1 in [decisions.md](decisions.md). |
