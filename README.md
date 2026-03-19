# S&P 500 News Sentiment & Return Prediction

15-year feature extraction pipeline for S&P 500 stocks, combining traditional financial factors with news sentiment from GDELT and SEC EDGAR filings, scored by FinBERT.

## Pipeline Architecture

```
01_data_prep       →  Prices, momentum, value & quality factors
02_gdelt_collection →  GDELT news tone + article headlines
02b_edgar_collection → SEC EDGAR 8-K press-release headlines
03_finbert_aggregation → FinBERT sentiment scoring
04_feature_merge   →  Merge all features → model_table.parquet
05_backtest        →  Expanding-window Ridge regression backtest
```

## Feature Set

| Feature | Source | Description |
|---------|--------|-------------|
| `momentum` | YFinance | 12-month ex-1-month price momentum |
| `value_composite` | YFinance | Z-scored blend of E/P, B/M, EV/EBITDA |
| `quality_composite` | YFinance | Z-scored blend of ROE, ROA, margins, leverage |
| `finbert_sent_7d` | GDELT + EDGAR | 7-day rolling FinBERT sentiment |
| `finbert_sent_30d` | GDELT + EDGAR | 30-day rolling FinBERT sentiment |
| `sent_momentum` | GDELT + EDGAR | Sentiment 7d − 30d (trend) |
| `gdelt_tone_7d` | GDELT | 7-day rolling GDELT tone score |
| `gdelt_tone_30d` | GDELT | 30-day rolling GDELT tone score |
| `news_volume_7d` | GDELT + EDGAR | 7-day article count |
| `news_volume_30d` | GDELT + EDGAR | 30-day article count |
| `ret_t1` | YFinance | Next-day return (target) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the full pipeline in order:

```bash
# Stage 1: Download prices & compute traditional factors (~6 min)
python run_stage1.py

# Stage 2-4: GDELT collection + FinBERT + feature merge (~10 min with cache)
python run_pipeline.py

# Stage 2b: SEC EDGAR 8-K headlines + merge (~3 hrs first run, cached after)
python run_edgar.py

# Stage 5: Backtest
python run_backtest.py
```

Or run notebooks interactively in `notebooks/`.

## Data

All intermediate data is saved to `data/intermediate/` as parquet files.
API responses are cached in `data/cache/` (GDELT and EDGAR) to avoid re-fetching.

## Data Limits

- **GDELT**: Articles available from Feb 2017 onward
- **SEC EDGAR**: 8-K filings available for full history; rate-limited to 10 req/sec
- **YFinance**: 15-year daily price history
# sentimentaugmentedstockselection
