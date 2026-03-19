# Financial Sentiment Pipeline - Feature Specification

## Implemented Features

### 1. `news_sentiment_avg` (SINGLE feature)
**Average of 5 sources:** YFinance, Bloomberg, NYT, Reuters, CNBC  
- Daily sentiment from each source → mean across sources  
- Range: -1 (negative) to +1 (positive)  
- Updates: Daily  

### 2. SEC Filing Sentiment (SEPARATE features)
**Constant until next filing** (forward-filled):

| Feature | Filing | Update Frequency |
|---------|--------|------------------|
| `sec_10k_sentiment` | 10-K Annual Report | ~1x/year |
| `sec_10q_sentiment` | 10-Q Quarterly Report | ~4x/year |
| `sec_8k_sentiment` | 8-K Material Events | As needed |

### 3. `earnings_sentiment` (SEPARATE feature)
**Earnings call transcript sentiment**  
- Constant until next earnings call (~4x/year)  
- Requires: user-provided CSV via `EARNINGS_CSV`

### 4. Inferred Features (Market Data)
| Feature | Description | Source |
|---------|-------------|--------|
| `vix` | CBOE Volatility Index | yfinance ^VIX |

---

## Additional Inferred Features (Recommended)

For stronger ML/trading models, consider adding:

| Feature | Description | Rationale |
|---------|-------------|------------|
| `price_return_1d` | 1-day S&P 500 return | Momentum signal |
| `price_return_5d` | 5-day return | Short-term trend |
| `price_return_21d` | 21-day return | Monthly momentum |
| `volume_change` | Volume vs 20-day MA | Unusual activity |
| `fed_funds_rate` | Federal funds rate | Macro regime |
| `term_spread` | 10Y - 2Y Treasury | Yield curve / recession |
| `sector_sentiment` | Industry-level news sentiment | Sector rotation |
| `analyst_rating_change` | Upgrades/downgrades | Consensus shift |
| `insider_trading_signal` | Net insider buy/sell | Information signal |
| `options_put_call_ratio` | Put vs call volume | Market positioning |

---

## Data Limitations

- **News**: Historical data availability varies by provider
- **SEC filings**: Full history (15+ years) via SEC EDGAR
- **Earnings transcripts**: Provide via `EARNINGS_CSV`
