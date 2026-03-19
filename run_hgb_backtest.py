# HGB walk-forward backtest with quality-filtered sentiment; biweekly rebalancing, evaluated 2021+.
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'intermediate'

REBAL_FREQ = 10                                                   
MIN_TRAIN  = 250_000                                                      
EVAL_START = pd.Timestamp('2021-01-01')                                        
TC_BPS     = 10                                                                   
N_LONG     = 50
N_SHORT    = 50

TRAD_FEATS = ['momentum', 'value_composite', 'quality_composite']
SENT_FEATS = ['finbert_sent_7d', 'finbert_sent_30d', 'finbert_sent_60d',
              'sent_momentum', 'news_volume_7d', 'news_volume_30d']
                                                                           
HGB_PARAMS = dict(
    max_iter         = 150,
    max_depth        = 4,
    learning_rate    = 0.05,
    min_samples_leaf = 50,
    l2_regularization= 1.0,
    random_state     = 42,
)

QUALITY_DOMAINS = {
    'sec.gov',
    'finance.yahoo.com', 'news.yahoo.com',
    'prnewswire.com', 'businesswire.com', 'globenewswire.com',
    'marketwatch.com', 'fool.com', 'benzinga.com',
    'seekingalpha.com', 'insidermonkey.com', 'investing.com',
    'reuters.com', 'bloomberg.com', 'wsj.com', 'ft.com',
    'cnbc.com', 'forbes.com',
}

def rebuild_model_table():
    print('Rebuilding model_table')
    scored   = pd.read_parquet(DATA_DIR / 'gdelt_articles_scored.parquet')
    filtered = scored[scored['domain'].isin(QUALITY_DOMAINS)].copy()
    filtered['date'] = pd.to_datetime(filtered['date'], utc=True).dt.tz_localize(None)
    print(f'Articles: {len(filtered):,} / {len(scored):,} after quality filter')
    print(filtered['domain'].value_counts().to_string())

    daily = (
        filtered.groupby(['ticker', 'date'])['finbert_score']
        .agg(['mean', 'count'])
        .rename(columns={'mean': 'finbert_daily', 'count': 'news_count'})
    )
    daily.index.set_names(['ticker', 'date'], inplace=True)
    finbert_wide = daily['finbert_daily'].unstack('ticker')
    count_wide   = daily['news_count'].unstack('ticker').fillna(0)

    finbert_wide.to_parquet(DATA_DIR / 'finbert_daily_wide.parquet')
    count_wide.to_parquet(DATA_DIR / 'news_count_wide.parquet')

    factors   = pd.read_parquet(DATA_DIR / 'factors_traditional.parquet')
    target    = pd.read_parquet(DATA_DIR / 'target_next_ret.parquet')
    tone_wide = pd.read_parquet(DATA_DIR / 'gdelt_tone_wide.parquet')

    def to_naive(idx):
        return pd.to_datetime(idx, utc=True).tz_convert(None)

    tone_wide.index    = to_naive(tone_wide.index)
    finbert_wide.index = to_naive(finbert_wide.index)
    count_wide.index   = to_naive(count_wide.index)

    cal = pd.date_range(
        min(finbert_wide.index.min(), tone_wide.index.min()),
        max(finbert_wide.index.max(), tone_wide.index.max()),
        freq='D',
    )
    finbert_wide = finbert_wide.reindex(cal)
    tone_wide    = tone_wide.reindex(cal)
    count_wide   = count_wide.reindex(cal).fillna(0)

    FFILL = 60                                                                 

    def rm(df, w): return df.rolling(f'{w}D', min_periods=1).mean()
    def rs(df, w): return df.rolling(f'{w}D', min_periods=1).sum()
    def w2l(df, col):
        s = df.stack()
        s.index.set_names(['date', 'ticker'], inplace=True)
        return s.to_frame(col)

    fb7  = rm(finbert_wide,  7).shift(1).ffill(limit=FFILL)
    fb30 = rm(finbert_wide, 30).shift(1).ffill(limit=FFILL)
    fb60 = rm(finbert_wide, 60).shift(1).ffill(limit=FFILL)
    gt7  = rm(tone_wide,   7).shift(1).ffill(limit=FFILL)
    gt30 = rm(tone_wide,  30).shift(1).ffill(limit=FFILL)
    nv7  = rs(count_wide,  7).shift(1).ffill(limit=FFILL)
    nv30 = rs(count_wide, 30).shift(1).ffill(limit=FFILL)

    sent_feats = (
        w2l(fb7,        'finbert_sent_7d')
        .join(w2l(fb30,        'finbert_sent_30d'), how='outer')
        .join(w2l(fb60,        'finbert_sent_60d'), how='outer')
        .join(w2l(fb7 - fb30,  'sent_momentum'),    how='outer')
        .join(w2l(gt7,         'gdelt_tone_7d'),    how='outer')
        .join(w2l(gt30,        'gdelt_tone_30d'),   how='outer')
        .join(w2l(nv7,         'news_volume_7d'),   how='outer')
        .join(w2l(nv30,        'news_volume_30d'),  how='outer')
    ).sort_index()

    model_table = factors.join(sent_feats, how='left').join(target[['ret_t1']], how='left')
    model_table.sort_index().to_parquet(DATA_DIR / 'model_table.parquet')
    print(f'model_table saved: {model_table.shape}\n')

def run_walkforward(panel, feature_cols, label):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)
    for d in tqdm(rebal_dates, desc=label):
        train = panel.loc[
            panel.index.get_level_values('date') < d
        ].dropna(subset=['ret_t1'] + TRAD_FEATS)

        if len(train) < MIN_TRAIN:
            continue

        X  = train[feature_cols].copy()
        q1, q9 = X.quantile(0.01), X.quantile(0.99)
        X  = X.clip(lower=q1, upper=q9, axis=1)
        y  = train.loc[X.index, 'ret_t1'].values

        model = HistGradientBoostingRegressor(**HGB_PARAMS)
        model.fit(X.values, y)

        try:
            today = panel.loc[(d, slice(None)), feature_cols].copy()
        except KeyError:
            continue
        today = today.dropna(subset=TRAD_FEATS).clip(lower=q1, upper=q9, axis=1)
        if today.empty:
            continue

        pred = (
            pd.Series(model.predict(today.values),
                      index=today.index.get_level_values('ticker'))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        longs, shorts = pred.nlargest(N_LONG).index, pred.nsmallest(N_SHORT).index
        w = pd.Series(0.0, index=pred.index)
        w.loc[longs]  =  1 / max(len(longs),  1)
        w.loc[shorts] = -1 / max(len(shorts), 1)
        gross = w.abs().sum()
        if gross > 0:
            w /= gross

        turnover = (w - last_w.reindex(w.index, fill_value=0)).abs().sum()
        last_w   = w

        i        = np.where(rebal_dates == d)[0][0] + 1
        next_cut = rebal_dates[i] if i < len(rebal_dates) else dates[-1]
        span     = dates[(dates >= d) & (dates < next_cut)]
        for dd in span:
            port.loc[dd] = (w * panel.loc[(dd, w.index), 'ret_t1'].fillna(0.0)).sum()
        if len(span) > 0:
            port.loc[span[0]] -= turnover * TC_BPS / 10_000

    perf    = port.loc[port.index >= EVAL_START].dropna()
    cum     = (1 + perf).cumprod()
    ann_ret = perf.mean() * 252
    ann_vol = perf.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = (cum / cum.cummax() - 1).min()

    stats = {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}
    print(f'  [{label}] ann_ret={ann_ret*100:.2f}% | vol={ann_vol*100:.2f}% | '
          f'sharpe={sharpe:.3f} | max_dd={max_dd*100:.2f}%')

    return perf, cum, stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true',
                        help='Rebuild model_table from scored articles')
    args = parser.parse_args()

    if args.rebuild:
        rebuild_model_table()

    df = pd.read_parquet(DATA_DIR / 'model_table.parquet').sort_index()
    df = df.dropna(subset=['ret_t1'])
    df = df[df.index.get_level_values('date') >= '2017-01-01']
    print(f'Panel: {len(df):,} rows | '
          f'{df.index.get_level_values("date").min().date()} -> '
          f'{df.index.get_level_values("date").max().date()}')
    print(f'Rebal: every {REBAL_FREQ} days | eval from {EVAL_START.date()} | '
          f'burn-in: {MIN_TRAIN:,} rows\n')

    print('Baseline (momentum / value / quality)')
    _, cum_base, stats_base = run_walkforward(df, TRAD_FEATS, 'baseline')

    print('Augmented (+ FinBERT sentiment + news volume)')
    _, cum_aug, stats_aug   = run_walkforward(df, TRAD_FEATS + SENT_FEATS, 'augmented')

    print('Final results:')
    res = pd.DataFrame([
        {'model': 'HGB baseline',  **stats_base},
        {'model': 'HGB augmented', **stats_aug},
    ])
    for c in ['ann_ret', 'ann_vol', 'max_dd']:
        res[c] = (res[c] * 100).round(2).astype(str) + '%'
    res['sharpe'] = res['sharpe'].round(4)
    print(res.to_string(index=False))

    out = pd.DataFrame([
        {'model': 'hgb_baseline',  **stats_base},
        {'model': 'hgb_augmented', **stats_aug},
    ])
    out.to_parquet(DATA_DIR / 'backtest_summary_final.parquet', index=False)
    out.to_csv(DATA_DIR    / 'backtest_summary_final.csv',       index=False)
    print('\nSaved backtest_summary_final.parquet / .csv')
