# Tweak sweep: cascade K values, pure-sent re-rank vs ridge+hgb re-rank, stacked interaction feature.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

HGB_PARAMS = dict(max_iter=150, max_depth=4, learning_rate=0.05,
                  min_samples_leaf=50, l2_regularization=1.0, random_state=42)
RIDGE_ALPHA = 50.0

def summarize(port, label):
    perf    = port.loc[port.index >= EVAL_START].dropna()
    cum     = (1 + perf).cumprod()
    ann_ret = perf.mean() * 252
    ann_vol = perf.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = (cum / cum.cummax() - 1).min()
    stats   = {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}
    print(f'  [{label}] ann_ret={ann_ret*100:.2f}% | vol={ann_vol*100:.2f}% | '
          f'sharpe={sharpe:.3f} | max_dd={max_dd*100:.2f}%')
    return stats

def fit_ridge_and_hgb(train_all, q1, q9, sq1, sq9):
    """Fit Ridge on trad + HGB on residuals. Returns (ridge, sc, hgb, rs_mean, rs_std)."""
    X_trad = train_all[TRAD_FEATS].clip(lower=q1, upper=q9, axis=1)
    y      = train_all.loc[X_trad.index, 'ret_t1'].values
    sc     = StandardScaler()
    ridge  = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(sc.fit_transform(X_trad.values), y)
    ridge_pred = ridge.predict(sc.transform(X_trad.values))

    rs_mean = ridge_pred.mean()
    rs_std  = ridge_pred.std() + 1e-8
    ridge_z = (ridge_pred - rs_mean) / rs_std

    X_sent = train_all[SENT_FEATS].clip(lower=sq1, upper=sq9, axis=1).copy()
    X_sent.insert(0, 'ridge_score_z', ridge_z)
    y_res  = y - ridge_pred
    hgb    = HistGradientBoostingRegressor(**HGB_PARAMS)
    hgb.fit(X_sent.values, y_res)

    return ridge, sc, hgb, rs_mean, rs_std

def apply_cascade_port(ridge_today, sent_score, dates, d, rebal_dates, panel,
                        last_w, K, pure_sent_rerank, today_fb30=None):
    """Build cascade portfolio and return (daily_rets, new_last_w)."""
    n    = len(ridge_today)
    k    = min(K, n // 2)
    long_uni  = ridge_today.nlargest(k).index
    short_uni = ridge_today.nsmallest(k).index

    if pure_sent_rerank and today_fb30 is not None:
                                                                                
        fb30_long  = today_fb30.reindex(long_uni)
        fb30_short = today_fb30.reindex(short_uni)
        med_l = fb30_long.median()
        med_s = fb30_short.median()
        rank_long  = fb30_long.fillna(med_l)
        rank_short = fb30_short.fillna(med_s)
    else:
        rank_long  = sent_score.reindex(long_uni).fillna(sent_score.reindex(long_uni).median())
        rank_short = sent_score.reindex(short_uni).fillna(sent_score.reindex(short_uni).median())

    final_longs  = rank_long.nlargest(N_LONG).index
    final_shorts = rank_short.nsmallest(N_SHORT).index

    all_t = final_longs.union(final_shorts)
    w = pd.Series(0.0, index=all_t)
    w.loc[final_longs]  =  1 / max(len(final_longs),  1)
    w.loc[final_shorts] = -1 / max(len(final_shorts), 1)
    gross = w.abs().sum()
    if gross > 0:
        w /= gross

    turnover = (w - last_w.reindex(w.index, fill_value=0)).abs().sum()

    i        = np.where(rebal_dates == d)[0][0] + 1
    next_cut = rebal_dates[i] if i < len(rebal_dates) else dates[-1]
    span     = dates[(dates >= d) & (dates < next_cut)]

    daily_rets = {}
    for dd in span:
        daily_rets[dd] = (w * panel.loc[(dd, w.index), 'ret_t1'].fillna(0.0)).sum()
    if len(span) > 0:
        daily_rets[span[0]] -= turnover * TC_BPS / 10_000

    return daily_rets, w

def run_cascade(panel, K, pure_sent_rerank=False):
    label = f'cascade_K{K}{"_puresent" if pure_sent_rerank else ""}'
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

        X_trad = train[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        X_sent = train[SENT_FEATS]
        sq1, sq9 = X_sent.quantile(0.01), X_sent.quantile(0.99)

        ridge, sc, hgb, rs_mean, rs_std = fit_ridge_and_hgb(train, q1, q9, sq1, sq9)

        try:
            today_trad = panel.loc[(d, slice(None)), TRAD_FEATS].copy()
        except KeyError:
            continue
        today_trad = today_trad.dropna(subset=TRAD_FEATS).clip(lower=q1, upper=q9, axis=1)
        if today_trad.empty:
            continue

        tickers_trad = today_trad.index.get_level_values('ticker')
        ridge_today  = pd.Series(
            ridge.predict(sc.transform(today_trad.values)), index=tickers_trad
        )
        ridge_today_z = (ridge_today - rs_mean) / rs_std

        try:
            today_sent = panel.loc[(d, slice(None)), SENT_FEATS].copy()
        except KeyError:
            today_sent = pd.DataFrame()

        today_fb30 = None
        if not today_sent.empty:
            tickers_sent  = today_sent.index.get_level_values('ticker')
            today_fb30    = today_sent['finbert_sent_30d'].copy()
            today_fb30.index = tickers_sent
            today_sent_cl = today_sent.clip(lower=sq1, upper=sq9, axis=1)
            today2 = today_sent_cl.copy()
            today2.insert(0, 'ridge_score_z', ridge_today_z.reindex(tickers_sent).values)
            hgb_resid = pd.Series(
                hgb.predict(today2.values), index=tickers_sent
            ).replace([np.inf, -np.inf], np.nan)
            sent_score = ridge_today.copy()
            sent_score.update(ridge_today.reindex(tickers_sent) + hgb_resid)
        else:
            sent_score = ridge_today

        daily_rets, last_w = apply_cascade_port(
            ridge_today, sent_score, dates, d, rebal_dates, panel,
            last_w, K, pure_sent_rerank, today_fb30
        )
        for dd, r in daily_rets.items():
            port.loc[dd] = r

    return summarize(port, label)

def run_stacked_residual_with_interaction(panel):
    """Stacked residual + ridge_score_z * finbert_30d_z interaction."""
    label       = 'stacked_residual_interaction'
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

        X_trad = train[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        X_tc   = X_trad.clip(lower=q1, upper=q9, axis=1)
        y      = train.loc[X_tc.index, 'ret_t1'].values

        sc    = StandardScaler()
        ridge = Ridge(alpha=RIDGE_ALPHA)
        ridge.fit(sc.fit_transform(X_tc.values), y)
        ridge_pred = ridge.predict(sc.transform(X_tc.values))

        rs_mean = ridge_pred.mean()
        rs_std  = ridge_pred.std() + 1e-8
        ridge_z = (ridge_pred - rs_mean) / rs_std

        X_sent = train[SENT_FEATS].copy()
        sq1, sq9 = X_sent.quantile(0.01), X_sent.quantile(0.99)
        X_sent   = X_sent.clip(lower=sq1, upper=sq9, axis=1)

        fb30      = train['finbert_sent_30d'].values
        fb30_mean = np.nanmean(fb30)
        fb30_std  = np.nanstd(fb30) + 1e-8
        fb30_z    = (fb30 - fb30_mean) / fb30_std                 

        X2 = X_sent.copy()
        X2.insert(0, 'ridge_score_z', ridge_z)
        X2['ridge_x_fb30'] = ridge_z * np.nan_to_num(fb30_z, nan=0.0)
        y_res = y - ridge_pred

        hgb = HistGradientBoostingRegressor(**HGB_PARAMS)
        hgb.fit(X2.values, y_res)

        try:
            today_trad = panel.loc[(d, slice(None)), TRAD_FEATS].copy()
        except KeyError:
            continue
        today_trad = today_trad.dropna(subset=TRAD_FEATS).clip(lower=q1, upper=q9, axis=1)
        if today_trad.empty:
            continue

        tickers_trad = today_trad.index.get_level_values('ticker')
        ridge_today  = pd.Series(
            ridge.predict(sc.transform(today_trad.values)), index=tickers_trad
        )
        ridge_today_z = (ridge_today - rs_mean) / rs_std

        try:
            today_sent = panel.loc[(d, slice(None)), SENT_FEATS].copy()
        except KeyError:
            continue
        tickers_sent = today_sent.index.get_level_values('ticker')
        today_sent   = today_sent.loc[
            today_sent.index.get_level_values('ticker').isin(tickers_trad)
        ].clip(lower=sq1, upper=sq9, axis=1)
        tickers_sent = today_sent.index.get_level_values('ticker')

        fb30_today   = today_sent['finbert_sent_30d'].values
        fb30_today_z = (fb30_today - fb30_mean) / fb30_std

        today2 = today_sent.copy()
        today2.insert(0, 'ridge_score_z', ridge_today_z.reindex(tickers_sent).values)
        today2['ridge_x_fb30'] = (
            ridge_today_z.reindex(tickers_sent).values * np.nan_to_num(fb30_today_z, nan=0.0)
        )

        hgb_pred = pd.Series(
            hgb.predict(today2.values), index=tickers_sent
        ).replace([np.inf, -np.inf], np.nan).dropna()

        common = hgb_pred.index.intersection(ridge_today.index)
        pred   = (ridge_today.reindex(common) + hgb_pred.reindex(common)).dropna()
        if pred.empty:
            continue

        longs  = pred.nlargest(N_LONG).index
        shorts = pred.nsmallest(N_SHORT).index
        w      = pd.Series(0.0, index=pred.index)
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

    return summarize(port, label)

if __name__ == '__main__':
    df = pd.read_parquet(DATA_DIR / 'model_table.parquet').sort_index()
    df = df.dropna(subset=['ret_t1'])
    df = df[df.index.get_level_values('date') >= '2017-01-01']
    print(f'Panel: {len(df):,} rows | eval from {EVAL_START.date()}\n')

    results = {}

    print('Cascade K=150 (ridge+hgb re-rank)')
    results['cascade_K150']         = run_cascade(df, K=150, pure_sent_rerank=False)

    print('\nCascade K=125')
    results['cascade_K125']         = run_cascade(df, K=125, pure_sent_rerank=False)

    print('\nCascade K=100')
    results['cascade_K100']         = run_cascade(df, K=100, pure_sent_rerank=False)

    print('\nCascade K=75')
    results['cascade_K75']          = run_cascade(df, K=75,  pure_sent_rerank=False)

    print('\nCascade K=100, pure-sentiment re-rank')
    results['cascade_K100_puresent'] = run_cascade(df, K=100, pure_sent_rerank=True)

    print('\nCascade K=150, pure-sentiment re-rank')
    results['cascade_K150_puresent'] = run_cascade(df, K=150, pure_sent_rerank=True)

    print('\nStacked residual + ridge x sentiment interaction')
    results['stacked_res_interact'] = run_stacked_residual_with_interaction(df)

    print('\nResults:')
    rows = [{'model': k, **v} for k, v in results.items()]
    res  = pd.DataFrame(rows)
    for c in ['ann_ret', 'ann_vol', 'max_dd']:
        res[c] = (res[c] * 100).round(2).astype(str) + '%'
    res['sharpe'] = res['sharpe'].round(4)
    print(res.to_string(index=False))

    out = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
    out.to_parquet(DATA_DIR / 'backtest_tweak_search.parquet', index=False)
    out.to_csv(DATA_DIR    / 'backtest_tweak_search.csv',       index=False)
    print('\nSaved backtest_tweak_search.parquet / .csv')
