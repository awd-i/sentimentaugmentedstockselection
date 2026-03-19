# Stacked Ridge->HGB v2: sentiment-filtered training, cascade portfolio, interaction feature.
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
CASCADE_K  = 150                                                                  

TRAD_FEATS = ['momentum', 'value_composite', 'quality_composite']
SENT_FEATS = ['finbert_sent_7d', 'finbert_sent_30d', 'finbert_sent_60d',
              'sent_momentum', 'news_volume_7d', 'news_volume_30d']

HGB_BASE = dict(
    max_iter          = 150,
    max_depth         = 4,
    learning_rate     = 0.05,
    min_samples_leaf  = 50,
    l2_regularization = 1.0,
    random_state      = 42,
)
HGB_AGGRESSIVE = dict(
    max_iter          = 300,
    max_depth         = 5,
    learning_rate     = 0.04,
    min_samples_leaf  = 20,
    l2_regularization = 0.5,
    random_state      = 42,
)
RIDGE_ALPHA = 50.0

def zscore(s, eps=1e-8):
    return (s - s.mean()) / (s.std() + eps)

def build_port(pred, last_w, dates, d, rebal_dates, panel):
    longs  = pred.nlargest(N_LONG).index
    shorts = pred.nsmallest(N_SHORT).index
    w = pd.Series(0.0, index=pred.index)
    w.loc[longs]  =  1 / max(len(longs),  1)
    w.loc[shorts] = -1 / max(len(shorts), 1)
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
    return perf, cum, stats

def fit_ridge(train, q1, q9):
    X_trad = train[TRAD_FEATS].clip(lower=q1, upper=q9, axis=1)
    y      = train.loc[X_trad.index, 'ret_t1'].values
    sc     = StandardScaler()
    ridge  = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(sc.fit_transform(X_trad.values), y)
    pred_train = ridge.predict(sc.transform(X_trad.values))
    return ridge, sc, pred_train, y

def run_ridge(panel):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='ridge_baseline'):
        train = panel.loc[
            panel.index.get_level_values('date') < d
        ].dropna(subset=['ret_t1'] + TRAD_FEATS)
        if len(train) < MIN_TRAIN:
            continue

        X_trad = train[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        ridge, sc, _, _ = fit_ridge(train, q1, q9)

        try:
            today = panel.loc[(d, slice(None)), TRAD_FEATS].copy()
        except KeyError:
            continue
        today = today.dropna(subset=TRAD_FEATS).clip(lower=q1, upper=q9, axis=1)
        if today.empty:
            continue

        pred = pd.Series(
            ridge.predict(sc.transform(today.values)),
            index=today.index.get_level_values('ticker')
        ).replace([np.inf, -np.inf], np.nan).dropna()

        daily_rets, last_w = build_port(pred, last_w, dates, d, rebal_dates, panel)
        for dd, r in daily_rets.items():
            port.loc[dd] = r

    return summarize(port, 'ridge_baseline')

def run_stacked_A(panel):
    label       = 'stacked_A_sent_filtered'
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc=label):
        train_all = panel.loc[
            panel.index.get_level_values('date') < d
        ].dropna(subset=['ret_t1'] + TRAD_FEATS)
        if len(train_all) < MIN_TRAIN:
            continue

        X_trad = train_all[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        ridge, sc, ridge_all_pred, _ = fit_ridge(train_all, q1, q9)

        train_sent = train_all[train_all['finbert_sent_30d'].notna()].copy()
                                                     
        if len(train_sent) < 10_000:
            continue

        ridge_sent_idx = train_all.index.get_loc                 
                                                     
        sent_in_all = train_all.index.isin(train_sent.index)
        ridge_sent_pred = ridge_all_pred[sent_in_all]

        rs_mean = ridge_sent_pred.mean()
        rs_std  = ridge_sent_pred.std() + 1e-8
        ridge_sent_z = (ridge_sent_pred - rs_mean) / rs_std

        fb30 = train_sent['finbert_sent_30d'].values
        fb30_mean, fb30_std = fb30.mean(), fb30.std() + 1e-8
        fb30_z = (fb30 - fb30_mean) / fb30_std

        X_sent = train_sent[SENT_FEATS].copy()
        sq1, sq9 = X_sent.quantile(0.01), X_sent.quantile(0.99)
        X_sent = X_sent.clip(lower=sq1, upper=sq9, axis=1)

        interaction = ridge_sent_z * fb30_z

        X2 = X_sent.copy()
        X2.insert(0, 'ridge_score_z', ridge_sent_z)
        X2['ridge_x_sent'] = interaction
        y2 = (train_sent['ret_t1'].values - ridge_sent_pred)             

        hgb = HistGradientBoostingRegressor(**HGB_AGGRESSIVE)
        hgb.fit(X2.values, y2)

        try:
            today_trad = panel.loc[(d, slice(None)), TRAD_FEATS].copy()
        except KeyError:
            continue
        today_trad = today_trad.dropna(subset=TRAD_FEATS).clip(lower=q1, upper=q9, axis=1)
        if today_trad.empty:
            continue

        tickers_trad  = today_trad.index.get_level_values('ticker')
        ridge_today   = pd.Series(
            ridge.predict(sc.transform(today_trad.values)), index=tickers_trad
        )
        ridge_today_z = (ridge_today - rs_mean) / rs_std

        try:
            today_sent = panel.loc[(d, slice(None)), SENT_FEATS].copy()
        except KeyError:
            today_sent = pd.DataFrame()

        if not today_sent.empty:
            tickers_sent = today_sent.index.get_level_values('ticker')
            has_fb30     = today_sent['finbert_sent_30d'].notna()

            today_sent_clipped = today_sent.clip(lower=sq1, upper=sq9, axis=1)
            today2 = today_sent_clipped.copy()
            today2.insert(0, 'ridge_score_z', ridge_today_z.reindex(tickers_sent).values)

            fb30_today = today_sent['finbert_sent_30d'].values
            fb30_today_z = (fb30_today - fb30_mean) / fb30_std
            today2['ridge_x_sent'] = (
                ridge_today_z.reindex(tickers_sent).values * fb30_today_z
            )

            hgb_pred = pd.Series(
                hgb.predict(today2.values), index=tickers_sent
            ).replace([np.inf, -np.inf], np.nan)

            pred = ridge_today.copy()
            with_sent = tickers_sent[has_fb30.values]
            stacked_full = ridge_today.reindex(with_sent) + hgb_pred.reindex(with_sent)
            pred.update(stacked_full)
        else:
            pred = ridge_today

        pred = pred.dropna()
        if pred.empty:
            continue

        daily_rets, last_w = build_port(pred, last_w, dates, d, rebal_dates, panel)
        for dd, r in daily_rets.items():
            port.loc[dd] = r

    return summarize(port, label)

def run_stacked_B(panel):
    label       = 'stacked_B_cascade'
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc=label):
        train_all = panel.loc[
            panel.index.get_level_values('date') < d
        ].dropna(subset=['ret_t1'] + TRAD_FEATS)
        if len(train_all) < MIN_TRAIN:
            continue

        X_trad = train_all[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        ridge, sc, ridge_all_pred, _ = fit_ridge(train_all, q1, q9)

        rs_mean = ridge_all_pred.mean()
        rs_std  = ridge_all_pred.std() + 1e-8
        ridge_all_z = (ridge_all_pred - rs_mean) / rs_std

        X_sent = train_all[SENT_FEATS].copy()
        sq1, sq9 = X_sent.quantile(0.01), X_sent.quantile(0.99)
        X_sent = X_sent.clip(lower=sq1, upper=sq9, axis=1)
        X2 = X_sent.copy()
        X2.insert(0, 'ridge_score_z', ridge_all_z)
        y_res = train_all['ret_t1'].values - ridge_all_pred

        hgb = HistGradientBoostingRegressor(**HGB_BASE)
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

        n_stocks = len(ridge_today)
        k_long   = min(CASCADE_K, n_stocks // 2)
        k_short  = min(CASCADE_K, n_stocks // 2)
        long_universe  = ridge_today.nlargest(k_long).index
        short_universe = ridge_today.nsmallest(k_short).index

        try:
            today_sent = panel.loc[(d, slice(None)), SENT_FEATS].copy()
        except KeyError:
            today_sent = pd.DataFrame()

        ridge_today_z = (ridge_today - rs_mean) / rs_std

        if not today_sent.empty:
            tickers_sent = today_sent.index.get_level_values('ticker')
            today_sent_cl = today_sent.clip(lower=sq1, upper=sq9, axis=1)
            today2 = today_sent_cl.copy()
            today2.insert(0, 'ridge_score_z', ridge_today_z.reindex(tickers_sent).values)
            hgb_resid = pd.Series(
                hgb.predict(today2.values), index=tickers_sent
            ).replace([np.inf, -np.inf], np.nan)
            sent_score = ridge_today.reindex(tickers_sent) + hgb_resid.reindex(tickers_sent)
        else:
            sent_score = ridge_today

        long_scores  = sent_score.reindex(long_universe).fillna(
            sent_score.reindex(long_universe).median()
        )
        short_scores = sent_score.reindex(short_universe).fillna(
            sent_score.reindex(short_universe).median()
        )

        final_longs  = long_scores.nlargest(N_LONG).index
        final_shorts = short_scores.nsmallest(N_SHORT).index

        all_tickers = final_longs.union(final_shorts)
        w = pd.Series(0.0, index=all_tickers)
        w.loc[final_longs]  =  1 / max(len(final_longs),  1)
        w.loc[final_shorts] = -1 / max(len(final_shorts), 1)
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

def run_stacked_C(panel):
    label       = 'stacked_C_A_plus_B'
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc=label):
        train_all = panel.loc[
            panel.index.get_level_values('date') < d
        ].dropna(subset=['ret_t1'] + TRAD_FEATS)
        if len(train_all) < MIN_TRAIN:
            continue

        X_trad = train_all[TRAD_FEATS]
        q1, q9 = X_trad.quantile(0.01), X_trad.quantile(0.99)
        ridge, sc, ridge_all_pred, _ = fit_ridge(train_all, q1, q9)

        train_sent = train_all[train_all['finbert_sent_30d'].notna()].copy()
        if len(train_sent) < 10_000:
            continue

        sent_in_all     = train_all.index.isin(train_sent.index)
        ridge_sent_pred = ridge_all_pred[sent_in_all]
        rs_mean = ridge_sent_pred.mean()
        rs_std  = ridge_sent_pred.std() + 1e-8
        ridge_sent_z = (ridge_sent_pred - rs_mean) / rs_std

        fb30 = train_sent['finbert_sent_30d'].values
        fb30_mean, fb30_std = fb30.mean(), fb30.std() + 1e-8
        fb30_z = (fb30 - fb30_mean) / fb30_std

        X_sent = train_sent[SENT_FEATS].copy()
        sq1, sq9 = X_sent.quantile(0.01), X_sent.quantile(0.99)
        X_sent = X_sent.clip(lower=sq1, upper=sq9, axis=1)

        X2 = X_sent.copy()
        X2.insert(0, 'ridge_score_z', ridge_sent_z)
        X2['ridge_x_sent'] = ridge_sent_z * fb30_z
        y2 = train_sent['ret_t1'].values - ridge_sent_pred

        hgb = HistGradientBoostingRegressor(**HGB_AGGRESSIVE)
        hgb.fit(X2.values, y2)

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

        if not today_sent.empty:
            tickers_sent  = today_sent.index.get_level_values('ticker')
            has_fb30      = today_sent['finbert_sent_30d'].notna()
            today_sent_cl = today_sent.clip(lower=sq1, upper=sq9, axis=1)

            today2 = today_sent_cl.copy()
            today2.insert(0, 'ridge_score_z', ridge_today_z.reindex(tickers_sent).values)
            fb30_today   = today_sent['finbert_sent_30d'].values
            fb30_today_z = (fb30_today - fb30_mean) / fb30_std
            today2['ridge_x_sent'] = (
                ridge_today_z.reindex(tickers_sent).values * fb30_today_z
            )

            hgb_pred = pd.Series(
                hgb.predict(today2.values), index=tickers_sent
            ).replace([np.inf, -np.inf], np.nan)

            sent_score = ridge_today.copy()
            with_sent  = tickers_sent[has_fb30.values]
            stacked    = ridge_today.reindex(with_sent) + hgb_pred.reindex(with_sent)
            sent_score.update(stacked)
        else:
            sent_score = ridge_today

        n_stocks     = len(ridge_today)
        k            = min(CASCADE_K, n_stocks // 2)
        long_uni     = ridge_today.nlargest(k).index
        short_uni    = ridge_today.nsmallest(k).index

        long_scores  = sent_score.reindex(long_uni).fillna(
            sent_score.reindex(long_uni).median()
        )
        short_scores = sent_score.reindex(short_uni).fillna(
            sent_score.reindex(short_uni).median()
        )

        final_longs  = long_scores.nlargest(N_LONG).index
        final_shorts = short_scores.nsmallest(N_SHORT).index

        all_tickers = final_longs.union(final_shorts)
        w = pd.Series(0.0, index=all_tickers)
        w.loc[final_longs]  =  1 / max(len(final_longs),  1)
        w.loc[final_shorts] = -1 / max(len(final_shorts), 1)
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
    print(f'Panel: {len(df):,} rows | '
          f'{df.index.get_level_values("date").min().date()} -> '
          f'{df.index.get_level_values("date").max().date()}')
    print(f'Rebal: every {REBAL_FREQ} days | eval from {EVAL_START.date()} | '
          f'burn-in: {MIN_TRAIN:,} rows')
    print(f'Cascade K={CASCADE_K}: Ridge pre-filters top/bottom {CASCADE_K}, '
          f'sentiment re-ranks to {N_LONG}/{N_SHORT}\n')

    print('0. Ridge baseline')
    _, _, stats_ridge = run_ridge(df)

    print('\nVariant A: sent-filtered HGB + interaction + aggressive params')
    _, _, stats_A = run_stacked_A(df)

    print('\nVariant B: cascade (Ridge pre-filter + sentiment re-rank)')
    _, _, stats_B = run_stacked_B(df)

    print('\nVariant C: A + B combined')
    _, _, stats_C = run_stacked_C(df)

    print('\nFinal results:')
    res = pd.DataFrame([
        {'model': 'ridge_baseline', **stats_ridge},
        {'model': 'stacked_A',      **stats_A},
        {'model': 'stacked_B',      **stats_B},
        {'model': 'stacked_C',      **stats_C},
    ])
    for c in ['ann_ret', 'ann_vol', 'max_dd']:
        res[c] = (res[c] * 100).round(2).astype(str) + '%'
    res['sharpe'] = res['sharpe'].round(4)
    print(res.to_string(index=False))

    out = pd.DataFrame([
        {'model': 'ridge_baseline', **stats_ridge},
        {'model': 'stacked_A',      **stats_A},
        {'model': 'stacked_B',      **stats_B},
        {'model': 'stacked_C',      **stats_C},
    ])
    out.to_parquet(DATA_DIR / 'backtest_summary_stacked_v2.parquet', index=False)
    out.to_csv(DATA_DIR    / 'backtest_summary_stacked_v2.csv',       index=False)
    print('\nSaved backtest_summary_stacked_v2.parquet / .csv')
