# Final comparison of all 6 models: biweekly rebalancing, evaluated 2021+.
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'intermediate'

REBAL_FREQ  = 10
MIN_TRAIN   = 250_000
EVAL_START  = pd.Timestamp('2021-01-01')
TC_BPS      = 10
N_LONG      = 50
N_SHORT     = 50
CASCADE_K   = 150                                                 

RIDGE_ALPHA = 50.0
HGB_PARAMS  = dict(
    max_iter          = 150,
    max_depth         = 4,
    learning_rate     = 0.05,
    min_samples_leaf  = 50,
    l2_regularization = 1.0,
    random_state      = 42,
)

TRAD_FEATS = ['momentum', 'value_composite', 'quality_composite']
SENT_FEATS = ['finbert_sent_7d', 'finbert_sent_30d', 'finbert_sent_60d',
              'sent_momentum', 'news_volume_7d', 'news_volume_30d']

RIDGE_ALPHAS = [10, 50, 100, 300]
HGB_GRID = [
    dict(max_iter=150, max_depth=d, learning_rate=lr, min_samples_leaf=msl,
         l2_regularization=1.0, random_state=42)
    for d   in [3, 4]
    for lr  in [0.03, 0.05]
    for msl in [25, 50]
]                   

def stats_and_print(port, label):
    perf    = port.loc[port.index >= EVAL_START].dropna()
    cum     = (1 + perf).cumprod()
    ann_ret = perf.mean() * 252
    ann_vol = perf.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = (cum / cum.cummax() - 1).min()
    print(f'  [{label}] ann_ret={ann_ret*100:.2f}%  vol={ann_vol*100:.2f}%  '
          f'sharpe={sharpe:.3f}  max_dd={max_dd*100:.2f}%')
    stats = {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}
    return stats, perf

def apply_weights(w, last_w, dates, d, rebal_dates, panel, port):
    turnover = (w - last_w.reindex(w.index, fill_value=0)).abs().sum()
    i        = np.where(rebal_dates == d)[0][0] + 1
    next_cut = rebal_dates[i] if i < len(rebal_dates) else dates[-1]
    span     = dates[(dates >= d) & (dates < next_cut)]
    for dd in span:
        port.loc[dd] = (w * panel.loc[(dd, w.index), 'ret_t1'].fillna(0.0)).sum()
    if len(span) > 0:
        port.loc[span[0]] -= turnover * TC_BPS / 10_000
    return w                         

def build_weights(pred):
    longs  = pred.nlargest(N_LONG).index
    shorts = pred.nsmallest(N_SHORT).index
    w      = pd.Series(0.0, index=pred.index)
    w.loc[longs]  =  1 / max(len(longs),  1)
    w.loc[shorts] = -1 / max(len(shorts), 1)
    gross = w.abs().sum()
    if gross > 0:
        w /= gross
    return w

def clip_quantile(df, q1=None, q9=None):
    if q1 is None:
        q1, q9 = df.quantile(0.01), df.quantile(0.99)
    return df.clip(lower=q1, upper=q9, axis=1), q1, q9

def setup(panel, d):
    """Return train slice or None if burn-in not met."""
    train = panel.loc[
        panel.index.get_level_values('date') < d
    ].dropna(subset=['ret_t1'] + TRAD_FEATS)
    return train if len(train) >= MIN_TRAIN else None

def get_today(panel, d, feats, q1, q9, drop_trad=True):
    try:
        today = panel.loc[(d, slice(None)), feats].copy()
    except KeyError:
        return None
    if drop_trad:
        today = today.dropna(subset=TRAD_FEATS)
    return today.clip(lower=q1, upper=q9, axis=1) if not today.empty else None

def _get_params(tune_schedule, year):
    """Look up tuned params for a given year, falling back to defaults."""
    p = tune_schedule.get(year)
    if p is None:
        return RIDGE_ALPHA, HGB_PARAMS
    return p['ridge_alpha'], p['hgb_params']

def _cross_section_ic(pred_arr, y_arr, date_arr):
    """Mean per-date Spearman rank IC (cross-sectional information coefficient)."""
    ics = []
    for d in np.unique(date_arr):
        mask = date_arr == d
        if mask.sum() < 10:
            continue
        ic, _ = spearmanr(pred_arr[mask], y_arr[mask])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0

def _tune_year(df, val_start, val_end):
    """
    Tune ridge_alpha and hgb_params by maximising rank IC on [val_start, val_end].
    Training uses all data from 2017 up to val_start (entirely pre-test).
    """
    train_start = pd.Timestamp('2017-01-01')
    dates_col   = df.index.get_level_values('date')

    train = df.loc[(dates_col >= train_start) & (dates_col < val_start)
                   ].dropna(subset=['ret_t1'] + TRAD_FEATS)
    val   = df.loc[(dates_col >= val_start)   & (dates_col <= val_end)
                   ].dropna(subset=['ret_t1'] + TRAD_FEATS)

    if len(train) < 20_000 or len(val) < 5_000:
        return {'ridge_alpha': RIDGE_ALPHA, 'hgb_params': HGB_PARAMS}

    y_tr      = train['ret_t1'].values
    y_vl      = val['ret_t1'].values
    val_dates = val.index.get_level_values('date').values

    X_tr_t, q1, q9 = clip_quantile(train[TRAD_FEATS])
    X_vl_t         = val[TRAD_FEATS].clip(lower=q1, upper=q9, axis=1)
    sc_t = StandardScaler().fit(X_tr_t.values)

    best_r_alpha, best_r_ic = RIDGE_ALPHA, -np.inf
    for alpha in RIDGE_ALPHAS:
        pred = Ridge(alpha=alpha).fit(
            sc_t.transform(X_tr_t.values), y_tr
        ).predict(sc_t.transform(X_vl_t.values))
        ic = _cross_section_ic(pred, y_vl, val_dates)
        if ic > best_r_ic:
            best_r_ic, best_r_alpha = ic, alpha

    all_feats   = TRAD_FEATS + SENT_FEATS
    X_tr_h, sq1, sq9 = clip_quantile(train[all_feats])
    X_vl_h           = val[all_feats].clip(lower=sq1, upper=sq9, axis=1)

    best_h_params, best_h_ic = HGB_PARAMS, -np.inf
    for params in HGB_GRID:
        pred = HistGradientBoostingRegressor(**params).fit(
            X_tr_h.values, y_tr
        ).predict(X_vl_h.values)
        ic = _cross_section_ic(pred, y_vl, val_dates)
        if ic > best_h_ic:
            best_h_ic, best_h_params = ic, params

    print(f'  {val_start.date()} → {val_end.date()} | '
          f'Ridge α={best_r_alpha} (IC={best_r_ic:.4f}) | '
          f'HGB depth={best_h_params["max_depth"]} '
          f'lr={best_h_params["learning_rate"]} '
          f'msl={best_h_params["min_samples_leaf"]} (IC={best_h_ic:.4f})')

    return {'ridge_alpha': best_r_alpha, 'hgb_params': best_h_params}

def build_tune_schedule(df):
    """Tune hyperparams once on pre-test data (train 2017-2019, val 2020); lock for 2021+."""
    val_start = EVAL_START - pd.DateOffset(years=1)               
    val_end   = EVAL_START - pd.Timedelta(days=1)                 

    print('Hyperparameter tuning (pre-test only: train 2017-2019, val 2020)')
    best = _tune_year(df, val_start, val_end)
    print()

    last_year = df.index.get_level_values('date').max().year
    return {yr: best for yr in range(EVAL_START.year, last_year + 2)}

def run_ridge_baseline(panel, tune_schedule):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='ridge_baseline'):
        train = setup(panel, d)
        if train is None:
            continue
        ridge_alpha, _ = _get_params(tune_schedule, d.year)

        X, q1, q9 = clip_quantile(train[TRAD_FEATS])
        y  = train['ret_t1'].values
        sc = StandardScaler()
        m  = Ridge(alpha=ridge_alpha).fit(sc.fit_transform(X.values), y)

        today = get_today(panel, d, TRAD_FEATS, q1, q9)
        if today is None or today.empty:
            continue
        pred = pd.Series(m.predict(sc.transform(today.values)),
                         index=today.index.get_level_values('ticker')
                         ).replace([np.inf, -np.inf], np.nan).dropna()
        last_w = apply_weights(build_weights(pred), last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'ridge_baseline')

def run_ridge_augmented(panel, tune_schedule):
    all_feats   = TRAD_FEATS + SENT_FEATS
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='ridge_augmented'):
        train = setup(panel, d)
        if train is None:
            continue
        ridge_alpha, _ = _get_params(tune_schedule, d.year)

        X, q1, q9 = clip_quantile(train[all_feats])
        col_means = X.mean()
        X_imp     = X.fillna(col_means)
        y  = train['ret_t1'].values
        sc = StandardScaler()
        m  = Ridge(alpha=ridge_alpha).fit(sc.fit_transform(X_imp.values), y)

        today = get_today(panel, d, all_feats, q1, q9)
        if today is None or today.empty:
            continue
        today_imp = today.fillna(col_means)
        pred = pd.Series(m.predict(sc.transform(today_imp.values)),
                         index=today.index.get_level_values('ticker')
                         ).replace([np.inf, -np.inf], np.nan).dropna()
        last_w = apply_weights(build_weights(pred), last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'ridge_augmented')

def run_hgb_baseline(panel, tune_schedule):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='hgb_baseline'):
        train = setup(panel, d)
        if train is None:
            continue
        _, hgb_params = _get_params(tune_schedule, d.year)

        X, q1, q9 = clip_quantile(train[TRAD_FEATS])
        y = train['ret_t1'].values
        m = HistGradientBoostingRegressor(**hgb_params).fit(X.values, y)

        today = get_today(panel, d, TRAD_FEATS, q1, q9)
        if today is None or today.empty:
            continue
        pred = pd.Series(m.predict(today.values),
                         index=today.index.get_level_values('ticker')
                         ).replace([np.inf, -np.inf], np.nan).dropna()
        last_w = apply_weights(build_weights(pred), last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'hgb_baseline')

def run_hgb_augmented(panel, tune_schedule):
    all_feats   = TRAD_FEATS + SENT_FEATS
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='hgb_augmented'):
        train = setup(panel, d)
        if train is None:
            continue
        _, hgb_params = _get_params(tune_schedule, d.year)

        X, q1, q9 = clip_quantile(train[all_feats])
        y = train['ret_t1'].values
        m = HistGradientBoostingRegressor(**hgb_params).fit(X.values, y)

        today = get_today(panel, d, all_feats, q1, q9)
        if today is None or today.empty:
            continue
        pred = pd.Series(m.predict(today.values),
                         index=today.index.get_level_values('ticker')
                         ).replace([np.inf, -np.inf], np.nan).dropna()
        last_w = apply_weights(build_weights(pred), last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'hgb_augmented')

def run_stacked_residual(panel, tune_schedule):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='stacked_residual'):
        train = setup(panel, d)
        if train is None:
            continue
        ridge_alpha, hgb_params = _get_params(tune_schedule, d.year)

        X_t, q1, q9 = clip_quantile(train[TRAD_FEATS])
        y   = train['ret_t1'].values
        sc  = StandardScaler()
        r   = Ridge(alpha=ridge_alpha).fit(sc.fit_transform(X_t.values), y)
        r_pred = r.predict(sc.transform(X_t.values))

        rs_mean, rs_std = r_pred.mean(), r_pred.std() + 1e-8

        X_s, sq1, sq9 = clip_quantile(train[SENT_FEATS])
        X2  = X_s.copy()
        X2.insert(0, 'ridge_score_z', (r_pred - rs_mean) / rs_std)
        hgb = HistGradientBoostingRegressor(**hgb_params).fit(X2.values, y - r_pred)

        today_t = get_today(panel, d, TRAD_FEATS, q1, q9)
        if today_t is None or today_t.empty:
            continue
        tickers = today_t.index.get_level_values('ticker')
        r_today = pd.Series(r.predict(sc.transform(today_t.values)), index=tickers)
        r_today_z = (r_today - rs_mean) / rs_std

        try:
            today_s = panel.loc[(d, slice(None)), SENT_FEATS].copy()
        except KeyError:
            continue
        today_s = today_s.loc[
            today_s.index.get_level_values('ticker').isin(tickers)
        ].clip(lower=sq1, upper=sq9, axis=1)
        t_sent = today_s.index.get_level_values('ticker')

        today2 = today_s.copy()
        today2.insert(0, 'ridge_score_z', r_today_z.reindex(t_sent).values)
        h_pred = pd.Series(hgb.predict(today2.values), index=t_sent
                           ).replace([np.inf, -np.inf], np.nan).dropna()

        common = h_pred.index.intersection(r_today.index)
        pred   = (r_today.reindex(common) + h_pred.reindex(common)).dropna()
        if pred.empty:
            continue
        last_w = apply_weights(build_weights(pred), last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'stacked_residual')

def run_cascade_puresent(panel, tune_schedule):
    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), REBAL_FREQ)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc='cascade_puresent'):
        train = setup(panel, d)
        if train is None:
            continue
        ridge_alpha, _ = _get_params(tune_schedule, d.year)

        X_t, q1, q9 = clip_quantile(train[TRAD_FEATS])
        y  = train['ret_t1'].values
        sc = StandardScaler()
        r  = Ridge(alpha=ridge_alpha).fit(sc.fit_transform(X_t.values), y)

        today_t = get_today(panel, d, TRAD_FEATS, q1, q9)
        if today_t is None or today_t.empty:
            continue
        tickers     = today_t.index.get_level_values('ticker')
        ridge_today = pd.Series(r.predict(sc.transform(today_t.values)), index=tickers)

        k          = min(CASCADE_K, len(ridge_today) // 2)
        long_uni   = ridge_today.nlargest(k).index
        short_uni  = ridge_today.nsmallest(k).index

        try:
            fb30_today = panel.loc[(d, slice(None)), 'finbert_sent_30d'].copy()
            fb30_today.index = fb30_today.index.get_level_values('ticker')
        except KeyError:
            fb30_today = pd.Series(dtype=float)

        def rerank(universe, ascending):
            fb30  = fb30_today.reindex(universe)
            ridge = ridge_today.reindex(universe)
            fb_min, fb_max = fb30.min(), fb30.max()
            if pd.isna(fb_min) or fb_min == fb_max:
                score = ridge
            else:
                r_min, r_max = ridge.min(), ridge.max()
                ridge_scaled = (ridge - r_min) / (r_max - r_min + 1e-8) * (fb_max - fb_min) + fb_min
                score = fb30.fillna(ridge_scaled)
            return score.nsmallest(N_SHORT).index if ascending else score.nlargest(N_LONG).index

        final_longs  = rerank(long_uni,  ascending=False)
        final_shorts = rerank(short_uni, ascending=True)

        all_t = final_longs.union(final_shorts)
        w     = pd.Series(0.0, index=all_t)
        w.loc[final_longs]  =  1 / max(len(final_longs),  1)
        w.loc[final_shorts] = -1 / max(len(final_shorts), 1)
        gross = w.abs().sum()
        if gross > 0:
            w /= gross

        last_w = apply_weights(w, last_w, dates, d, rebal_dates, panel, port)

    return stats_and_print(port, 'cascade_puresent')

if __name__ == '__main__':
    df = pd.read_parquet(DATA_DIR / 'model_table.parquet').sort_index()
    df = df.dropna(subset=['ret_t1'])
    df = df[df.index.get_level_values('date') >= '2017-01-01']
    print(f'Panel: {len(df):,} rows | '
          f'{df.index.get_level_values("date").min().date()} → '
          f'{df.index.get_level_values("date").max().date()}')
    print(f'Rebal every {REBAL_FREQ} days | eval from {EVAL_START.date()} | '
          f'burn-in {MIN_TRAIN:,} rows\n')

    tune_schedule = build_tune_schedule(df)

    results = {}
    series  = {}

    print('1. Ridge baseline')
    results['ridge_baseline'],  series['ridge_baseline']  = run_ridge_baseline(df, tune_schedule)

    print('\n2. Ridge augmented (+ sentiment, mean imputation)')
    results['ridge_augmented'], series['ridge_augmented'] = run_ridge_augmented(df, tune_schedule)

    print('\n3. HGB baseline')
    results['hgb_baseline'],    series['hgb_baseline']    = run_hgb_baseline(df, tune_schedule)

    print('\n4. HGB augmented (+ sentiment, NaN-native)')
    results['hgb_augmented'],   series['hgb_augmented']   = run_hgb_augmented(df, tune_schedule)

    print('\n5. Stacked residual (Ridge trad -> HGB sentiment residual)')
    results['stacked_residual'], series['stacked_residual'] = run_stacked_residual(df, tune_schedule)

    print('\n6. Cascade pure-sent')
    results['cascade_puresent'], series['cascade_puresent'] = run_cascade_puresent(df, tune_schedule)

    print('\nFinal results:')
    res = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
    for c in ['ann_ret', 'ann_vol', 'max_dd']:
        res[c] = (res[c] * 100).round(2).astype(str) + '%'
    res['sharpe'] = res['sharpe'].round(4)
    print(res.to_string(index=False))

    out = pd.DataFrame([{'model': k, **v} for k, v in results.items()])
    out.to_parquet(DATA_DIR / 'backtest_final.parquet', index=False)
    out.to_csv(DATA_DIR    / 'backtest_final.csv',       index=False)

    series_df = pd.DataFrame(series)
    series_df.to_parquet(DATA_DIR / 'backtest_series.parquet')
    print('\nSaved backtest_final.parquet / .csv / backtest_series.parquet')
