from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'intermediate'

df = pd.read_parquet(DATA_DIR / 'model_table.parquet').sort_index()
df = df.dropna(subset=['ret_t1'])

df = df[df.index.get_level_values('date') >= '2017-01-01']
print(f'Panel: {len(df):,} rows | {df.index.get_level_values("date").min().date()} -> {df.index.get_level_values("date").max().date()}')

TRAD_FEATS = ['momentum', 'value_composite', 'quality_composite']
SENT_FEATS = ['finbert_sent_7d', 'finbert_sent_30d', 'sent_momentum',
              'gdelt_tone_7d', 'gdelt_tone_30d', 'news_volume_7d', 'news_volume_30d']
ALL_FEATS  = TRAD_FEATS + SENT_FEATS

MIN_TRAIN_ROWS = 250_000                     
ALPHA_GRID     = [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0]

def _clip_and_prepare(X, q1, q9, fill_na, imputer=None):
    X = X.clip(lower=q1, upper=q9, axis=1)
    if fill_na:
        if imputer is None:
            imputer = SimpleImputer(strategy='mean')
            arr = imputer.fit_transform(X.values)
        else:
            arr = imputer.transform(X.values)
        arr = np.nan_to_num(arr)
        return arr, imputer, X.index
    else:
        X = X.dropna()
        return X.values, None, X.index

def prescan_best_alpha(panel, feature_cols, fill_na, n_scans=5, val_days=252, label='prescan'):
    dates     = panel.index.get_level_values('date').unique().sort_values()
    start_idx = len(dates) // 3
    end_idx   = len(dates) - val_days - 5
    if end_idx <= start_idx:
        return ALPHA_GRID[len(ALPHA_GRID) // 2]

    cutoff_indices = np.linspace(start_idx, end_idx, n_scans, dtype=int)
    all_scores = []

    for ci in cutoff_indices:
        window_end  = dates[min(ci + val_days, len(dates) - 1)]
        sub         = panel.loc[panel.index.get_level_values('date') <= window_end]
        sub_dates   = sub.index.get_level_values('date').unique().sort_values()
        if len(sub_dates) < val_days + 1:
            continue
        val_start   = sub_dates[-val_days]
        inner_train = sub.loc[sub.index.get_level_values('date') <  val_start].dropna(subset=['ret_t1'] + TRAD_FEATS)
        val_panel   = sub.loc[sub.index.get_level_values('date') >= val_start]
        if len(inner_train) < 5000 or len(val_panel) < 1000:
            continue

        X_tr = inner_train[feature_cols].copy()
        q1, q9 = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_tr_arr, imp, tr_idx = _clip_and_prepare(X_tr, q1, q9, fill_na)
        y_tr = inner_train.loc[tr_idx, 'ret_t1'].values
        X_vl_arr, _, vl_idx = _clip_and_prepare(val_panel[feature_cols].copy(), q1, q9, fill_na, imp)
        y_vl = val_panel.loc[vl_idx, 'ret_t1']
        if y_vl.empty:
            continue

        sc = StandardScaler()
        Xs_tr = sc.fit_transform(X_tr_arr)
        Xs_vl = sc.transform(X_vl_arr)

        scores = {}
        for a in ALPHA_GRID:
            m = Ridge(alpha=a); m.fit(Xs_tr, y_tr)
            pred = pd.Series(m.predict(Xs_vl), index=vl_idx)
            joined = pd.DataFrame({'pred': pred, 'ret': y_vl}).dropna()
            if joined.empty:
                scores[a] = np.nan; continue
            ic = joined.groupby(level='date').apply(
                lambda g: g['pred'].corr(g['ret'], method='spearman')
            ).dropna().mean()
            scores[a] = ic
        all_scores.append(pd.Series(scores))

    if not all_scores:
        return ALPHA_GRID[len(ALPHA_GRID) // 2]

    avg = pd.concat(all_scores, axis=1).mean(axis=1)
    best = float(avg.idxmax())
    print(f'  [{label}] avg IC: {avg.round(5).to_dict()}  => best alpha={best}')
    return best

def run_walkforward(panel, feature_cols, fill_na, label):
    alpha = prescan_best_alpha(panel, feature_cols, fill_na, label=label)
    print(f'  [{label}] fixed alpha={alpha}')

    dates       = panel.index.get_level_values('date').unique().sort_values()
    rebal_dates = dates[np.arange(0, len(dates), 5)]
    port        = pd.Series(index=dates, dtype=float)
    last_w      = pd.Series(dtype=float)

    for d in tqdm(rebal_dates, desc=label):
        train = panel.loc[panel.index.get_level_values('date') < d].dropna(subset=['ret_t1'] + TRAD_FEATS)
        if len(train) < MIN_TRAIN_ROWS:
            continue

        X = train[feature_cols].copy()
        q1, q9 = X.quantile(0.01), X.quantile(0.99)
        X_arr, imp, tr_idx = _clip_and_prepare(X, q1, q9, fill_na)
        y = train.loc[tr_idx, 'ret_t1'].values

        sc = StandardScaler()
        Xs = sc.fit_transform(X_arr)
        model = Ridge(alpha=alpha); model.fit(Xs, y)

        try:
            today = panel.loc[(d, slice(None)), feature_cols].copy()
        except KeyError:
            continue
        today_arr, _, today_idx = _clip_and_prepare(today, q1, q9, fill_na, imp)
        if len(today_arr) == 0:
            continue

        pred = pd.Series(model.predict(sc.transform(today_arr)), index=today_idx.get_level_values('ticker'))
        pred = pred.replace([np.inf, -np.inf], np.nan).dropna()

        longs  = pred.nlargest(50).index
        shorts = pred.nsmallest(50).index
        w = pd.Series(0.0, index=pred.index)
        w.loc[longs]  =  1 / max(len(longs),  1)
        w.loc[shorts] = -1 / max(len(shorts), 1)
        gross = w.abs().sum()
        if gross > 0: w /= gross

        turnover = (w - last_w.reindex(w.index, fill_value=0)).abs().sum()
        cost     = turnover * 10 / 10000
        last_w   = w

        i        = np.where(rebal_dates == d)[0][0] + 1
        next_cut = rebal_dates[i] if i < len(rebal_dates) else dates[-1]
        span     = dates[(dates >= d) & (dates < next_cut)]
        for dd in span:
            r = panel.loc[(dd, w.index), 'ret_t1'].fillna(0.0)
            port.loc[dd] = (w * r).sum()
        if len(span) > 0:
            port.loc[span[0]] -= cost

    perf = port.dropna()
    cum  = (1 + perf).cumprod()
    ann_ret = perf.mean() * 252
    ann_vol = perf.std(ddof=1) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd  = (cum / cum.cummax() - 1).min()
    return perf, cum, {'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 'max_dd': max_dd}

print('\n=== BASELINE (momentum/value/quality only) ===')
perf_base, cum_base, stats_base = run_walkforward(df, TRAD_FEATS, fill_na=False, label='baseline')

print('\n=== AUGMENTED (+ sentiment) ===')
perf_aug, cum_aug, stats_aug = run_walkforward(df, ALL_FEATS, fill_na=True, label='augmented')

print('\n========== RESULTS (2017+, 2yr burn-in) ==========')
res = pd.DataFrame([
    {'model': 'Ridge baseline',   **stats_base},
    {'model': 'Ridge augmented',  **stats_aug},
])
res['ann_ret'] = (res['ann_ret'] * 100).round(2).astype(str) + '%'
res['ann_vol'] = (res['ann_vol'] * 100).round(2).astype(str) + '%'
res['sharpe']  = res['sharpe'].round(4)
res['max_dd']  = (res['max_dd']  * 100).round(2).astype(str) + '%'
print(res.to_string(index=False))

out = pd.DataFrame([
    {'model': 'ridge_baseline',  **stats_base},
    {'model': 'ridge_augmented', **stats_aug},
])
out.to_parquet(DATA_DIR / 'backtest_summary_ridge_2017.parquet', index=False)
out.to_csv(DATA_DIR / 'backtest_summary_ridge_2017.csv', index=False)
print('\nSaved backtest_summary_ridge_2017.parquet / .csv')
