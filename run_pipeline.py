"""
Full pipeline runner: GDELT collection → FinBERT scoring → feature merge.
Run this to refresh all news features for all 503 tickers.
"""
import json, hashlib, time, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'intermediate'
CACHE_DIR = ROOT / 'data' / 'cache' / 'gdelt'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print('\n=== STAGE 2: GDELT collection ===')

tickers = pd.read_csv(DATA_DIR / 'sp500_tickers.csv')['ticker'].tolist()
close   = pd.read_parquet(DATA_DIR / 'close_prices.parquet')
fetch_start = pd.to_datetime(close.index.min())
fetch_end   = pd.to_datetime(close.index.max())
gdelt_start = max(pd.Timestamp('2017-02-01'), fetch_start)

with open(DATA_DIR / 'ticker_to_name.json') as f:
    ticker_names = json.load(f)

GDELT_BASE = 'https://api.gdeltproject.org/api/v2/doc/doc'

def clean_name(name):
    if not name:
        return None
    for suf in [', Inc.', ' Inc.', ' Inc', ', Corp.', ' Corp.', ' Corp',
                ' Corporation', ' Company', ' Co.', ', Ltd.', ' Ltd.',
                ' Ltd', ' PLC', ' plc', ' N.V.', ' S.A.', ' AG', ' SE',
                ' Holdings', ' Group', ' Enterprises', ', L.P.']:
        name = name.replace(suf, '')
    return name.strip()

def cache_path(mode, params):
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()
    return CACHE_DIR / f'{mode}_{key}.json'

def query_gdelt(query, mode, start_dt, end_dt, maxrecords=250):
    params = {'query': query, 'mode': mode, 'format': 'json',
              'sourcelang': 'english', 'startdatetime': start_dt, 'enddatetime': end_dt}
    if mode == 'artlist':
        params['maxrecords'] = maxrecords
    cp = cache_path(mode, params)
    if cp.exists():
        return json.loads(cp.read_text())
    try:
        r = requests.get(GDELT_BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        cp.write_text(json.dumps(data))
        return data
    except Exception:
        return None

def parse_timeline(data):
    if not data or 'timeline' not in data:
        return pd.Series(dtype=float)
    rows = []
    for block in data['timeline']:
        series = block.get('data', block.get('series', []))
        if isinstance(series, str):
            for line in series.strip().split('\n'):
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    try:
                        rows.append((pd.to_datetime(parts[0].strip()), float(parts[1])))
                    except Exception:
                        pass
        elif isinstance(series, list):
            for pt in series:
                try:
                    rows.append((pd.to_datetime(pt.get('date', pt.get('bin', ''))),
                                 float(pt.get('value', pt.get('count', 0)))))
                except Exception:
                    pass
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows, columns=['date', 'value']).set_index('date')['value']
    return s.groupby(level=0).mean().sort_index()

def parse_artlist(data):
    if not data or 'articles' not in data:
        return []
    out = []
    for a in data['articles']:
        title = a.get('title', '').strip()
        seen  = a.get('seendate', '')
        if title and seen:
            try:
                out.append({'date': pd.to_datetime(seen).normalize(),
                            'title': title, 'domain': a.get('domain', ''),
                            'url': a.get('url', '')})
            except Exception:
                pass
    return out

start_fmt   = gdelt_start.strftime('%Y%m%d%H%M%S')
end_fmt     = fetch_end.strftime('%Y%m%d%H%M%S')
yearly_starts = pd.date_range(gdelt_start, fetch_end, freq='YS')

def fetch_ticker(t):
    name  = ticker_names.get(t, t)
    query = f'"{name}"'
    tone_data   = query_gdelt(query, 'timelinetone', start_fmt, end_fmt)
    tone_series = parse_timeline(tone_data)
    arts = []
    for ys in yearly_starts:
        ye  = min(ys + pd.DateOffset(years=1) - pd.Timedelta(days=1), fetch_end)
        raw = query_gdelt(query, 'artlist',
                          ys.strftime('%Y%m%d%H%M%S'), ye.strftime('%Y%m%d%H%M%S'),
                          maxrecords=250)
        for a in parse_artlist(raw):
            arts.append({'ticker': t, 'date': a['date'], 'title': a['title'],
                         'domain': a['domain'], 'url': a['url']})
    return t, tone_series, arts

tone_dict    = {}
article_rows = []
WORKERS      = 8

with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(fetch_ticker, t): t for t in tickers}
    for fut in tqdm(as_completed(futures), total=len(futures), desc='GDELT collection'):
        t, tone_series, arts = fut.result()
        if not tone_series.empty:
            tone_dict[t] = tone_series
        article_rows.extend(arts)

tone_wide = pd.DataFrame(tone_dict)
tone_wide.index = pd.to_datetime(tone_wide.index)
articles_raw = pd.DataFrame(article_rows)
if not articles_raw.empty:
    articles_raw['date'] = pd.to_datetime(articles_raw['date'])

tone_wide.to_parquet(DATA_DIR / 'gdelt_tone_wide.parquet')
articles_raw.to_parquet(DATA_DIR / 'gdelt_articles_raw.parquet')
print(f'Tone tickers: {tone_wide.shape[1]}  Articles: {len(articles_raw)}')

print('\n=== STAGE 3: FinBERT scoring ===')

if articles_raw.empty:
    print('No articles, skipping FinBERT.')
else:
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    MODEL_NAME = 'yiyanghkust/finbert-tone'
    tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    bert_model.eval()

    def finbert_score(texts, batch_size=64):
        label_weight = {0: 0.0, 1: 1.0, 2: -1.0}
        out = np.empty(len(texts), dtype=np.float32)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors='pt')
            with torch.no_grad():
                logits = bert_model(**enc).logits
            probs = torch.softmax(logits, dim=1).numpy()
            for j, p in enumerate(probs):
                out[i + j] = sum(label_weight[k] * p[k] for k in label_weight)
        return out

    titles        = articles_raw['title'].fillna('').astype(str)
    unique_titles = titles.drop_duplicates().tolist()
    score_map     = {}
    batch         = 64
    for i in tqdm(range(0, len(unique_titles), batch), desc='FinBERT scoring'):
        chunk  = unique_titles[i:i + batch]
        scores = finbert_score(chunk, batch_size=batch)
        for t, s in zip(chunk, scores):
            score_map[t] = float(s)

    articles_raw['finbert_score'] = titles.map(score_map).astype(float)
    articles_raw.to_parquet(DATA_DIR / 'gdelt_articles_scored.parquet')

    daily = (articles_raw.groupby(['ticker', 'date'])['finbert_score']
             .agg(['mean', 'count'])
             .rename(columns={'mean': 'finbert_daily', 'count': 'news_count'}))
    daily.index.set_names(['ticker', 'date'], inplace=True)

    finbert_wide = daily['finbert_daily'].unstack('ticker')
    count_wide   = daily['news_count'].unstack('ticker').fillna(0)

    daily.to_parquet(DATA_DIR / 'news_daily_long.parquet')
    finbert_wide.to_parquet(DATA_DIR / 'finbert_daily_wide.parquet')
    count_wide.to_parquet(DATA_DIR / 'news_count_wide.parquet')
    print(f'FinBERT done. Tickers with scores: {finbert_wide.shape[1]}')

print('\n=== STAGE 4: Feature merge ===')

factors      = pd.read_parquet(DATA_DIR / 'factors_traditional.parquet')
target       = pd.read_parquet(DATA_DIR / 'target_next_ret.parquet')
tone_wide    = pd.read_parquet(DATA_DIR / 'gdelt_tone_wide.parquet')
finbert_wide = pd.read_parquet(DATA_DIR / 'finbert_daily_wide.parquet')
count_wide   = pd.read_parquet(DATA_DIR / 'news_count_wide.parquet')

def to_naive(idx):
    idx = pd.to_datetime(idx, utc=True)
    return idx.tz_convert(None)

tone_wide.index    = to_naive(tone_wide.index)
finbert_wide.index = to_naive(finbert_wide.index)
count_wide.index   = to_naive(count_wide.index)

_d0 = min(finbert_wide.index.min(), tone_wide.index.min())
_d1 = max(finbert_wide.index.max(), tone_wide.index.max())
_cal = pd.date_range(_d0, _d1, freq='D')

finbert_wide = finbert_wide.reindex(_cal)
tone_wide    = tone_wide.reindex(_cal)
count_wide   = count_wide.reindex(_cal).fillna(0)

def rolling_mean(df, w): return df.rolling(f'{w}D', min_periods=1).mean()
def rolling_sum(df, w):  return df.rolling(f'{w}D', min_periods=1).sum()

def wide_to_long(df, col):
    s = df.stack()
    s.index.set_names(['date', 'ticker'], inplace=True)
    return s.to_frame(col)

fb7  = rolling_mean(finbert_wide, 7).shift(1).ffill(limit=30)
fb30 = rolling_mean(finbert_wide, 30).shift(1).ffill(limit=30)
gt7  = rolling_mean(tone_wide, 7).shift(1).ffill(limit=30)
gt30 = rolling_mean(tone_wide, 30).shift(1).ffill(limit=30)
nv7  = rolling_sum(count_wide, 7).shift(1).ffill(limit=30)
nv30 = rolling_sum(count_wide, 30).shift(1).ffill(limit=30)
smom = fb7 - fb30

sent_feats = (
    wide_to_long(fb7,  'finbert_sent_7d')
    .join(wide_to_long(fb30,  'finbert_sent_30d'),  how='outer')
    .join(wide_to_long(smom,  'sent_momentum'),      how='outer')
    .join(wide_to_long(gt7,   'gdelt_tone_7d'),      how='outer')
    .join(wide_to_long(gt30,  'gdelt_tone_30d'),     how='outer')
    .join(wide_to_long(nv7,   'news_volume_7d'),     how='outer')
    .join(wide_to_long(nv30,  'news_volume_30d'),    how='outer')
).sort_index()

model_table = factors.join(sent_feats, how='left').join(target[['ret_t1']], how='left')
model_table = model_table.sort_index()

model_table.to_parquet(DATA_DIR / 'model_table.parquet')
sent_feats.to_parquet(DATA_DIR / 'sentiment_features_long.parquet')

null_pct = (model_table[['finbert_sent_7d','gdelt_tone_7d']].isnull().mean() * 100).round(1)
n_tickers_sent = model_table[model_table['finbert_sent_7d'].notna()].index.get_level_values('ticker').nunique()
print(f'Model table: {model_table.shape}')
print(f'Null %: {null_pct.to_dict()}')
print(f'Tickers with finbert coverage: {n_tickers_sent} / {model_table.index.get_level_values("ticker").nunique()}')
print('\nPipeline complete.')
