# SEC EDGAR 8-K pipeline: fetch press-release headlines for SP500, merge with GDELT, rebuild model table.
import json, re, time, hashlib, warnings, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from tqdm import tqdm

warnings.filterwarnings('ignore', category=XMLParsedAsHTMLWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / 'data' / 'intermediate'
CACHE_DIR = ROOT / 'data' / 'cache' / 'edgar'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS      = {'User-Agent': 'cs229project academic-research@stanford.edu'}
WINDOW_START = pd.Timestamp('2017-01-01')
WINDOW_END   = pd.Timestamp.today().normalize()
WORKERS      = 6                                     
SEC_SLEEP    = 0.6                                                                
                                                                                   
def sec_get(url, **kwargs):
    """GET with SEC-safe rate limiting and retries."""
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=20, **kwargs)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return r
        except Exception:
            time.sleep(1)
    return None

print('Resolving CIKs')

CIK_CACHE = CACHE_DIR / 'cik_map.json'
ticker_to_cik: dict[str, str] = json.loads(CIK_CACHE.read_text()) if CIK_CACHE.exists() else {}

tickers   = pd.read_csv(DATA_DIR / 'sp500_tickers.csv')['ticker'].tolist()
missing   = [t for t in tickers if t not in ticker_to_cik]
print(f'Already resolved: {len(ticker_to_cik)}  |  Need: {len(missing)}')

def resolve_cik(ticker: str) -> str | None:
    r = sec_get('https://www.sec.gov/cgi-bin/browse-edgar',
                params={'CIK': ticker, 'action': 'getcompany', 'type': '10-K',
                        'owner': 'include', 'count': '1', 'output': 'atom'})
    if not r:
        return None
    m = re.search(r'CIK=(\d+)', r.url + r.text)
    return m.group(1).zfill(10) if m else None

for t in tqdm(missing, desc='Resolving CIKs'):
    cik = resolve_cik(t)
    if cik:
        ticker_to_cik[t] = cik
    time.sleep(SEC_SLEEP)

CIK_CACHE.write_text(json.dumps(ticker_to_cik))
print(f'CIKs resolved: {len(ticker_to_cik)}/{len(tickers)}')

print('\nFetching 8-K filing indexes')

def get_8k_filings(cik: str) -> list[dict]:
    cache_file = CACHE_DIR / f'filings_{cik}.json'
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    filings = []
    r = sec_get(f'https://data.sec.gov/submissions/CIK{cik}.json')
    if not r or r.status_code != 200:
        cache_file.write_text('[]')
        return []

    data = r.json()

    def extract(block):
        forms   = block.get('form', [])
        dates   = block.get('filingDate', [])
        accs    = block.get('accessionNumber', [])
        items_l = block.get('items', [''] * len(forms))
        for i, form in enumerate(forms):
            if form != '8-K':
                continue
            d = pd.Timestamp(dates[i])
            if d < WINDOW_START or d > WINDOW_END:
                continue
            filings.append({
                'date':      dates[i],
                'accession': accs[i],                                               
                'items':     items_l[i] if i < len(items_l) else '',
            })

    extract(data['filings']['recent'])
    for archive in data['filings'].get('files', []):
        if pd.Timestamp(archive.get('filingTo', '2000-01-01')) < WINDOW_START:
            continue
        time.sleep(SEC_SLEEP)
        r2 = sec_get(f'https://data.sec.gov/submissions/{archive["name"]}')
        if r2 and r2.status_code == 200:
            extract(r2.json())

    cache_file.write_text(json.dumps(filings))
    return filings

filing_index: dict[str, list] = {}
for ticker in tqdm(tickers, desc='8-K indexes'):
    cik = ticker_to_cik.get(ticker)
    if not cik:
        continue
    filings = get_8k_filings(cik)
    if filings:
        filing_index[ticker] = filings
    time.sleep(SEC_SLEEP)

total = sum(len(v) for v in filing_index.values())
print(f'Tickers with 8-Ks: {len(filing_index)}/{len(tickers)}  |  Total filings: {total}')

print('\nFetching press-release headlines')

def acc_nodash(acc: str) -> str:
    return acc.replace('-', '')

def get_exhibit_url(cik_int: int, acc: str) -> str | None:
    """Parse the filing index page to find exhibit 99.1 URL."""
    nodash  = acc_nodash(acc)
    idx_url = (f'https://www.sec.gov/Archives/edgar/data/{cik_int}/'
               f'{nodash}/{acc}-index.htm')
    cache_f = CACHE_DIR / f'idx_{nodash}.json'
    if cache_f.exists():
        return json.loads(cache_f.read_text()).get('url')

    r = sec_get(idx_url)
    if not r or r.status_code != 200:
        cache_f.write_text(json.dumps({'url': None}))
        return None

    soup = BeautifulSoup(r.text, 'lxml')
    exhibit_url = None
    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
                                          
        if re.search(r'ex[-_]?99[-_]?1|exhibit[-_]?99[-_]?1|ex991', href):
            full = ('https://www.sec.gov' + a['href']
                    if a['href'].startswith('/') else a['href'])
            exhibit_url = full
            break

    if not exhibit_url:
        for a in soup.find_all('a', href=True):
            href = a['href']
            if (href.endswith('.htm') and 'Archives' in href
                    and 'xsd' not in href and '.xml' not in href
                    and 'index' not in href.lower()):
                                              
                if not href.startswith('/ix?'):
                    exhibit_url = ('https://www.sec.gov' + href
                                   if href.startswith('/') else href)
                    break

    cache_f.write_text(json.dumps({'url': exhibit_url}))
    return exhibit_url

def extract_headline(html: str) -> str | None:
    """Pull the first meaningful headline from press-release HTML."""
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'table']): tag.decompose()

    for tag_name in ['h1', 'h2', 'h3']:
        for el in soup.find_all(tag_name):
            txt = el.get_text(' ', strip=True)
            if len(txt) > 20:
                return txt[:512]

    for p in soup.find_all('p'):
        txt = p.get_text(' ', strip=True)
        if len(txt) > 40:
            return txt[:512]

    return None

def fetch_headline(ticker: str, cik: str, filing: dict) -> dict | None:
    cik_int  = int(cik)
    acc      = filing['accession']
    nodash   = acc_nodash(acc)
    doc_date = pd.Timestamp(filing['date'])

    hl_cache = CACHE_DIR / f'hl_{nodash}.txt'
    if hl_cache.exists():
        txt = hl_cache.read_text(encoding='utf-8', errors='ignore').strip()
        if not txt:
            return None
        return {'ticker': ticker, 'date': doc_date,
                'title': txt, 'domain': 'sec.gov', 'url': ''}

    time.sleep(SEC_SLEEP)
    exhibit_url = get_exhibit_url(cik_int, acc)
    if not exhibit_url:
        hl_cache.write_text('', encoding='utf-8')
        return None

    time.sleep(SEC_SLEEP)
    r = sec_get(exhibit_url)
    if not r or r.status_code != 200:
        hl_cache.write_text('', encoding='utf-8')
        return None

    headline = extract_headline(r.text)
    if not headline:
        hl_cache.write_text('', encoding='utf-8')
        return None

    hl_cache.write_text(headline, encoding='utf-8')
    return {'ticker': ticker, 'date': doc_date,
            'title': headline, 'domain': 'sec.gov',
            'url': exhibit_url}

SKIP_ITEMS = {'5.02', '5.03', '9.01'}
tasks = []
for ticker, filings in filing_index.items():
    cik = ticker_to_cik.get(ticker)
    if not cik:
        continue
    for f in filings:
        item_set = set(re.split(r'[,\s]+', f.get('items', ''))) - {''}
                                                                      
        if not item_set or not item_set.issubset(SKIP_ITEMS):
            tasks.append((ticker, cik, f))

print(f'Filings to process (after filtering low-signal items): {len(tasks)}')

article_rows = []
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    futures = {pool.submit(fetch_headline, t, c, f): (t, f['date'])
               for t, c, f in tasks}
    for fut in tqdm(as_completed(futures), total=len(futures),
                    desc='Fetching headlines'):
        row = fut.result()
        if row:
            article_rows.append(row)

edgar_df = pd.DataFrame(article_rows)
if not edgar_df.empty:
    edgar_df['date'] = pd.to_datetime(edgar_df['date'], utc=True)

edgar_df.to_parquet(DATA_DIR / 'edgar_articles_raw.parquet', index=False)
covered = edgar_df['ticker'].nunique() if not edgar_df.empty else 0
print(f'Saved edgar_articles_raw.parquet')
print(f'  Articles : {len(edgar_df):,}')
print(f'  Tickers  : {covered}')

print('\nMerging GDELT + EDGAR')

gdelt = pd.read_parquet(DATA_DIR / 'gdelt_articles_raw.parquet')
gdelt['date'] = pd.to_datetime(gdelt['date'], utc=True)

combined = (pd.concat([gdelt, edgar_df], ignore_index=True)
              .drop_duplicates(subset=['ticker', 'date', 'title'])
              .sort_values(['ticker', 'date']))

backup = DATA_DIR / 'gdelt_articles_gdelt_only.parquet'
if not backup.exists():
    gdelt.to_parquet(backup, index=False)
    print(f'  GDELT-only backup saved.')

combined.to_parquet(DATA_DIR / 'gdelt_articles_raw.parquet', index=False)
print(f'  GDELT:    {len(gdelt):>8,}')
print(f'  EDGAR:    {len(edgar_df):>8,}')
print(f'  Combined: {len(combined):>8,}  ({combined["ticker"].nunique()} tickers)')

print('\nFinBERT scoring')

from transformers import BertTokenizer, BertForSequenceClassification
import torch

MODEL_NAME = 'yiyanghkust/finbert-tone'
tokenizer  = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
bert_model.eval()

id2label = bert_model.config.id2label
assert id2label[1].lower() == 'positive', f'Unexpected label order: {id2label}'

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

articles = combined.copy()
titles         = articles['title'].fillna('').astype(str)
unique_titles  = titles.drop_duplicates().tolist()
score_map      = {}

scored_path = DATA_DIR / 'gdelt_articles_scored.parquet'
if scored_path.exists():
    old_scored = pd.read_parquet(scored_path)
    if 'title' in old_scored.columns and 'finbert_score' in old_scored.columns:
        score_map = old_scored.dropna(subset=['title', 'finbert_score'])\
                               .set_index('title')['finbert_score'].to_dict()
    print(f'  Pre-loaded {len(score_map):,} existing FinBERT scores')

new_titles = [t for t in unique_titles if t not in score_map]
print(f'  New titles to score: {len(new_titles):,}')

batch = 64
for i in tqdm(range(0, len(new_titles), batch), desc='FinBERT scoring'):
    chunk  = new_titles[i:i + batch]
    scores = finbert_score(chunk, batch_size=batch)
    for t, s in zip(chunk, scores):
        score_map[t] = float(s)

articles['finbert_score'] = titles.map(score_map).astype(float)
articles.to_parquet(DATA_DIR / 'gdelt_articles_scored.parquet', index=False)

daily = (articles.groupby(['ticker', 'date'])['finbert_score']
         .agg(['mean', 'count'])
         .rename(columns={'mean': 'finbert_daily', 'count': 'news_count'}))
daily.index.set_names(['ticker', 'date'], inplace=True)

finbert_wide = daily['finbert_daily'].unstack('ticker')
count_wide   = daily['news_count'].unstack('ticker').fillna(0)

daily.to_parquet(DATA_DIR / 'news_daily_long.parquet')
finbert_wide.to_parquet(DATA_DIR / 'finbert_daily_wide.parquet')
count_wide.to_parquet(DATA_DIR / 'news_count_wide.parquet')
print(f'  FinBERT done. Tickers with scores: {finbert_wide.shape[1]}')

print('\nFeature merge')

factors      = pd.read_parquet(DATA_DIR / 'factors_traditional.parquet')
target       = pd.read_parquet(DATA_DIR / 'target_next_ret.parquet')
tone_wide    = pd.read_parquet(DATA_DIR / 'gdelt_tone_wide.parquet')
finbert_wide = pd.read_parquet(DATA_DIR / 'finbert_daily_wide.parquet')
count_wide   = pd.read_parquet(DATA_DIR / 'news_count_wide.parquet')

def to_naive(idx):
    return pd.to_datetime(idx, utc=True).tz_convert(None)

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

null_pct = (model_table[['finbert_sent_7d', 'gdelt_tone_7d']].isnull().mean() * 100).round(1)
n_tickers = model_table[model_table['finbert_sent_7d'].notna()].index.get_level_values('ticker').nunique()
print(f'Model table: {model_table.shape}')
print(f'Null rates:  {null_pct.to_dict()}')
print(f'Tickers with finbert: {n_tickers} / {model_table.index.get_level_values("ticker").nunique()}')
print('\nEDGAR pipeline complete.')
