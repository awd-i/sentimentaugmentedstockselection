# Alpha decay and crowding analysis: IC decay curves, annual IC trend, portfolio return profile, rolling IC.
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

ROOT   = Path(__file__).resolve().parent
FIGDIR = ROOT / 'paper' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150,
})

prices = pd.read_parquet(ROOT / 'data' / 'intermediate' / 'close_prices.parquet')
panel  = pd.read_parquet(ROOT / 'data' / 'intermediate' / 'model_table.parquet').sort_index()
panel  = panel[panel.index.get_level_values('date') >= '2019-01-01']

horizons = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60]
print("Computing multi-horizon returns...")
fwd_rets = {}
for h in horizons:
    r = np.log(prices / prices.shift(h)).shift(-h)                              
    fwd_rets[h] = r.stack().rename(f'ret_{h}d')
    fwd_rets[h].index.names = ['date', 'ticker']

sig = panel[['momentum', 'finbert_sent_30d', 'finbert_sent_7d',
             'news_volume_30d']].copy()
                                                                          
sig = sig.reset_index()
sig['date'] = pd.to_datetime(sig['date'])
sig = sig.set_index(['date', 'ticker'])

print("Computing IC decay curves...")
signals_to_test = {
    'Momentum':        'momentum',
    'FinBERT 30d Sent':'finbert_sent_30d',
    'FinBERT 7d Sent': 'finbert_sent_7d',
}

ic_decay = {s: [] for s in signals_to_test}
ic_decay_std = {s: [] for s in signals_to_test}

for h in horizons:
    ret_h = fwd_rets[h].reindex(sig.index)
    for name, col in signals_to_test.items():
                                                        
        merged = pd.concat([sig[col], ret_h], axis=1).dropna()
        daily_ic = merged.groupby(level='date').apply(
            lambda g: spearmanr(g.iloc[:, 0], g.iloc[:, 1])[0]
            if len(g) > 10 else np.nan
        ).dropna()
        ic_decay[name].append(daily_ic.mean())
        ic_decay_std[name].append(daily_ic.std())
    print(f"  h={h} done")

print("Computing annual IC trends...")
years = range(2019, 2027)
annual_ic = {'Momentum': [], 'FinBERT 30d Sent': []}
ret_1d = fwd_rets[1].reindex(sig.index)

for yr in years:
    mask = sig.index.get_level_values('date').year == yr
    sub  = sig[mask]
    r1   = ret_1d[mask]
    for name, col in [('Momentum', 'momentum'),
                      ('FinBERT 30d Sent', 'finbert_sent_30d')]:
        merged = pd.concat([sub[col], r1], axis=1).dropna()
        if len(merged) < 1000:
            annual_ic[name].append(np.nan)
            continue
        daily_ic = merged.groupby(level='date').apply(
            lambda g: spearmanr(g.iloc[:, 0], g.iloc[:, 1])[0]
            if len(g) > 10 else np.nan
        ).dropna()
        annual_ic[name].append(daily_ic.mean())

print("Computing portfolio return profile...")
ret_series = pd.read_parquet(ROOT / 'data' / 'intermediate' / 'backtest_series.parquet')

cascade = ret_series['cascade_puresent'].dropna()
ridge   = ret_series['ridge_baseline'].dropna()

day_profile_cascade = []
day_profile_ridge   = []
REBAL = 10
dates = cascade.index
for start_i in range(0, len(dates) - REBAL, REBAL):
    window_c = cascade.iloc[start_i:start_i + REBAL].values
    window_r = ridge.iloc[start_i:start_i + REBAL].values
    if len(window_c) == REBAL:
        day_profile_cascade.append(window_c)
        day_profile_ridge.append(window_r)

day_profile_cascade = np.array(day_profile_cascade)
day_profile_ridge   = np.array(day_profile_ridge)
avg_cum_cascade = np.cumprod(1 + day_profile_cascade, axis=1).mean(axis=0) - 1
avg_cum_ridge   = np.cumprod(1 + day_profile_ridge,   axis=1).mean(axis=0) - 1
se_cascade = day_profile_cascade.std(axis=0) / np.sqrt(len(day_profile_cascade))
se_ridge   = day_profile_ridge.std(axis=0)   / np.sqrt(len(day_profile_ridge))

print("Computing rolling IC over time...")
merged_all = pd.concat([sig[['momentum', 'finbert_sent_30d']], ret_1d], axis=1).dropna(subset=['ret_1d'])
monthly_ic_mom  = []
monthly_ic_sent = []
monthly_dates   = []

dates_all = merged_all.index.get_level_values('date').unique().sort_values()
window    = 63                     

for i in range(window, len(dates_all)):
    win_dates = dates_all[i - window:i]
    sub = merged_all[merged_all.index.get_level_values('date').isin(win_dates)]
    for col, store in [('momentum', monthly_ic_mom),
                       ('finbert_sent_30d', monthly_ic_sent)]:
        ic = sub.groupby(level='date').apply(
            lambda g: spearmanr(g[col], g['ret_1d'])[0]
            if len(g) > 10 else np.nan
        ).dropna().mean()
        store.append(ic)
    monthly_dates.append(dates_all[i])

monthly_dates   = pd.DatetimeIndex(monthly_dates)
monthly_ic_mom  = pd.Series(monthly_ic_mom,  index=monthly_dates)
monthly_ic_sent = pd.Series(monthly_ic_sent, index=monthly_dates)

def half_life(ics, horizons):
    peak = max(abs(ic) for ic in ics if not np.isnan(ic))
    half = peak / 2
    for i, (h, ic) in enumerate(zip(horizons, ics)):
        if abs(ic) <= half:
            if i == 0:
                return h
                           
            h0, ic0 = horizons[i-1], abs(ics[i-1])
            frac = (ic0 - half) / (ic0 - abs(ic) + 1e-9)
            return h0 + frac * (h - h0)
    return horizons[-1]

hl_mom  = half_life(ic_decay['Momentum'],         horizons)
hl_sent = half_life(ic_decay['FinBERT 30d Sent'],  horizons)
print(f"\nHalf-life (days): Momentum={hl_mom:.1f}d | FinBERT 30d={hl_sent:.1f}d")

COLORS_SIG = {
    'Momentum':         '#2c7bb6',
    'FinBERT 30d Sent': '#d73027',
    'FinBERT 7d Sent':  '#f46d43',
}

fig = plt.figure(figsize=(13, 12))
gs  = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.32)

ax1 = fig.add_subplot(gs[0, :])                                
ax2 = fig.add_subplot(gs[1, 0])                    
ax3 = fig.add_subplot(gs[1, 1])                             
ax4 = fig.add_subplot(gs[2, :])                                      

for name, ics in ic_decay.items():
    ics_arr = np.array(ics)
    std_arr = np.array(ic_decay_std[name])
    ax1.plot(horizons, ics_arr * 100, 'o-', color=COLORS_SIG[name],
             lw=2, ms=5, label=f'{name}  (half-life ≈ {half_life(ics, horizons):.0f}d)')
    ax1.fill_between(horizons,
                     (ics_arr - std_arr / np.sqrt(300)) * 100,
                     (ics_arr + std_arr / np.sqrt(300)) * 100,
                     alpha=0.12, color=COLORS_SIG[name])

ax1.axhline(0, color='black', lw=0.8, alpha=0.4, ls='--')
ax1.set_xlabel('Forward Horizon (days)', fontsize=10)
ax1.set_ylabel('Mean Rank IC (×100)', fontsize=10)
ax1.set_title('Alpha Decay: Rank IC vs Forward Return Horizon\n'
              'Faster decay = signal arbitraged away quickly by large funds',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, frameon=False)
ax1.yaxis.grid(True, alpha=0.3)
ax1.set_xticks(horizons)

for name, hl, offset in [('Momentum', hl_mom, 0.008),
                           ('FinBERT 30d Sent', hl_sent, -0.010)]:
    ics = np.array(ic_decay[name])
    peak = ics[0]
    ax1.annotate(f'½-life\n≈{hl:.0f}d',
                 xy=(hl, peak * 50),
                 xytext=(hl + 4, (peak + offset) * 100),
                 fontsize=7.5, color=COLORS_SIG[name],
                 arrowprops=dict(arrowstyle='->', color=COLORS_SIG[name], lw=0.8))

x = np.arange(len(years))
w = 0.35
bars_m = ax2.bar(x - w/2, [v * 100 for v in annual_ic['Momentum']],
                 width=w, color='#2c7bb6', label='Momentum', alpha=0.85)
bars_s = ax2.bar(x + w/2, [v * 100 if v else 0
                             for v in annual_ic['FinBERT 30d Sent']],
                 width=w, color='#d73027', label='FinBERT 30d', alpha=0.85)

ax2.axhline(0, color='black', lw=0.8, alpha=0.4)
                         
valid_m = [(i, v) for i, v in enumerate(annual_ic['Momentum']) if not np.isnan(v)]
if len(valid_m) >= 3:
    xi, yi = zip(*valid_m)
    z = np.polyfit(xi, yi, 1)
    ax2.plot(x, np.polyval(z, x) * 100, '--', color='#2c7bb6', lw=1.5, alpha=0.6)

ax2.set_xticks(x)
ax2.set_xticklabels([str(y) for y in years], fontsize=8, rotation=30)
ax2.set_ylabel('Mean Rank IC (×100)', fontsize=9)
ax2.set_title('Annual IC: Is Alpha Being\nArbitraged Over Time?',
              fontsize=10, fontweight='bold')
ax2.legend(fontsize=8, frameon=False)
ax2.yaxis.grid(True, alpha=0.3)

slope_m = np.polyfit(range(len([v for v in annual_ic['Momentum'] if not np.isnan(v)])),
                     [v for v in annual_ic['Momentum'] if not np.isnan(v)], 1)[0]
direction = 'Declining ↓' if slope_m < 0 else 'Stable →'
ax2.text(0.97, 0.97, f'Momentum trend:\n{direction}',
         transform=ax2.transAxes, ha='right', va='top',
         fontsize=8, color='#2c7bb6',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

days = np.arange(1, REBAL + 1)
ax3.plot(days, avg_cum_cascade * 100, 'o-', color='#1a9850', lw=2, ms=4,
         label='Cascade')
ax3.fill_between(days,
                 (avg_cum_cascade - 2 * se_cascade) * 100,
                 (avg_cum_cascade + 2 * se_cascade) * 100,
                 alpha=0.15, color='#1a9850')
ax3.plot(days, avg_cum_ridge * 100, 's-', color='#2c7bb6', lw=2, ms=4,
         label='Ridge Baseline')
ax3.fill_between(days,
                 (avg_cum_ridge - 2 * se_ridge) * 100,
                 (avg_cum_ridge + 2 * se_ridge) * 100,
                 alpha=0.15, color='#2c7bb6')

ax3.axhline(0, color='black', lw=0.7, alpha=0.4, ls='--')
ax3.set_xlabel('Days After Rebalance', fontsize=9)
ax3.set_ylabel('Average Cumulative Return (%)', fontsize=9)
ax3.set_title('Portfolio Return Profile\n(avg within 10-day holding window)',
              fontsize=10, fontweight='bold')
ax3.legend(fontsize=8.5, frameon=False)
ax3.yaxis.grid(True, alpha=0.3)
ax3.set_xticks(days)

ax4.axhline(0, color='black', lw=0.8, alpha=0.4, ls='--')
ax4.plot(monthly_ic_mom.index, monthly_ic_mom.values * 100, color='#2c7bb6', lw=1.6,
         label='Momentum (63d rolling IC)', alpha=0.85)
ax4.plot(monthly_ic_sent.index, monthly_ic_sent.values * 100, color='#d73027', lw=1.6,
         label='FinBERT 30d (63d rolling IC)', alpha=0.85)

crowding_events = [
    ('2022-01', '2022-12', '#fff3cd', 'Rate Hike\nDisruption'),
    ('2023-06', '2024-06', '#e8f5e9', 'AI Rally\n(factor crowding)'),
]
for start, end, color, label in crowding_events:
    ax4.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                color=color, alpha=0.45, zorder=0)
    mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
    ymax = ax4.get_ylim()[1] if ax4.get_ylim()[1] != 0 else 0.5
    ax4.text(mid, 0.35, label, ha='center', va='bottom',
             fontsize=7, color='#555', style='italic', linespacing=1.2)

ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
ax4.tick_params(axis='x', labelsize=8, rotation=30)
ax4.set_ylabel('Rank IC ×100 (63d rolling)', fontsize=10)
ax4.set_xlabel('Date', fontsize=10)
ax4.set_title('Rolling Rank IC Over Time — Alpha Crowding Indicator\n'
              'Declining IC in recent years suggests institutional arbitrage',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=9, frameon=False)
ax4.yaxis.grid(True, alpha=0.3)
ax4.set_xlim(monthly_dates[0], monthly_dates[-1])

recent_mask = monthly_ic_mom.index >= '2023-01-01'
if recent_mask.sum() > 5:
    recent_mom  = monthly_ic_mom[recent_mask].mean() * 100
    recent_sent = monthly_ic_sent[recent_mask].mean() * 100
    ax4.text(0.99, 0.05,
             f'2023–2026 avg IC:\nMomentum: {recent_mom:.3f}\nFinBERT: {recent_sent:.3f}',
             transform=ax4.transAxes, ha='right', va='bottom',
             fontsize=8, color='#333',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

fig.suptitle('Alpha Decay & Crowding Analysis\nDo Hedge Funds Trade Away Our Signals?',
             fontsize=13, fontweight='bold', y=1.01)

fig.savefig(FIGDIR / 'alpha_decay.png', bbox_inches='tight', dpi=150)
fig.savefig(FIGDIR / 'alpha_decay.pdf', bbox_inches='tight')
plt.close()
print(f"\nSaved {FIGDIR / 'alpha_decay.png'}")

print("\nIC Decay Summary")
print(f"{'Horizon':>8} {'Momentum IC':>13} {'FinBERT30d IC':>15} {'FinBERT7d IC':>14}")
for i, h in enumerate(horizons):
    print(f"{h:>8}d  {ic_decay['Momentum'][i]*100:>11.4f}  "
          f"{ic_decay['FinBERT 30d Sent'][i]*100:>13.4f}  "
          f"{ic_decay['FinBERT 7d Sent'][i]*100:>12.4f}")

print(f"\nHalf-lives:")
for name in signals_to_test:
    hl = half_life(ic_decay[name], horizons)
    print(f"  {name:<22}: {hl:.1f} days")

print("\nAnnual IC (momentum)")
for yr, ic in zip(years, annual_ic['Momentum']):
    bar = '█' * int(abs(ic) * 5000) if not np.isnan(ic) else ''
    print(f"  {yr}: {ic*100:+.4f}  {bar}")
