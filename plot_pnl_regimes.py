# PnL and regime analysis for all 6 models; produces paper/figures/pnl_regimes.png/pdf.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT   = Path(__file__).resolve().parent
FIGDIR = ROOT / 'paper' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'ridge_baseline':   '#2c7bb6',
    'ridge_augmented':  '#74add1',
    'hgb_baseline':     '#d73027',
    'hgb_augmented':    '#f46d43',
    'stacked_residual': '#9970ab',
    'cascade_puresent': '#1a9850',
}
LABELS = {
    'ridge_baseline':   'Ridge Baseline',
    'ridge_augmented':  'Ridge Augmented',
    'hgb_baseline':     'HGB Baseline',
    'hgb_augmented':    'HGB Augmented',
    'stacked_residual': 'Stacked Residual',
    'cascade_puresent': 'Cascade (Ours)',
}
ORDER = ['hgb_baseline', 'hgb_augmented', 'stacked_residual',
         'ridge_augmented', 'ridge_baseline', 'cascade_puresent']

ret = pd.read_parquet(ROOT / 'data' / 'intermediate' / 'backtest_series.parquet')
cum = (1 + ret.fillna(0)).cumprod()
dd  = cum / cum.cummax() - 1

roll_sharpe = {}
for col in ret.columns:
    r   = ret[col].dropna()
    mu  = r.rolling(63, min_periods=30).mean() * 252
    sig = r.rolling(63, min_periods=30).std(ddof=1) * np.sqrt(252)
    roll_sharpe[col] = (mu / sig).reindex(ret.index)
roll_sharpe = pd.DataFrame(roll_sharpe)

REGIMES = [
    ('#fff3cd', 'COVID Recovery',    '2021-01-01', '2021-12-31'),
    ('#fce4e4', 'Rate Hike Cycle',   '2022-01-01', '2023-07-31'),
    ('#e8f5e9', 'AI Bull Market',    '2023-08-01', '2024-12-31'),
    ('#e3f2fd', '2025+ Uncertainty', '2025-01-01', '2026-03-05'),
]

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 10,
    'axes.labelsize'   : 10,
    'axes.titlesize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 9,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

fig = plt.figure(figsize=(14, 17))
gs  = fig.add_gridspec(
    4, 2,
    height_ratios=[2.6, 2.0, 2.0, 2.2],
    hspace=0.42, wspace=0.30,
    top=0.88, bottom=0.05, left=0.07, right=0.97,
)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd_l = fig.add_subplot(gs[1, 0])
ax_dd_r = fig.add_subplot(gs[1, 1])
ax_sr_l = fig.add_subplot(gs[2, 0])
ax_sr_r = fig.add_subplot(gs[2, 1])
ax_yr   = fig.add_subplot(gs[3, :])

def shade_regimes(ax):
    for color, _, start, end in REGIMES:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=color, alpha=0.45, zorder=0)

def add_regime_labels(ax, ypos_frac=0.97):
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * ypos_frac
    for _, label, start, end in REGIMES:
        xmid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(xmid, y, label, ha='center', va='top',
                fontsize=8, color='#444', style='italic')

def fmt_xaxis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

shade_regimes(ax_cum)
for m in ORDER:
    lw = 2.5 if m == 'cascade_puresent' else 1.3
    ls = '-'  if m in ('cascade_puresent', 'ridge_baseline') else '--'
    ax_cum.plot(cum.index, cum[m], color=COLORS[m], lw=lw, ls=ls,
                label=LABELS[m], zorder=3 if m == 'cascade_puresent' else 2)

ax_cum.axhline(1.0, color='black', lw=0.7, alpha=0.35)
ax_cum.set_ylabel('Cumulative Return (1 = start)')
ax_cum.set_title('Cumulative PnL — All 6 Models (2021–2026)', fontweight='bold')
ax_cum.yaxis.grid(True, alpha=0.3, zorder=0)
fmt_xaxis(ax_cum)
ax_cum.set_xlim(cum.index[0], cum.index[-1])
add_regime_labels(ax_cum, 0.97)

handles, labels_list = ax_cum.get_legend_handles_labels()
fig.legend(
    handles, labels_list,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.975),                                           
    ncol=3,
    frameon=True, framealpha=0.97,
    fontsize=10.5,
    columnspacing=1.8, handlelength=2.5, handletextpad=0.7,
    edgecolor='#ccc',
)

linear_models  = ['ridge_baseline', 'ridge_augmented', 'stacked_residual', 'cascade_puresent']
boosted_models = ['hgb_baseline', 'hgb_augmented']

for ax, models, title in [
    (ax_dd_l, linear_models,  'Drawdown — Ridge / Stacked / Cascade'),
    (ax_dd_r, boosted_models, 'Drawdown — HGB Models'),
]:
    shade_regimes(ax)
    ax.axhline(0, color='black', lw=0.7, alpha=0.35)
    for m in models:
        lw = 2.0 if m == 'cascade_puresent' else 1.1
        ax.fill_between(dd.index, dd[m] * 100, 0, alpha=0.15, color=COLORS[m])
        ax.plot(dd.index, dd[m] * 100, color=COLORS[m], lw=lw)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3)
    fmt_xaxis(ax)
    ax.set_xlim(cum.index[0], cum.index[-1])

for ax, models, title in [
    (ax_sr_l, linear_models,  'Rolling Sharpe (63d) — Ridge / Stacked / Cascade'),
    (ax_sr_r, boosted_models, 'Rolling Sharpe (63d) — HGB Models'),
]:
    shade_regimes(ax)
    ax.axhline(0, color='black', lw=0.8, alpha=0.4, ls='--')
    for m in models:
        lw = 2.0 if m == 'cascade_puresent' else 1.0
        ax.plot(roll_sharpe.index, roll_sharpe[m],
                color=COLORS[m], lw=lw, alpha=0.85)
    ax.set_ylabel('Rolling Sharpe')
    ax.set_title(title, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.3)
    fmt_xaxis(ax)
    ax.set_xlim(cum.index[0], cum.index[-1])
    ax.set_ylim(-4, 8)

years   = sorted(ret.index.year.unique())
width   = 0.13
offsets = np.linspace(-(len(ORDER)-1)/2*width, (len(ORDER)-1)/2*width, len(ORDER))
x       = np.arange(len(years))

for i, m in enumerate(ORDER):
    ann = [
        ret[m][ret.index.year == yr].dropna().mean() * 252 * 100
        if len(ret[m][ret.index.year == yr].dropna()) > 20 else np.nan
        for yr in years
    ]
    ax_yr.bar(x + offsets[i], ann, width=width * 0.88,
              color=COLORS[m], edgecolor='white', linewidth=0.4, zorder=3)

ax_yr.axhline(0, color='black', lw=0.8, alpha=0.5)
ax_yr.set_xticks(x)
ax_yr.set_xticklabels([str(y) for y in years])
ax_yr.set_ylabel('Annualized Return (%)')
ax_yr.set_title('Annual Return by Model and Year', fontweight='bold')
ax_yr.yaxis.grid(True, alpha=0.35, zorder=0)
ax_yr.set_axisbelow(True)
for i, yr in enumerate(years):
    if i % 2 == 0:
        ax_yr.axvspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

print("\nPer-regime Sharpe: Cascade vs Ridge Baseline")
print(f"{'Regime':<22} {'Cascade':>10} {'Ridge BL':>10} {'Delta':>8}")
for _, label, start, end in REGIMES:
    mask = (ret.index >= start) & (ret.index <= end)
    def _sr(m):
        r = ret[m][mask].dropna()
        return (r.mean()*252) / (r.std(ddof=1)*np.sqrt(252)) if r.std() > 0 else np.nan
    sr_cas, sr_rdg = _sr('cascade_puresent'), _sr('ridge_baseline')
    print(f"{label:<22} {sr_cas:>10.3f} {sr_rdg:>10.3f} {sr_cas-sr_rdg:>+8.3f}")

out_png = FIGDIR / 'pnl_regimes.png'
out_pdf = FIGDIR / 'pnl_regimes.pdf'
fig.savefig(out_png, bbox_inches='tight', dpi=150)
fig.savefig(out_pdf, bbox_inches='tight')
plt.close()
print(f"\nSaved {out_png}")
