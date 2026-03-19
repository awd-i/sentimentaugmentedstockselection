"""
Generate all paper figures.
Run after run_final_comparison.py (produces backtest_series.parquet).
Figures saved to paper/figures/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data' / 'intermediate'
FIG_DIR  = ROOT / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

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

_stats_path = DATA_DIR / 'backtest_final.parquet'
if _stats_path.exists():
    _df = pd.read_parquet(_stats_path).set_index('model')
    STATS = {
        m: {
            'ann_ret': float(_df.loc[m, 'ann_ret']) * 100,
            'vol':     float(_df.loc[m, 'ann_vol']) * 100,
            'sharpe':  float(_df.loc[m, 'sharpe']),
            'max_dd':  float(_df.loc[m, 'max_dd']) * 100,
        }
        for m in _df.index if m in COLORS
    }
else:
    STATS = {
        'ridge_baseline':   {'ann_ret': 8.92, 'vol': 6.80, 'sharpe': 1.311, 'max_dd':  -5.94},
        'ridge_augmented':  {'ann_ret': 8.15, 'vol': 6.47, 'sharpe': 1.259, 'max_dd':  -5.86},
        'hgb_baseline':     {'ann_ret': 4.80, 'vol': 7.43, 'sharpe': 0.647, 'max_dd': -16.91},
        'hgb_augmented':    {'ann_ret': 7.63, 'vol': 7.52, 'sharpe': 1.015, 'max_dd': -11.68},
        'stacked_residual': {'ann_ret': 7.69, 'vol': 6.24, 'sharpe': 1.231, 'max_dd':  -6.28},
        'cascade_puresent': {'ann_ret': 8.33, 'vol': 5.25, 'sharpe': 1.587, 'max_dd':  -5.59},
    }

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.labelsize'   : 11,
    'axes.titlesize'   : 12,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'legend.fontsize'  : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
})

def fig_pipeline():
                                                           
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis('off')

    def box(x, y, w, h, label, sublabel='', color='#dce8f5', fontsize=8.5):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
            boxstyle='round,pad=0.08', linewidth=1.2,
            edgecolor='#555', facecolor=color, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.13 if sublabel else 0),
                label, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', zorder=4)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22,
                    sublabel, ha='center', va='center', fontsize=7.5,
                    color='#444', zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#444', lw=1.4), zorder=5)

    TY = 2.6                                   
    box(0.1, TY, 1.7, 1.1, 'GDELT + EDGAR\nNews Articles', color='#fff3cd')
    box(2.2, TY, 1.6, 1.1, 'FinBERT', 'Sentiment Scoring', color='#fce4d6')
    box(4.2, TY, 1.9, 1.1, 'Sentiment\nFeatures', 'finbert 7/30/60d', color='#fce4d6')
    arrow(1.8,  3.15, 2.2,  3.15)
    arrow(3.8,  3.15, 4.2,  3.15)

    BY = 0.8                                      
    box(0.1,  BY, 1.7, 1.1, 'S&P 500 Price\n& Financial Data', color='#d4edda')
    box(2.2,  BY, 1.6, 1.1, 'Factor\nEngineering', 'Mom / Val / Qual', color='#d4edda')
    box(4.2,  BY, 1.9, 1.1, 'Traditional\nFeatures', 'momentum, value,\nquality', color='#d4edda')
    box(6.5,  BY, 2.0, 1.1, 'Stage 1: Ridge', 'Factor ranking\n→ Top/Bottom 150', color='#d4edda')
    box(8.9,  BY, 2.0, 1.1, 'Stage 2: Cascade', 'Re-rank by\nfinbert_30d', color='#fce4d6')
    box(11.2, BY, 1.7, 1.1, 'Long/Short\nPortfolio', 'Long 50 / Short 50\n10 bps TC', color='#c3e6cb')
    arrow(1.8,  1.35, 2.2,  1.35)
    arrow(3.8,  1.35, 4.2,  1.35)
    arrow(6.1,  1.35, 6.5,  1.35)
    arrow(8.5,  1.35, 8.9,  1.35)
    arrow(10.9, 1.35, 11.2, 1.35)

    RED = '#c0392b'
    ax.plot([5.15, 5.15, 9.9], [3.7, 4.0, 4.0], color=RED, lw=1.6, zorder=6, solid_capstyle='round')
    ax.annotate('', xy=(9.9, 1.9), xytext=(9.9, 4.0),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.6), zorder=6)

    ax.set_title('Cascade Pure-Sentiment Pipeline', fontweight='bold', pad=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_pipeline.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_pipeline.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig1_pipeline')

def fig_sharpe():
    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    sharpes = [STATS[m]['sharpe'] for m in ORDER]
    colors  = [COLORS[m]          for m in ORDER]
    xlabels = [LABELS[m]          for m in ORDER]

    bars = ax.bar(xlabels, sharpes, color=colors, edgecolor='white',
                  linewidth=0.8, width=0.6, zorder=3)

    ridge_sr = STATS['ridge_baseline']['sharpe']
    ax.axhline(ridge_sr, color=COLORS['ridge_baseline'], linestyle='--',
               linewidth=1.5, alpha=0.75, zorder=2)
                                                                           
    ax.text(0.08, ridge_sr + 0.04,
            f'Ridge Baseline  {ridge_sr:.3f}',
            ha='left', va='bottom', fontsize=9.5,
            color=COLORS['ridge_baseline'], style='italic')

    for bar, val in zip(bars, sharpes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10.5, fontweight='bold')

    bars[-1].set_edgecolor('#155724')
    bars[-1].set_linewidth(2.2)

    ax.set_ylabel('Sharpe Ratio (2021–2026)')
    ax.set_ylim(0, 2.10)
    ax.yaxis.grid(True, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title('Sharpe Ratio by Model Architecture', fontweight='bold', pad=10)
    ax.tick_params(axis='x', labelsize=10.5, rotation=15)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_sharpe.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_sharpe.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig2_sharpe')

def fig_risk_return():
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    label_xy = {
        'cascade_puresent': (3.9,  9.2),                                     
        'ridge_baseline':   (7.3,  9.7),                                       
        'ridge_augmented':  (7.5,  8.3),                     
        'stacked_residual': (3.9,  7.3),                               
        'hgb_baseline':     (8.9,  5.3),                                        
        'hgb_augmented':    (8.9,  4.0),                       
    }

    for m in STATS:
        s = STATS[m]
        ax.scatter(s['vol'], s['ann_ret'],
                   s=230, color=COLORS[m], edgecolors='white',
                   linewidths=1.0, zorder=5)
        lx, ly = label_xy[m]
        ax.annotate(
            LABELS[m],
            xy=(s['vol'], s['ann_ret']),
            xytext=(lx, ly),
            fontsize=10, color=COLORS[m], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85),
            arrowprops=dict(arrowstyle='-', color=COLORS[m], lw=1.0, alpha=0.7),
            zorder=6,
        )

    vols = np.linspace(3.5, 10, 300)
    for sr in [0.8, 1.1, 1.3, 1.587]:
        is_top = sr == 1.587
        color  = COLORS['cascade_puresent'] if is_top else 'gray'
        ls     = '-'  if is_top else '--'
        lw     = 1.8  if is_top else 0.9
        alpha  = 0.75 if is_top else 0.30
        ax.plot(vols, vols * sr, ls=ls, lw=lw, color=color, alpha=alpha, zorder=1)
                                                                         
        x_lab = 9.4
        y_lab = x_lab * sr
        if y_lab < 12.5:                                             
            ax.text(x_lab, y_lab, f'SR={sr}',
                    fontsize=9, color=color if is_top else '#666',
                    va='center', ha='left', alpha=0.95,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))

    ax.set_xlabel('Annualized Volatility (%)')
    ax.set_ylabel('Annualized Return (%)')
    ax.set_xlim(3.5, 10.5)
    ax.set_ylim(2.5, 12.5)
    ax.yaxis.grid(True, alpha=0.25, zorder=0)
    ax.xaxis.grid(True, alpha=0.25, zorder=0)
    ax.set_title('Risk–Return Frontier by Model', fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_risk_return.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_risk_return.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig3_risk_return')

def fig_sentiment():
    df = pd.read_parquet(DATA_DIR / 'model_table.parquet').sort_index()
    df = df.dropna(subset=['ret_t1'])
    df = df[df.index.get_level_values('date') >= '2019-01-01']

    cov = (df.groupby(level='date')['finbert_sent_30d']
             .apply(lambda x: x.notna().mean()) * 100)

    ic_sent = df.groupby(level='date').apply(
        lambda g: g['finbert_sent_30d'].corr(g['ret_t1'], method='spearman')
    ).dropna().rolling(60, min_periods=20).mean()

    ic_mom = df.groupby(level='date').apply(
        lambda g: g['momentum'].corr(g['ret_t1'], method='spearman')
    ).dropna().rolling(60, min_periods=20).mean()

    VAL_START  = pd.Timestamp('2020-01-01')
    VAL_END    = pd.Timestamp('2020-12-31')
    TEST_START = pd.Timestamp('2021-01-01')

    def shade(ax):
        ax.axvspan(VAL_START,  VAL_END,         alpha=0.12, color='orange', zorder=0)
        ax.axvspan(TEST_START, cov.index[-1],   alpha=0.08, color='red',    zorder=0)
        ax.axvline(VAL_START,  color='darkorange', lw=1.2, ls=':', alpha=0.8)
        ax.axvline(TEST_START, color='crimson',    lw=1.3, ls='--', alpha=0.8)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), sharex=True)

    ax1.fill_between(cov.index, cov.values, alpha=0.45, color='#2c7bb6')
    ax1.plot(cov.index, cov.values, color='#2c7bb6', lw=1.3)
    shade(ax1)
    ax1.set_ylabel('Sentiment Coverage (%)')
    ax1.set_ylim(0, 85)
    ax1.yaxis.grid(True, alpha=0.3)
                                                                  
    ax1.text(pd.Timestamp('2019-06-01'), 81, 'Train',
             fontsize=11, color='#333',      va='top', ha='center', fontweight='bold')
    ax1.text(pd.Timestamp('2020-07-01'), 81, 'Val',
             fontsize=11, color='darkorange', va='top', ha='center', fontweight='bold')
    ax1.text(pd.Timestamp('2023-06-01'), 81, 'Test',
             fontsize=11, color='crimson',    va='top', ha='center', fontweight='bold')

    ax2.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax2.plot(ic_sent.index, ic_sent.values, color='#d73027', lw=1.8,
             label='Sentiment IC (60-day rolling)')
    ax2.plot(ic_mom.index,  ic_mom.values,  color='#1a9850', lw=1.8,
             label='Momentum IC (60-day rolling)')
    shade(ax2)
    ax2.set_ylabel('Rank IC (Spearman)')
    ax2.set_xlabel('Date')
    ax2.yaxis.grid(True, alpha=0.3)

    ax2.legend(frameon=False, ncol=2,
               bbox_to_anchor=(0.5, -0.22), loc='upper center', borderaxespad=0)

    ax1.set_title('FinBERT Sentiment Coverage and Signal IC', fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_sentiment.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_sentiment.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig4_sentiment')

def fig_cumulative():
    series_path = DATA_DIR / 'backtest_series.parquet'
    if not series_path.exists():
        print('backtest_series.parquet not found — skipping fig5.')
        return

    series = pd.read_parquet(series_path)
    series = series[series.index >= '2021-01-01'].dropna(how='all')
    cum    = (1 + series.fillna(0)).cumprod()

    fig, ax = plt.subplots(figsize=(9, 4.2))

    for m in ORDER:
        if m not in cum.columns:
            continue
        is_cascade = m == 'cascade_puresent'
        ax.plot(cum.index, cum[m],
                color=COLORS[m],
                lw=2.5 if is_cascade else 1.3,
                linestyle='-' if is_cascade else '--' if 'hgb' in m else '-',
                alpha=1.0 if is_cascade else 0.82,
                label=LABELS[m])

    ax.axhline(1.0, color='black', lw=0.7, alpha=0.4)
    ax.set_ylabel('Cumulative Return (1 = start)')
    ax.set_xlabel('Date')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_title('Cumulative Returns — Evaluation Period (2021–2026)', fontweight='bold')

    ax.legend(frameon=True, framealpha=0.95, ncol=1,
              bbox_to_anchor=(1.01, 0.5), loc='center left',
              handlelength=2.2, borderpad=0.8, edgecolor='#ccc')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_cumulative.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_cumulative.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig5_cumulative')

def fig_risk_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))

    labels  = [LABELS[m]          for m in ORDER]
    colors  = [COLORS[m]          for m in ORDER]
    vols    = [STATS[m]['vol']     for m in ORDER]
    max_dds = [abs(STATS[m]['max_dd']) for m in ORDER]

    for ax, vals, title, ylabel in [
        (ax1, vols,    'Annualized Volatility',  'Vol (%)'),
        (ax2, max_dds, 'Maximum Drawdown (abs)', 'Max DD (%)'),
    ]:
        bars = ax.bar(labels, vals, color=colors, edgecolor='white',
                      linewidth=0.8, width=0.6, zorder=3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9.5)
        bars[-1].set_edgecolor('#155724')
        bars[-1].set_linewidth(2.0)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelsize=9, rotation=30)
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_risk.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig6_risk.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved fig6_risk')

if __name__ == '__main__':
    print('Generating figures...')
    fig_pipeline()
    fig_sharpe()
    fig_risk_return()
    fig_sentiment()
    fig_cumulative()
    fig_risk_breakdown()
    print(f'\nAll figures saved to {FIG_DIR}')
