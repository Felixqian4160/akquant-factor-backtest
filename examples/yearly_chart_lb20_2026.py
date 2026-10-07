"""Generate yearly returns bar chart for v33_factorrank_lb20 OOS."""
import csv, json, pathlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = pathlib.Path('/media/felix/f/quant/akquant-factor-backtest')
PANEL = ROOT/'data/wavehunter_hs300_v33_with_new_factors_20261003.parquet'
OUT_DIR = ROOT/'evidence/sweep/v33_factorrank_lb20_2026_oos'

with open(OUT_DIR/'akquant'/'V14_0'/'nav.csv') as f:
    nav = list(csv.DictReader(f))

idx = (pl.read_parquet(PANEL, columns=['trade_date','idx_close'])
       .filter(pl.col('idx_close').is_not_null())
       .group_by('trade_date').agg(pl.col('idx_close').first())
       .sort('trade_date'))
idx_dates = [str(x)[:10] for x in idx['trade_date'].to_list()]
idx_vals = idx['idx_close'].to_numpy()
bench_map = dict(zip(idx_dates, idx_vals))

nav_by_year = {}
for r in nav:
    y = int(r['date'][:4])
    nav_by_year.setdefault(y, []).append(float(r['value']))
hs_by_year = {}
for d, v in bench_map.items():
    y = int(d[:4])
    hs_by_year.setdefault(y, []).append(v)

years = sorted(nav_by_year.keys())
strat_rets = []
hs_rets = []
for y in years:
    if len(nav_by_year[y]) < 2: continue
    sr = (nav_by_year[y][-1] / nav_by_year[y][0] - 1) * 100
    if y in hs_by_year and len(hs_by_year[y]) >= 2:
        hr = (hs_by_year[y][-1] / hs_by_year[y][0] - 1) * 100
    else:
        hr = 0
    strat_rets.append(sr); hs_rets.append(hr)

fig, ax = plt.subplots(figsize=(16, 7))
x = np.arange(len(years))
ax.bar(x - 0.2, strat_rets, width=0.4, color='#1565c0', label='Strategy')
ax.bar(x + 0.2, hs_rets, width=0.4, color='#b71c1c', label='HS300', alpha=0.75)
ax.axhline(0, color='#000', lw=0.5)
for i, (y, sr, hr) in enumerate(zip(years, strat_rets, hs_rets)):
    ax.text(i - 0.2, sr + (2 if sr >= 0 else -3), f'{sr:+.0f}%', ha='center', fontsize=7, color='#1565c0')
    ax.text(i + 0.2, hr + (2 if hr >= 0 else -3), f'{hr:+.0f}%', ha='center', fontsize=7, color='#b71c1c')
    if y == 2026:
        ax.axvspan(i - 0.5, i + 0.5, color='#90caf9', alpha=0.15)
ax.set_xticks(x); ax.set_xticklabels(years, rotation=0)
ax.set_xlabel('Year'); ax.set_ylabel('Annual Return (%)')
ax.set_title('V33 Factor-Return Voting (lookback=20) — Annual Returns vs HS300 (2010-2026)')
ax.legend(loc='upper left'); ax.grid(True, alpha=.22, axis='y')
fig.tight_layout()
out = OUT_DIR / 'yearly_returns_bar.png'
fig.savefig(out, dpi=160, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')

# OOS-only zoom (2026)
om_fig, om_ax = plt.subplots(figsize=(12, 6))
sr_2026 = strat_rets[-1]
hr_2026 = hs_rets[-1]
bars = om_ax.bar(['Strategy (lb=20)', 'HS300 buy&hold'], [sr_2026, hr_2026],
                 color=['#1565c0', '#b71c1c'])
for bar, val in zip(bars, [sr_2026, hr_2026]):
    om_ax.text(bar.get_x() + bar.get_width()/2, val + (0.5 if val >= 0 else -1.5),
               f'{val:+.2f}%', ha='center', fontsize=14, fontweight='bold')
om_ax.axhline(0, color='#000', lw=0.5)
om_ax.set_title('OOS 2026 (8 months, 2026-01-02 → 2026-08-27)\nStrategy +12.0% vs HS300 -2.1% → +14.1% excess')
om_ax.set_ylabel('Return (%)')
om_ax.grid(True, alpha=.22, axis='y')
om_fig.tight_layout()
out2 = OUT_DIR / 'oos_2026_only.png'
om_fig.savefig(out2, dpi=160, bbox_inches='tight')
plt.close(om_fig)
print(f'Saved {out2}')