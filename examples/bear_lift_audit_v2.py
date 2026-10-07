"""2022-2024 整体熊市 — 快速版 (用 numpy 加速)
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "/media/felix/f/quant/aurumq-rl/data/wavehunter_v10_1_hs300_20040102_20260827.parquet"

raw = pl.read_parquet(PANEL)

base_cols = ['trade_date', 'ts_code', 'open', 'close']
factor_cols = [c for c in raw.columns if c not in base_cols
               and not c.startswith('v10_1_zig') and not c.startswith('v10_1_peak')
               and not c.startswith('v10_1_valley') and not c.startswith('v10_1_a')
               and not c.startswith('v10_1_b1') and not c.startswith('v10_1_down')
               and c not in ('ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                             'pct_chg', 'vol', 'amount', 'adj_factor', 'adj_factor_inferred')]
print(f"Factor cols: {len(factor_cols)}")

panel_pd = raw.select(['trade_date', 'ts_code', 'open', 'close'] + factor_cols).to_pandas()
panel_pd['trade_date'] = pd.to_datetime(panel_pd['trade_date'])
panel_pd = panel_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

panel_pd['open_t1'] = panel_pd.groupby('ts_code')['open'].shift(-1)
panel_pd['close_t21'] = panel_pd.groupby('ts_code')['close'].shift(-21)
panel_pd['net20'] = (panel_pd['close_t21'] / panel_pd['open_t1'] - 1) - 0.005

panel_pd['date'] = panel_pd['trade_date'].dt.date
mask = (panel_pd['date'] >= date(2022, 1, 1)) & (panel_pd['date'] <= date(2024, 12, 31))
df = panel_pd[mask].copy().dropna(subset=['net20'])
print(f"2022-2024 samples: {len(df)}")
print(f"Unique dates: {df['date'].nunique()}")

results = []
for fac in factor_cols:
    sub = df[['date', fac, 'net20']].dropna()
    if sub[fac].nunique() < 10:
        continue
    try:
        # Use scipy-style rank percentile per group
        # Compute percentile rank (0-1) per date
        sub['pct_rank'] = sub.groupby('date')[fac].rank(pct=True, na_option='keep')
        sub = sub.dropna(subset=['pct_rank'])
        if len(sub) < 100:
            continue
        # Top quintile = pct_rank >= 0.8, Bottom = pct_rank <= 0.2
        top_q = sub[sub['pct_rank'] >= 0.8]['net20']
        bot_q = sub[sub['pct_rank'] <= 0.2]['net20']
        if len(top_q) < 10 or len(bot_q) < 10:
            continue
        lift = top_q.mean() - bot_q.mean()
        # top 10 vs bottom 10 per day
        per_day_top10 = sub.groupby('date').apply(
            lambda g: g.nlargest(10, 'pct_rank')['net20'].mean()
        )
        per_day_bot10 = sub.groupby('date').apply(
            lambda g: g.nsmallest(10, 'pct_rank')['net20'].mean()
        )
        # per-day corr (using Spearman via rank)
        per_day_corr = sub.groupby('date').apply(
            lambda g: g['pct_rank'].corr(g['net20']) if g['pct_rank'].std() > 0 else 0
        )
        results.append({
            'factor': fac,
            'lift': lift,
            'top_q_mean': top_q.mean(),
            'bot_q_mean': bot_q.mean(),
            'top10_per_day': per_day_top10.mean(),
            'bot10_per_day': per_day_bot10.mean(),
            'mean_corr': per_day_corr.mean(),
        })
    except Exception as e:
        continue

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"\n=== TOP 30 ===")
print(results_df.head(30).to_string(index=False))
print(f"\n=== BOTTOM 30 ===")
print(results_df.tail(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total factors: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift'] > 0.10).sum()}")
print(f"lift < -0.01: {(results_df['lift'] < -0.01).sum()}")
print(f"Mean lift: {results_df['lift'].mean()*100:.3f}%")
print(f"Top 10 mean lift: {results_df.head(10)['lift'].mean()*100:.3f}%")
print(f"Top 20 mean lift: {results_df.head(20)['lift'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v22_v27_bear_factor_lift_2022_2024_overall.csv', index=False)