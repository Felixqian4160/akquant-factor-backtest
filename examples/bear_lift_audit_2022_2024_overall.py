"""2022-2024 整体熊市研究 — 简单版
- 整个 2022-2024 当作熊市段
- per-day top-k net20 (T+1 open → T+21 close)
- 单因子 lift = mean(top quintile net20) - mean(bottom quintile net20)
- 全因子 panel
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "/media/felix/f/quant/aurumq-rl/data/wavehunter_v10_1_hs300_20040102_20260827.parquet"

print("Loading...")
raw = pl.read_parquet(PANEL)
print(f"Panel: {raw.shape}")

base_cols = ['trade_date', 'ts_code', 'open', 'close']
factor_cols = [c for c in raw.columns if c not in base_cols
               and not c.startswith('v10_1_zig') and not c.startswith('v10_1_peak')
               and not c.startswith('v10_1_valley') and not c.startswith('v10_1_a')
               and not c.startswith('v10_1_b1') and not c.startswith('v10_1_down')
               and c not in ('ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                             'pct_chg', 'vol', 'amount', 'adj_factor', 'adj_factor_inferred')]
print(f"Factor cols: {len(factor_cols)}")

# Compute net20
print("\nComputing net20 (T+1 open → T+21 close, -0.5% cost)...")
panel_pd = raw.select(['trade_date', 'ts_code', 'open', 'close'] + factor_cols).to_pandas()
panel_pd['trade_date'] = pd.to_datetime(panel_pd['trade_date'])
panel_pd = panel_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

panel_pd['open_t1'] = panel_pd.groupby('ts_code')['open'].shift(-1)
panel_pd['close_t21'] = panel_pd.groupby('ts_code')['close'].shift(-21)
panel_pd['net20'] = (panel_pd['close_t21'] / panel_pd['open_t1'] - 1) - 0.005

# 整体 2022-2024
panel_pd['date'] = panel_pd['trade_date'].dt.date
mask = (panel_pd['date'] >= date(2022, 1, 1)) & (panel_pd['date'] <= date(2024, 12, 31))
panel_22y = panel_pd[mask].copy()
panel_22y = panel_22y.dropna(subset=['net20'])
print(f"2022-2024 samples: {len(panel_22y)}")

# === Single-variable audit ===
results = []
for fac in factor_cols:
    sub = panel_22y[['date', fac, 'net20']].dropna()
    if sub[fac].nunique() < 10:
        continue
    try:
        sub['quintile'] = sub.groupby('date')[fac].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
        )
        sub = sub.dropna(subset=['quintile'])
        if sub['quintile'].nunique() < 5:
            continue
        q_means = sub.groupby('quintile')['net20'].mean()
        if 0 in q_means.index and 4 in q_means.index:
            lift = q_means[4] - q_means[0]
            # top-K lift
            top_q = sub[sub['quintile'] == 4]['net20'].mean()
            bot_q = sub[sub['quintile'] == 0]['net20'].mean()
            # per-day corr
            corr = sub.groupby('date').apply(
                lambda g: g[fac].corr(g['net20']) if g[fac].std() > 0 else 0
            ).mean()
            # Also: top-10 stocks per day by factor → mean net20
            top10_per_day = sub.groupby('date').apply(
                lambda g: g.nlargest(10, fac)['net20'].mean()
            ).mean()
            bot10_per_day = sub.groupby('date').apply(
                lambda g: g.nsmallest(10, fac)['net20'].mean()
            ).mean()
            results.append({
                'factor': fac,
                'lift_top_bottom': lift,
                'top_q_mean': top_q,
                'bot_q_mean': bot_q,
                'top10_per_day': top10_per_day,
                'bot10_per_day': bot10_per_day,
                'mean_corr': corr,
            })
    except Exception:
        continue

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('lift_top_bottom', ascending=False)

print(f"\n=== TOP 30 factors by bear lift (2022-2024) ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== BOTTOM 30 ===")
print(results_df.tail(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total factors: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift_top_bottom'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift_top_bottom'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift_top_bottom'] > 0.10).sum()}")
print(f"lift < -0.01: {(results_df['lift_top_bottom'] < -0.01).sum()}")
print(f"Mean lift: {results_df['lift_top_bottom'].mean()*100:.3f}%")
print(f"Median lift: {results_df['lift_top_bottom'].median()*100:.3f}%")
print(f"Top 10 mean lift: {results_df.head(10)['lift_top_bottom'].mean()*100:.3f}%")
print(f"Top 20 mean lift: {results_df.head(20)['lift_top_bottom'].mean()*100:.3f}%")

# Save
results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v22_v27_bear_factor_lift_2022_2024_overall.csv', index=False)
print(f"\nSaved CSV")