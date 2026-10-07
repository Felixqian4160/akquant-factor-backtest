"""熊市因子验证 (2022-2024 ONLY) — 单一最关键问题:
   当前因子库在熊市能不能找到有收益的股票?

合同:
- 窗口: 2022-01-01 ~ 2024-12-31
- Ground truth: ZigZag bear legs (5 段)
- 排除 bull/zone legs
- Forward 20d 净收益 (T+1 open → T+21 close, -0.5% round-trip cost)
- 单因子 lift = mean(top quintile net20) - mean(bottom quintile net20)
- top_k lift = mean(net20 of top_k stocks) - mean(net20 of bottom_k stocks)
- 不调参、不打包、不组合 — 只看 single-variable
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date

PANEL = "/media/felix/f/quant/aurumq-rl/data/wavehunter_v10_1_hs300_20040102_20260827.parquet"

print("Loading panel...")
raw = pl.read_parquet(PANEL)
print(f"Panel shape: {raw.shape}")

# Bear windows
bear_windows = [
    ('2022-01-01', '2022-04-26'),
    ('2022-07-04', '2022-10-31'),
    ('2023-01-30', '2024-02-02'),
    ('2024-05-20', '2024-09-13'),
    ('2024-10-08', '2024-12-31'),
]
bear_start = '2022-01-01'
bear_end = '2024-12-31'

# Build bear-day set
bear_days = set()
for s, e in bear_windows:
    dates = pd.date_range(s, e, freq='B')
    for d in dates:
        bear_days.add(d.date())
print(f"Bear days: {len(bear_days)}")

# Get base columns and factor columns
base_cols = ['trade_date', 'ts_code', 'open', 'close', 'adj_close']
factor_cols = [c for c in raw.columns if c not in base_cols
               and not c.startswith('v10_1_zig') and not c.startswith('v10_1_peak')
               and not c.startswith('v10_1_valley') and not c.startswith('v10_1_a')
               and not c.startswith('v10_1_b1') and not c.startswith('v10_1_down')
               and c not in ('ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                             'pct_chg', 'vol', 'amount', 'adj_factor', 'adj_factor_inferred')]
print(f"Factor columns: {len(factor_cols)}")

# Compute forward 20d net return: T+1 open → T+21 close, -0.5% round-trip
# First compute T+1_ret = close[t+1] / open[t] - 1
# Then forward 20d = T+1_ret * T+21_close/open — but simpler is:
# forward_20d = close[t+21] / open[t+1] - 1 (so we need 21-day forward)
# Actually for a 20-day holding from open[t+1] to close[t+21]:
# net20 = (close[t+21] / open[t+1] - 1) - 0.005

print("\nComputing forward 20d net return...")
panel_pd = raw.select(['trade_date', 'ts_code', 'open', 'close'] + factor_cols).to_pandas()
panel_pd['trade_date'] = pd.to_datetime(panel_pd['trade_date'])
panel_pd = panel_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

# Per stock, compute close[t+21] and open[t+1]
panel_pd['open_t1'] = panel_pd.groupby('ts_code')['open'].shift(-1)
panel_pd['close_t21'] = panel_pd.groupby('ts_code')['close'].shift(-21)
panel_pd['net20'] = (panel_pd['close_t21'] / panel_pd['open_t1'] - 1) - 0.005  # 0.5% round-trip

# Filter to bear days only
panel_pd['date'] = panel_pd['trade_date'].dt.date
bear_mask = panel_pd['date'].isin(bear_days)
bear_pd = panel_pd[bear_mask].copy()
bear_pd = bear_pd.dropna(subset=['net20'])
print(f"Bear stock-day samples: {len(bear_pd)}")

# === Single-variable lift audit ===
# For each factor: rank stocks by factor value within each date, compute net20 by quintile
results = []
for fac in factor_cols:
    sub = bear_pd[['date', fac, 'net20']].dropna()
    if sub[fac].nunique() < 10:
        continue
    # Compute quintile per date
    try:
        sub['quintile'] = sub.groupby('date')[fac].transform(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'))
        sub = sub.dropna(subset=['quintile'])
        if sub['quintile'].nunique() < 5:
            continue
        # Compute mean net20 by quintile
        q_means = sub.groupby('quintile')['net20'].mean()
        if 0 in q_means.index and 4 in q_means.index:
            lift = q_means[4] - q_means[0]  # top - bottom
            # top_k lift: top 10 stocks per day by factor
            top_k_means = sub[sub['quintile'] == 4]['net20'].mean()
            bot_k_means = sub[sub['quintile'] == 0]['net20'].mean()
            # Spearman correlation
            corr = sub.groupby('date').apply(
                lambda g: g[fac].corr(g['net20']) if g[fac].std() > 0 else 0
            ).mean()
            results.append({
                'factor': fac,
                'lift_top_bottom': lift,
                'top_q_mean': q_means[4],
                'bot_q_mean': q_means[0],
                'mean_corr': corr,
            })
    except Exception as e:
        continue

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('lift_top_bottom', ascending=False)
print(f"\nSingle-variable bear lift audit ({len(results_df)} valid factors)")
print(f"\n=== TOP 30 factors by lift (top quintile vs bottom quintile mean net20 diff) ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== BOTTOM 30 factors (negative lift = factor works in REVERSE direction) ===")
print(results_df.tail(30).to_string(index=False))

# Summary stats
print(f"\n=== SUMMARY ===")
print(f"Total factors analyzed: {len(results_df)}")
print(f"Factors with lift > 0.01 (1% spread): {(results_df['lift_top_bottom'] > 0.01).sum()}")
print(f"Factors with lift > 0.05 (5% spread): {(results_df['lift_top_bottom'] > 0.05).sum()}")
print(f"Factors with lift > 0.10 (10% spread): {(results_df['lift_top_bottom'] > 0.10).sum()}")
print(f"Factors with lift < -0.01: {(results_df['lift_top_bottom'] < -0.01).sum()}")
print(f"Mean lift: {results_df['lift_top_bottom'].mean()*100:.3f}%")
print(f"Median lift: {results_df['lift_top_bottom'].median()*100:.3f}%")
print(f"Top 10 lift mean: {results_df.head(10)['lift_top_bottom'].mean()*100:.3f}%")

# Save
results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v22_v27_bear_factor_lift_audit_20260924.csv', index=False)
print(f"\nSaved to evidence/v22_v27_bear_factor_lift_audit_20260924.csv")