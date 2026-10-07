"""Step 0-2: 用户合同熊市策略框架 (POLARS ONLY loop)
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

all_cols = pl.scan_parquet(PANEL).collect_schema().names()
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'cap', 'circ_cap', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate',
            'volume', 'amount')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]

print(f"StockPick: {len(STOCKPICK_COLS)} factors")

t0 = time.time()
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close'] + STOCKPICK_COLS)
df = df.sort(['ts_code', 'trade_date'])
df = df.with_columns([
    pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
    pl.col('close').shift(-21).over('ts_code').alias('close_t21'),
    pl.col('trade_date').dt.date().alias('date'),
])
df = df.with_columns([
    ((pl.col('close_t21') / pl.col('open_t1') - 1) - 0.005).alias('net20')
])
df = df.filter(
    (pl.col('date') >= pl.lit(date(2022, 1, 1)))
    & (pl.col('date') <= pl.lit(date(2024, 12, 31)))
).drop_nulls('net20')
print(f"  Loaded+prep: {df.shape}, {time.time()-t0:.1f}s")

results = []
t1 = time.time()
for fi, fac in enumerate(STOCKPICK_COLS):
    # Add rank column for this factor
    df_r = df.with_columns([
        (pl.col(fac).rank().over('date') / pl.col(fac).count().over('date')).alias('_r')
    ])
    # Top q
    top_q = df_r.filter(pl.col('_r') >= 0.8)['net20'].mean()
    bot_q = df_r.filter(pl.col('_r') <= 0.2)['net20'].mean()
    if top_q is None or bot_q is None or np.isnan(top_q) or np.isnan(bot_q):
        continue
    lift = top_q - bot_q
    # Per-day corr
    per_day_corr = df_r.group_by('date').agg([
        pl.corr(pl.col('_r'), pl.col('net20')).alias('c')
    ])['c'].drop_nulls().mean()
    if per_day_corr is None:
        per_day_corr = 0
    results.append({
        'factor': fac,
        'lift': lift,
        'top_q_mean': top_q,
        'bot_q_mean': bot_q,
        'mean_corr': per_day_corr,
    })
    if (fi + 1) % 50 == 0:
        print(f"  [{fi+1}/{len(STOCKPICK_COLS)}] elapsed: {time.time()-t1:.1f}s")

print(f"\nTotal: {time.time()-t1:.1f}s for {len(STOCKPICK_COLS)} factors")

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"\nValid: {len(results_df)}")
print(f"\n=== TOP 30 ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift'] > 0.10).sum()}")
print(f"lift < -0.01: {(results_df['lift'] < -0.01).sum()}")
print(f"Mean lift: {results_df['lift'].mean()*100:.3f}%")
print(f"Top 10 mean: {results_df.head(10)['lift'].mean()*100:.3f}%")
print(f"Top 30 mean: {results_df.head(30)['lift'].mean()*100:.3f}%")
print(f"Top 80 mean: {results_df.head(80)['lift'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v28_bear_lift_404factors_2022_2024.csv', index=False)
print(f"\nSaved!")