"""Step 0-2: 用户合同熊市策略框架 (BATCH computation)
- Use polars single-pass rank/correlation on all 394 factors
- Avoid per-factor Python loop
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

print(f"StockPick: {len(STOCKPICK_COLS)} factors", flush=True)

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
print(f"  Loaded+prep: {df.shape}, {time.time()-t0:.1f}s", flush=True)

# Step 2: For each factor, compute per-date rank and group into top-q / bot-q
# Strategy: melt to long format, then group
print(f"\n=== Step 2: melt to long + compute lift ===", flush=True)
t1 = time.time()

# Keep only essential cols for melt
id_cols = ['date', 'ts_code', 'net20']
value_cols = STOCKPICK_COLS

df_long = df.select(id_cols + value_cols).unpivot(
    index=id_cols,
    on=value_cols,
    variable_name='factor_name',
    value_name='factor_value'
).drop_nulls('factor_value')
print(f"  Melted: {df_long.shape}, {time.time()-t1:.1f}s", flush=True)

# Per (date, factor) rank
t2 = time.time()
df_long = df_long.with_columns([
    (pl.col('factor_value').rank().over(['date', 'factor_name'])
     / pl.col('factor_value').count().over(['date', 'factor_name'])).alias('pct_rank')
])
print(f"  Ranked: {df_long.shape}, {time.time()-t2:.1f}s", flush=True)

# Compute mean net20 for top_q (rank >= 0.8) and bot_q (rank <= 0.2) per factor
t3 = time.time()
top_q = df_long.filter(pl.col('pct_rank') >= 0.8).group_by('factor_name').agg(
    pl.col('net20').mean().alias('top_q_mean')
)
bot_q = df_long.filter(pl.col('pct_rank') <= 0.2).group_by('factor_name').agg(
    pl.col('net20').mean().alias('bot_q_mean')
)
print(f"  Top/Bot q: {time.time()-t3:.1f}s", flush=True)

# Join and compute lift
results_df = top_q.join(bot_q, on='factor_name').with_columns([
    (pl.col('top_q_mean') - pl.col('bot_q_mean')).alias('lift')
]).sort('lift', descending=True).to_pandas()
print(f"  Done: {results_df.shape}, {time.time()-t0:.1f}s total", flush=True)

print(f"\n=== TOP 30 ===")
print(results_df.head(30).to_string(index=False))

print(f"\n=== SUMMARY ===")
print(f"Total: {len(results_df)}")
print(f"lift > 0.01: {(results_df['lift'] > 0.01).sum()}")
print(f"lift > 0.05: {(results_df['lift'] > 0.05).sum()}")
print(f"lift > 0.10: {(results_df['lift'] > 0.10).sum()}")
print(f"lift > 0.20: {(results_df['lift'] > 0.20).sum()}")
print(f"lift < -0.01: {(results_df['lift'] < -0.01).sum()}")
print(f"lift < -0.05: {(results_df['lift'] < -0.05).sum()}")
print(f"Mean lift: {results_df['lift'].mean()*100:.3f}%")
print(f"Median lift: {results_df['lift'].median()*100:.3f}%")
print(f"Top 10 mean: {results_df.head(10)['lift'].mean()*100:.3f}%")
print(f"Top 30 mean: {results_df.head(30)['lift'].mean()*100:.3f}%")
print(f"Top 80 mean: {results_df.head(80)['lift'].mean()*100:.3f}%")

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v28_bear_lift_394factors_2022_2024.csv', index=False)
print(f"\nSaved!")