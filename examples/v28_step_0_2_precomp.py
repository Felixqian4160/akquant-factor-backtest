"""Step 0-2: 用户合同熊市策略框架 (PRECOMPUTE per-day rank once)
- Pre-compute per-day percentile rank for ALL 404 factors ONCE
- Then aggregate mean net20 by quintile buckets
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

all_cols = pl.scan_parquet(PANEL).collect_schema().names()
LABEL_COLS = [c for c in all_cols if c.startswith('v10_1_')]
TIMING_COLS = [c for c in all_cols if c.startswith('idx_')]
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'cap', 'circ_cap', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate',
            'volume', 'amount')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]

print(f"StockPick: {len(STOCKPICK_COLS)} factors")

# ========== Pre-compute: load panel, compute net20, filter 2022-2024 ==========
print(f"\n=== Load + prep ===")
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
print(f"  2022-2024: {df.shape}, time: {time.time()-t0:.1f}s")

# ========== Compute per-day percentile rank for all 404 factors ==========
# Use lazy plan: rank().over('date') per factor
print(f"\n=== Pre-compute per-day rank for 404 factors (vectorized polars) ===")
t1 = time.time()
# Use a single lazy pipeline: cast + rank
exprs = []
for fac in STOCKPICK_COLS:
    exprs.append(pl.col(fac).rank(pct=True).over('date').alias(fac + '_r'))
df_ranked = df.with_columns(exprs)
print(f"  Ranked: {df_ranked.shape}, time: {time.time()-t1:.1f}s")

# ========== Step 2: Single-variable lift (one big aggregate) ==========
print(f"\n=== Compute top-q / bot-q means for each factor ===")
# Use polars aggregation: filter to top-q (rank >= 0.8) for each factor, mean(net20)
# But we have 404 *_r columns and one net20. Can't pivot easily.
# Instead, group by date into one of [top_q, bot_q] per factor — too complex.
# Alternative: pivot to long format with factor/rank/net20, then aggregate

# Use numpy directly on the df_ranked dataframe for speed
# Convert to pandas
print(f"  Converting to pandas...")
df_pd = df_ranked.select(['date', 'net20'] + [f + '_r' for f in STOCKPICK_COLS]).to_pandas()
print(f"  pandas shape: {df_pd.shape}, time: {time.time()-t1:.1f}s")

# Pre-compute net20 mask once
net20 = df_pd['net20'].values
date_idx = df_pd['date'].values

results = []
print(f"  Computing lift for each factor...")
t2 = time.time()
for fac in STOCKPICK_COLS:
    r_col = f + '_r' if False else fac + '_r'
    r = df_pd[r_col].values
    valid = ~np.isnan(r)
    top_mask = valid & (r >= 0.8)
    bot_mask = valid & (r <= 0.2)
    if top_mask.sum() < 50 or bot_mask.sum() < 50:
        continue
    top_mean = net20[top_mask].mean()
    bot_mean = net20[bot_mask].mean()
    lift = top_mean - bot_mean
    results.append({
        'factor': fac,
        'lift': lift,
        'top_q_mean': top_mean,
        'bot_q_mean': bot_mean,
        'n_top': int(top_mask.sum()),
        'n_bot': int(bot_mask.sum()),
    })
print(f"  Done in {time.time()-t2:.1f}s")

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"Valid: {len(results_df)}")
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
print(f"\nSaved: evidence/v28_bear_lift_404factors_2022_2024.csv")