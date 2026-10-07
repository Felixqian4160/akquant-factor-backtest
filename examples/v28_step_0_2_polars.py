"""Step 0-2: 用户合同熊市策略框架 (polars 全程版)
- Step 0: 物理分层
- Step 1: 预处理
- Step 2: 熊市 lift audit 404 因子 — 不删除, 只排序看哪些有 alpha

策略: 用 polars group_by + quantile for per-day rank, much faster than pandas
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"

# ========== Step 0: 物理分层 ==========
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

print(f"Layer | Count")
print(f"Label (v10_1_*) | {len(LABEL_COLS)}")
print(f"Timing (idx_*) | {len(TIMING_COLS)}")
print(f"StockPick (gtja+alpha+talib+base) | {len(STOCKPICK_COLS)}")

# ========== Step 1: 预处理 ==========
t0 = time.time()
print(f"\n=== Loading panel + compute net20 (in polars) ===")
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close'] + STOCKPICK_COLS)
print(f"  Loaded: {df.shape}, time: {time.time()-t0:.1f}s")

# Compute net20 in polars (fast vectorized)
df = df.sort(['ts_code', 'trade_date'])
df = df.with_columns([
    pl.col('open').shift(-1).over('ts_code').alias('open_t1'),
    pl.col('close').shift(-21).over('ts_code').alias('close_t21'),
])
df = df.with_columns([
    ((pl.col('close_t21') / pl.col('open_t1') - 1) - 0.005).alias('net20')
])
df = df.with_columns([pl.col('trade_date').dt.date().alias('date')])

# Filter to 2022-2024
df = df.filter(
    (pl.col('date') >= pl.lit(date(2022, 1, 1)))
    & (pl.col('date') <= pl.lit(date(2024, 12, 31)))
)
df = df.drop_nulls('net20')
print(f"  2022-2024: {df.shape}, time: {time.time()-t0:.1f}s")

# ========== Step 2: Single-variable lift audit (polars) ==========
# For each factor, compute per-date rank + quintile mean net20
print(f"\n=== Step 2: Polars bear lift audit (404 factors) ===")
# Add a per-date row number for joining
df = df.with_row_index('row_idx')

results = []
t1 = time.time()
for i, fac in enumerate(STOCKPICK_COLS):
    # Per-date percentile rank using polars window
    fac_rank = f"{fac}_rank"
    df_fac = df.with_columns([
        pl.col(fac).rank(pct=True).over('date').alias(fac_rank)
    ])
    # Filter to top quintile (rank >= 0.8) and bottom (rank <= 0.2)
    top_mean = df_fac.filter(pl.col(fac_rank) >= 0.8)['net20'].mean()
    bot_mean = df_fac.filter(pl.col(fac_rank) <= 0.2)['net20'].mean()
    if top_mean is None or bot_mean is None:
        continue
    lift = top_mean - bot_mean
    # Per-day Spearman-like corr using pct_rank vs net20
    per_day_corrs = []
    # Approx corr via group_by + corr expression
    corr_df = df_fac.group_by('date').agg([
        pl.corr(pl.col(fac_rank), pl.col('net20')).alias('corr')
    ]).filter(pl.col('corr').is_not_null())
    if len(corr_df) == 0:
        continue
    mean_corr = corr_df['corr'].mean()
    results.append({
        'factor': fac,
        'lift': lift,
        'top_q_mean': top_mean,
        'bot_q_mean': bot_mean,
        'mean_corr': mean_corr,
    })

print(f"  Done {len(STOCKPICK_COLS)} factors in {time.time()-t1:.1f}s")

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
print(f"Valid: {len(results_df)}")
print(f"\n=== TOP 30 ===")
print(results_df.head(30).to_string(index=False))
print(f"\n=== BOTTOM 30 ===")
print(results_df.tail(30).to_string(index=False))

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

results_df.to_csv('/media/felix/f/quant/akquant-factor-backtest/evidence/v28_bear_lift_404factors_2022_2024.csv', index=False)