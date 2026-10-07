"""Step 0-2: 用户合同熊市策略框架 (FAST loop version)
- Pre-convert to numpy array, compute rank per (factor, date) on-the-fly
- Skip precompute to save memory
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
# Load + compute net20 in polars
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

# Convert to pandas with date as integer index for fast lookup
df_pd = df.to_pandas()
date_unique = df_pd['date'].unique()
date_to_int = {d: i for i, d in enumerate(date_unique)}
df_pd['date_int'] = df_pd['date'].map(date_to_int)
df_pd = df_pd.sort_values(['date_int', 'ts_code']).reset_index(drop=True)
n_dates = len(date_unique)
n_stocks = df_pd['ts_code'].nunique()
print(f"  Sorted by date,stock: {df_pd.shape}, n_dates={n_dates}, n_stocks={n_stocks}")

# Date group boundaries (start indices per date)
date_groups = df_pd.groupby('date_int').size().to_dict()
date_starts = {}
date_ends = {}
cum = 0
for d in sorted(date_groups.keys()):
    date_starts[d] = cum
    cum += date_groups[d]
    date_ends[d] = cum

date_int_arr = df_pd['date_int'].values
net20_arr = df_pd['net20'].values

results = []
t1 = time.time()
for fi, fac in enumerate(STOCKPICK_COLS):
    vals = df_pd[fac].values
    if np.isnan(vals).all():
        continue
    rank = np.empty(len(vals))
    rank[:] = np.nan
    for d in range(n_dates):
        s, e = date_starts[d], date_ends[d]
        sub = vals[s:e]
        # Skip if too few
        valid = ~np.isnan(sub)
        if valid.sum() < 10:
            continue
        # Rank within date, normalize to [0, 1] based on valid
        sub_rank = pd.Series(sub).rank(pct=True, na_option='keep').values
        rank[s:e] = sub_rank
    valid_mask = ~np.isnan(rank)
    top_mask = valid_mask & (rank >= 0.8)
    bot_mask = valid_mask & (rank <= 0.2)
    if top_mask.sum() < 50 or bot_mask.sum() < 50:
        continue
    top_mean = net20_arr[top_mask].mean()
    bot_mean = net20_arr[bot_mask].mean()
    lift = top_mean - bot_mean
    results.append({
        'factor': fac,
        'lift': lift,
        'top_q_mean': top_mean,
        'bot_q_mean': bot_mean,
        'n_top': int(top_mask.sum()),
        'n_bot': int(bot_mask.sum()),
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