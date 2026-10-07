"""Step 0-2: 用户合同熊市策略框架 (FAST per-factor, lazy-free)
- Step 0: 物理分层 (394 stock-pick factors)
- Step 1: 预处理
- Step 2: 单变量熊市 lift audit — per-factor loop with pandas groupby rank

Each factor: pandas groupby rank + boolean mask + mean = ~1-2s
394 factors * 1.5s = ~10 min
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
import time
import sys

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
# Load only essential columns: net20 needs open_t1, close_t21
df = pl.read_parquet(PANEL).select(['trade_date', 'ts_code', 'open', 'close'])
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
).drop_nulls('net20').select(['date', 'ts_code', 'net20']).to_pandas()
# In pandas, pl.Date converts to datetime64, cast to Python date for clean merge
df['date'] = pd.to_datetime(df['date']).dt.date
print(f"  net20 prep: {df.shape}, {time.time()-t0:.1f}s", flush=True)

# For each factor, load just that column + net20
results = []
t1 = time.time()
for fi, fac in enumerate(STOCKPICK_COLS):
    # Load only this factor
    fac_df = pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', fac]).to_pandas()
    fac_df = fac_df.rename(columns={fac: 'fval'}).dropna(subset=['fval'])
    fac_df['date'] = pd.to_datetime(fac_df['trade_date']).dt.date
    # Inner join with net20 df
    merged = fac_df.merge(df, on=['date', 'ts_code'], how='inner')
    # Per-date rank
    merged['_r'] = merged.groupby('date')['fval'].rank(pct=True, na_option='keep')
    valid = merged['_r'].notna()
    top = merged[valid & (merged['_r'] >= 0.8)]['net20'].mean()
    bot = merged[valid & (merged['_r'] <= 0.2)]['net20'].mean()
    if pd.isna(top) or pd.isna(bot):
        continue
    results.append({
        'factor': fac,
        'lift': top - bot,
        'top_q_mean': top,
        'bot_q_mean': bot,
        'n_samples': len(merged),
    })
    if (fi + 1) % 25 == 0:
        print(f"  [{fi+1}/{len(STOCKPICK_COLS)}] elapsed: {time.time()-t1:.1f}s, valid: {len(results)}", flush=True)

print(f"\nTotal: {time.time()-t1:.1f}s for {len(STOCKPICK_COLS)} factors, valid: {len(results)}", flush=True)

results_df = pd.DataFrame(results).sort_values('lift', ascending=False)
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