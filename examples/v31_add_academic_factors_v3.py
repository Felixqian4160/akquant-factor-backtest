"""V31 v3: 把 V30 学术因子加入 panel + 重构面板 (polars-native, 不 OOM)

策略:
- polars 原生 rolling 操作 (不转 pandas, 避免 OOM)
- 只 select 必要列 (~50 列)
- 写出 panel 时再合并其他原始列 (read_back + select)
"""

import polars as pl
import pandas as pd
import numpy as np
import time
import sys
import os

PANEL_IN = "data/wavehunter_hs300_with_talib_20260924.parquet"
PANEL_OUT = "data/wavehunter_hs300_v31_refactored_20260925.parquet"
PANEL_TIMING = "data/idx_timing_v31_20260925.parquet"
PANEL_LABEL = "data/v10_1_zig_labels_v31_20260925.parquet"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 80)
log("V31 v3: 把 V30 学术因子加入 panel + 重构面板 (polars native)")
log("=" * 80)

# === Load with all cols needed ===
log("Loading panel...")
t0 = time.time()
all_cols = pl.scan_parquet(PANEL_IN).collect_schema().names()
LABEL_COLS = [c for c in all_cols if c.startswith('v10_1_')]
TIMING_COLS = [c for c in all_cols if c.startswith('idx_')]
STOCKPICK_PREFIXES = ('alpha_', 'gtja_', 'talib_')
FIN_BASE = ('pe', 'pb', 'bps', 'roe', 'roe_waa', 'roa',
            'grossprofit_margin', 'netprofit_margin', 'fcff', 'cfps', 'ocfps',
            'netprofit_yoy', 'ocf_yoy', 'or_yoy', 'current_ratio',
            'quick_ratio', 'debt_to_assets', 'assets_turn', 'turnover_rate')
STOCKPICK_COLS = [c for c in all_cols
                  if c.startswith(STOCKPICK_PREFIXES) or c in FIN_BASE]
log(f"  标签层: {len(LABEL_COLS)}, 择时层: {len(TIMING_COLS)}, 选股层: {len(STOCKPICK_COLS)}")

# === Step 0: Save isolated layers ===
log("Saving label/timing isolated layers...")
df = pl.read_parquet(PANEL_IN, columns=['trade_date', 'ts_code'] + LABEL_COLS)
df.write_parquet(PANEL_LABEL)
log(f"  Saved: {PANEL_LABEL}")

df_idx = pl.read_parquet(PANEL_IN, columns=['trade_date', 'ts_code'] + TIMING_COLS)
df_idx = df_idx.sort(['ts_code', 'trade_date'])
# 加 idx_ret_1d (log return)
df_idx = df_idx.with_columns(
    (pl.col('idx_close').pct_change().over('ts_code').log1p()).alias('idx_ret_1d')
)
df_idx.write_parquet(PANEL_TIMING)
log(f"  Saved: {PANEL_TIMING}")

# === Step 1: Compute new factors on minimal cols ===
log("Loading minimal cols for new factor computation...")
BASIC_COLS = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
              'cap', 'circ_cap', 'pe', 'pb', 'bps']
df = pl.read_parquet(PANEL_IN, columns=BASIC_COLS)
df = df.sort(['ts_code', 'trade_date'])
log(f"  Loaded: {df.shape}, time: {time.time()-t0:.1f}s")

# Compute daily return
log("Computing daily_ret...")
df = df.with_columns(
    pl.col('close').pct_change().over('ts_code').alias('daily_ret')
)

# Market return (equal-weight per day)
log("Computing market_ret...")
mkt = (
    df.group_by('trade_date')
    .agg(pl.col('close').mean().alias('mkt_close'))
    .sort('trade_date')
    .with_columns(pl.col('mkt_close').pct_change().alias('mkt_ret'))
)
df = df.join(mkt.select('trade_date', 'mkt_ret'), on='trade_date', how='left')
log(f"  After market merge: {df.shape}")

# === Liquidity (8) ===
log("=== Computing 8 Liquidity factors ===")
df = df.with_columns([
    pl.max_horizontal(pl.col('circ_cap'), 1).log().alias('l_size'),
    (pl.max_horizontal(pl.col('circ_cap'), 1).log() ** 3).alias('l_size3'),
    (pl.col('amount') / pl.max_horizontal(pl.col('circ_cap'), 1)).alias('daily_turnover'),
])
log("  turnm/turna...")
df = df.with_columns([
    pl.col('daily_turnover').rolling_mean(21).over('ts_code').log1p().alias('l_turnm'),
    pl.col('daily_turnover').rolling_mean(252).over('ts_code').log1p().alias('l_turna'),
])
log("  ami...")
df = df.with_columns([
    (pl.col('daily_ret').abs() / pl.max_horizontal(pl.col('amount'), 1)).alias('abs_ret_amt'),
])
df = df.with_columns([
    pl.col('abs_ret_amt').rolling_mean(21).over('ts_code').alias('l_ami'),
])
df = df.drop('abs_ret_amt')
log("  dtvm/dtva/vdtv...")
df = df.with_columns([
    pl.col('amount').rolling_mean(21).over('ts_code').log1p().alias('l_dtvm'),
    pl.col('amount').rolling_mean(252).over('ts_code').log1p().alias('l_dtva'),
    pl.col('amount').rolling_std(120).over('ts_code').log1p().alias('l_vdtv'),
])

# === Risk (2) ===
log("=== Computing 2 Risk factors ===")
df = df.with_columns([
    pl.col('daily_ret').rolling_std(60).over('ts_code').alias('r_tv'),
])
log("  r_beta (60d cov/var)...")
# polars 没有 rolling_cov/rolling_corr, 改用 pandas groupby rolling
import pandas as pd
_beta_df = df.select(['ts_code', 'trade_date', 'daily_ret', 'mkt_ret']).to_pandas()
_beta_df = _beta_df.sort_values(['ts_code', 'trade_date'])
_beta_df['_beta'] = _beta_df.groupby('ts_code').apply(
    lambda g: g['daily_ret'].rolling(60).cov(g['mkt_ret']) / g['mkt_ret'].rolling(60).var().replace(0, np.nan),
    include_groups=False
).reset_index(level=0, drop=True).values
df = df.join(
    pl.from_pandas(_beta_df[['ts_code', 'trade_date', '_beta']].rename(columns={'_beta': 'r_beta'})),
    on=['ts_code', 'trade_date'], how='left',
)

# === Past Returns (10) ===
log("=== Computing 10 Past Returns factors ===")
df = df.with_columns([
    pl.col('close').shift(21).over('ts_code').pct_change(21).alias('p_m1'),
    pl.col('close').shift(21).over('ts_code').pct_change(63).alias('p_m3'),
    pl.col('close').shift(21).over('ts_code').pct_change(126).alias('p_m6'),
    pl.col('close').shift(21).over('ts_code').pct_change(231).alias('p_m11'),
    pl.col('close').shift(21).over('ts_code').pct_change(504).alias('p_m24'),
])
log("  p_mchg (m6 - m12)...")
df = df.with_columns([
    (pl.col('close').pct_change(126).shift(21).over('ts_code')
     - pl.col('close').pct_change(252).shift(21).over('ts_code')).alias('p_mchg'),
])
log("  p_52w...")
df = df.with_columns([
    (pl.col('close') / pl.col('close').rolling_max(252).over('ts_code')).alias('p_52w'),
    pl.col('daily_ret').rolling_max(21).over('ts_code').alias('p_mdr'),
])
log("  p_pr, p_season...")
df = df.with_columns([
    pl.max_horizontal(pl.col('close'), 0.01).log().alias('p_pr'),
    (pl.col('trade_date').dt.month() == 12).cast(pl.Int8).alias('p_season'),
])

# === Value (2) ===
log("=== Computing 2 Value factors ===")
df = df.with_columns([
    (pl.col('bps') / pl.max_horizontal(pl.col('close'), 0.01)).alias('v_bm'),
    (1.0 / pl.max_horizontal(pl.col('pe'), 1)).alias('v_ep'),
])

# Drop helpers
df = df.drop(['daily_ret', 'daily_turnover', 'mkt_ret'])

log(f"  After all factor computation: {df.shape}")
log(f"  Memory: {df.estimated_size('mb'):.1f} MB")

# === Now load original STOCKPICK_COLS from source and concat ===
log("Loading original STOCKPICK_COLS from source...")
df_orig = pl.read_parquet(PANEL_IN, columns=['trade_date', 'ts_code'] + STOCKPICK_COLS)
log(f"  Loaded orig: {df_orig.shape}")

log("Joining new factors to original stock-pick panel...")
# Join on trade_date + ts_code
df_final = df_orig.join(
    df.select(['trade_date', 'ts_code'] + [
        'l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami',
        'l_dtvm', 'l_dtva', 'l_vdtv',
        'r_tv', 'r_beta',
        'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
        'p_52w', 'p_mdr', 'p_pr', 'p_season',
        'v_bm', 'v_ep',
    ]),
    on=['trade_date', 'ts_code'],
    how='left',
)
log(f"  Joined: {df_final.shape}")

# === Save ===
log(f"Saving: {PANEL_OUT}")
df_final.write_parquet(PANEL_OUT)
log(f"  Size: {os.path.getsize(PANEL_OUT)/1e9:.2f} GB")
log(f"  Columns: {len(df_final.columns)} ({len(STOCKPICK_COLS)} orig + 22 new)")

# Verify
log("Verifying...")
df_verify = pl.read_parquet(PANEL_OUT)
log(f"Verify: {df_verify.shape}")
log(f"Date range: {df_verify['trade_date'].min()} → {df_verify['trade_date'].max()}")
log(f"Stocks: {df_verify['ts_code'].n_unique()}")

NEW_FACTORS = ['l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami',
                'l_dtvm', 'l_dtva', 'l_vdtv',
                'r_tv', 'r_beta',
                'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
                'p_52w', 'p_mdr', 'p_pr', 'p_season',
                'v_bm', 'v_ep']
log("New factor coverage:")
for fac in NEW_FACTORS:
    rate = df_verify[fac].is_null().mean()
    log(f"  {fac:15s}: {rate*100:.1f}% NaN")

log("=" * 80)
log("✅ Done")
log("=" * 80)