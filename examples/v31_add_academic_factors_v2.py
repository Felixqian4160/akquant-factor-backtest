"""V31: 把 V30 学术因子加入 panel + 重构面板 (简化版, fast)

不删除任何现有因子 (gtja + alpha + talib + base_fin 共 390)
只**追加**新因子到 panel, 形成 390 + 22 = 412 选股层

新因子 (从 Quantactix/ChinaAShareEquityCharacteristics 移植):
  Liquidity (8): l_size, l_size3, l_turnm, l_turna, l_ami, l_dtvm, l_dtva, l_vdtv
  Risk (2): r_tv, r_beta
  Past Returns (10): p_m1, p_m3, p_m6, p_m11, p_m24, p_mchg, p_52w, p_mdr, p_pr, p_season
  Value (2): v_bm, v_ep
"""

import polars as pl
import pandas as pd
import numpy as np
from datetime import date
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
log("V31: 把 V30 学术因子加入 panel + 重构面板")
log("=" * 80)

# === Step 0: 物理分层 ===
log("Step 0: 物理分层...")
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
log(f"  标签层 (v10_1_zig): {len(LABEL_COLS)} 列")
log(f"  择时层 (idx): {len(TIMING_COLS)} 列")
log(f"  选股层 (gtja+alpha+talib+base): {len(STOCKPICK_COLS)} 列")

# === Load df ===
t0 = time.time()
df = pl.read_parquet(PANEL_IN)
log(f"Loaded: {df.shape}, time: {time.time()-t0:.1f}s")

# === Save isolated label/timing layers ===
log("Saving isolated label/timing layers...")
df.select(['trade_date', 'ts_code'] + LABEL_COLS).write_parquet(PANEL_LABEL)
log(f"  Saved: {PANEL_LABEL}")

df_idx = df.select(['trade_date', 'ts_code'] + TIMING_COLS).sort(['ts_code', 'trade_date']).to_pandas()
# 加 idx_ret_1d (log return)
df_idx['idx_ret_1d'] = df_idx.groupby('ts_code')['idx_close'].transform(
    lambda x: np.log(x.pct_change(fill_method=None) + 1)
)
df_idx_pl = pl.from_pandas(df_idx)
df_idx_pl.write_parquet(PANEL_TIMING)
log(f"  Saved: {PANEL_TIMING}")

# === Step 1: Compute 22 Quantactix academic factors ===
log("Step 1: Compute 22 Quantactix academic factors...")

# Load stock-pick columns
selected_cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
                 'cap', 'circ_cap'] + STOCKPICK_COLS
# De-dup
selected_cols = list(dict.fromkeys(selected_cols))
log(f"  Selecting {len(selected_cols)} unique columns from panel")
df_sp_pd = df.select(selected_cols).to_pandas()
df_sp_pd = df_sp_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
log(f"  Loaded stock-pick panel: {df_sp_pd.shape}")

# Daily return (used in many factors)
log("  Computing daily_ret...")
df_sp_pd['daily_ret'] = df_sp_pd.groupby('ts_code')['close'].transform(
    lambda x: x.pct_change(fill_method=None)
)

# Market return (equal-weight average close)
log("  Computing market_ret...")
market_pd = df_sp_pd.groupby('trade_date')['close'].mean().reset_index()
market_pd.columns = ['trade_date', 'mkt_close']
market_pd['mkt_ret'] = market_pd['mkt_close'].pct_change(fill_method=None)
df_sp_pd = df_sp_pd.merge(market_pd[['trade_date', 'mkt_ret']], on='trade_date', how='left')

# === Liquidity (8) ===
log("  Computing 8 Liquidity factors...")
df_sp_pd['l_size'] = np.log(df_sp_pd['circ_cap'].clip(lower=1))
df_sp_pd['l_size3'] = df_sp_pd['l_size'] ** 3
df_sp_pd['daily_turnover'] = df_sp_pd['amount'] / df_sp_pd['circ_cap'].clip(lower=1)
df_sp_pd['l_turnm'] = np.log(df_sp_pd.groupby('ts_code')['daily_turnover'].transform(
    lambda x: x.rolling(21).mean().clip(lower=1e-9)))
df_sp_pd['l_turna'] = np.log(df_sp_pd.groupby('ts_code')['daily_turnover'].transform(
    lambda x: x.rolling(252).mean().clip(lower=1e-9)))
log("    l_ami...")
df_sp_pd['abs_ret_over_amt'] = df_sp_pd['daily_ret'].abs() / df_sp_pd['amount'].clip(lower=1)
df_sp_pd['l_ami'] = df_sp_pd.groupby('ts_code')['abs_ret_over_amt'].transform(
    lambda x: x.rolling(21).mean())
df_sp_pd.drop(columns=['abs_ret_over_amt'], inplace=True)
df_sp_pd['l_dtvm'] = np.log(df_sp_pd.groupby('ts_code')['amount'].transform(
    lambda x: x.rolling(21).mean().clip(lower=1)))
df_sp_pd['l_dtva'] = np.log(df_sp_pd.groupby('ts_code')['amount'].transform(
    lambda x: x.rolling(252).mean().clip(lower=1)))
df_sp_pd['l_vdtv'] = np.log(df_sp_pd.groupby('ts_code')['amount'].transform(
    lambda x: x.rolling(120).std().clip(lower=1)))
log("    8 Liquidity done")

# === Risk (2) ===
log("  Computing 2 Risk factors...")
df_sp_pd['r_tv'] = df_sp_pd.groupby('ts_code')['daily_ret'].transform(
    lambda x: x.rolling(60).std())

# beta per stock (vectorized)
log("    beta (cov with market, 60d)...")
df_sp_pd['r_beta'] = (
    df_sp_pd.groupby('ts_code')
    .apply(lambda g: g['daily_ret'].rolling(60).cov(g['mkt_ret']) / g['mkt_ret'].rolling(60).var().replace(0, np.nan), include_groups=False)
    .reset_index(level=0, drop=True)
)
log("    2 Risk done")

# === Past Returns (10) ===
log("  Computing 10 Past Returns factors...")
for label, days in [('m1', 21), ('m3', 63), ('m6', 126), ('m11', 231), ('m24', 504)]:
    log(f"    p_{label}...")
    df_sp_pd[f'p_{label}'] = df_sp_pd.groupby('ts_code')['close'].transform(
        lambda x: x.shift(21).pct_change(days))

log("    p_mchg...")
df_sp_pd['p_mchg'] = (
    df_sp_pd.groupby('ts_code')
    .apply(lambda g: g['close'].pct_change(126).shift(21) - g['close'].pct_change(252).shift(21), include_groups=False)
    .reset_index(level=0, drop=True)
)
log("    p_52w...")
df_sp_pd['p_52w'] = df_sp_pd.groupby('ts_code')['close'].transform(
    lambda x: x / x.rolling(252).max())
df_sp_pd['p_mdr'] = df_sp_pd.groupby('ts_code')['daily_ret'].transform(
    lambda x: x.rolling(21).max())
df_sp_pd['p_pr'] = np.log(df_sp_pd['close'].clip(lower=0.01))
df_sp_pd['p_season'] = (pd.to_datetime(df_sp_pd['trade_date']).dt.month == 12).astype(int)
log("    10 Past Returns done")

# === Value (2) ===
log("  Computing 2 Value factors...")
df_sp_pd['v_bm'] = df_sp_pd['bps'] / df_sp_pd['close'].clip(lower=0.01)
df_sp_pd['v_ep'] = 1.0 / df_sp_pd['pe'].clip(lower=1)
log("    2 Value done")

# Drop helper columns
df_sp_pd.drop(columns=['daily_ret', 'mkt_ret', 'mkt_close'], inplace=True, errors='ignore')

# === Save refactored panel ===
log("Saving refactored panel...")
NEW_FACTORS = ['l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami',
                'l_dtvm', 'l_dtva', 'l_vdtv',
                'r_tv', 'r_beta',
                'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
                'p_52w', 'p_mdr', 'p_pr', 'p_season',
                'v_bm', 'v_ep']
log(f"  Original stock-pick: {len(STOCKPICK_COLS)}")
log(f"  New factors added: {len(NEW_FACTORS)}")
log(f"  Total stock-pick: {len(STOCKPICK_COLS) + len(NEW_FACTORS)}")

# Convert back to polars, keep all original + new factors
keep_cols = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount'] + STOCKPICK_COLS + NEW_FACTORS
keep_cols = list(dict.fromkeys(keep_cols))
log(f"  Output columns: {len(keep_cols)}")
df_out = pl.from_pandas(df_sp_pd[keep_cols])
log(f"  Output shape: {df_out.shape}")
df_out.write_parquet(PANEL_OUT)
log(f"Saved: {PANEL_OUT}")
log(f"   Size: {os.path.getsize(PANEL_OUT)/1e9:.2f} GB")
log(f"   Columns: {len(df_out.columns)} ({len(STOCKPICK_COLS)} orig + {len(NEW_FACTORS)} new)")

# Verify
log("Verifying...")
df_verify = pl.read_parquet(PANEL_OUT)
log(f"Verify: {df_verify.shape}")
log(f"Date range: {df_verify['trade_date'].min()} → {df_verify['trade_date'].max()}")
log(f"Stocks: {df_verify['ts_code'].n_unique()}")

# Check new factors have valid values
log("New factor coverage:")
for fac in NEW_FACTORS:
    rate = df_verify[fac].is_null().mean()
    log(f"  {fac:15s}: {rate*100:.1f}% NaN")

log("=" * 80)
log("Done")
log("=" * 80)