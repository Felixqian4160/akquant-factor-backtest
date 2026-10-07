"""合并 talib 到当前因子面板, 删除 mw
- 当前 panel: v10_1 hs300 (346 列, 已有 v10_1_zig 标签)
- v2 panel: 含 talib 73 列
- 合并: 当前 panel + talib_* = 新 panel (不覆盖任何旧文件)
- 删除 mw_*** (用户明确要求)
- 不删除 v10_1_zig (事后峰谷标签, 不能用作 alpha 输入, 但保留作 ground truth)
"""

import polars as pl
import pandas as pd
from datetime import date

PANEL_V10 = "/media/felix/f/quant/aurumq-rl/data/wavehunter_v10_1_hs300_20040102_20260827.parquet"
PANEL_V2 = "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet"
OUTPUT = "/media/felix/f/quant/akquant-factor-backtest/data/wavehunter_hs300_with_talib_20260924.parquet"

print(f"Load v10_1 panel...")
v10 = pl.read_parquet(PANEL_V10)
print(f"  v10 shape: {v10.shape}, cols: {len(v10.columns)}")

print(f"\nLoad v2 panel (talib source)...")
v2 = pl.read_parquet(PANEL_V2)
print(f"  v2 shape: {v2.shape}, cols: {len(v2.columns)}")

# Extract talib columns from v2
talib_cols = [c for c in v2.columns if c.startswith('talib_')]
print(f"\nTalib columns from v2: {len(talib_cols)}")
print(f"v2 join keys: {v2.columns[:3]}")

# Cast trade_date to same precision (datetime[ms]) to match v2
v10_cast = v10.with_columns(pl.col('trade_date').dt.cast_time_unit('ms'))
print(f"v10 cast: {v10_cast.schema['trade_date']}")

# Check overlap of join keys
print(f"\n=== Trade_date / ts_code alignment check ===")
v10_dates = set(v10['trade_date'].unique().to_list())
v2_dates = set(v2['trade_date'].unique().to_list())
print(f"v10 unique dates: {len(v10_dates)}, range: {min(v10_dates)} → {max(v10_dates)}")
print(f"v2 unique dates: {len(v2_dates)}, range: {min(v2_dates)} → {max(v2_dates)}")
print(f"Overlap: {len(v10_dates & v2_dates)} dates")

v10_stocks = set(v10['ts_code'].unique().to_list())
v2_stocks = set(v2['ts_code'].unique().to_list())
print(f"v10 stocks: {len(v10_stocks)}, v2 stocks: {len(v2_stocks)}")
print(f"Stock overlap: {len(v10_stocks & v2_stocks)}")

# Extract talib + key columns from v2
v2_talib = v2.select(['trade_date', 'ts_code'] + talib_cols)
print(f"\nv2_talib shape: {v2_talib.shape}")

# Inner join v10 with v2_talib on (trade_date, ts_code)
print(f"\n=== Joining ===")
# Polars uses inner join by default
merged = v10_cast.join(v2_talib, on=['trade_date', 'ts_code'], how='left')
print(f"After left join: {merged.shape}")

# Check for talib_* duplicates (if v10 already had talib)
overlap = [c for c in talib_cols if c in v10.columns]
print(f"Talib cols already in v10: {len(overlap)}")
if overlap:
    print(f"  -> These will be dropped from v10 side (using v2 versions): {overlap[:5]}")
    merged = merged.drop(overlap)
    print(f"  After dropping: {merged.shape}")

# Check for any mw_* in v10 (user says mw can be deleted)
mw_cols = [c for c in merged.columns if c.startswith('mw_')]
print(f"\nmw_* cols in merged: {len(mw_cols)}")
if mw_cols:
    print(f"  Dropping mw_* (user contract)")
    merged = merged.drop(mw_cols)
    print(f"  After dropping mw: {merged.shape}")

# Sanity check: count columns by prefix
from collections import defaultdict
prefix_groups = defaultdict(int)
for c in merged.columns:
    if c.startswith('alpha_'): prefix_groups['alpha'] += 1
    elif c.startswith('gtja_'): prefix_groups['gtja'] += 1
    elif c.startswith('mw_'): prefix_groups['mw'] += 1
    elif c.startswith('talib_'): prefix_groups['talib'] += 1
    elif c.startswith('idx_'): prefix_groups['idx'] += 1
    elif c.startswith('v10_1_'): prefix_groups['v10_1_zig'] += 1
    else: prefix_groups['base'] += 1
print(f"\n=== Final column grouping ===")
for n, cnt in sorted(prefix_groups.items(), key=lambda x: -x[1]):
    print(f"  {n}: {cnt}")
print(f"  TOTAL: {len(merged.columns)}")

# Save
print(f"\nSaving to {OUTPUT}...")
merged.write_parquet(OUTPUT)
print(f"Done.")

# Verify
verify = pl.read_parquet(OUTPUT)
print(f"Verify: {verify.shape}")
PY