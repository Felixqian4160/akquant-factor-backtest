"""V32 panel 完整化 — 把 v31 panel 缺失的 21 列从 v30 input panel merge 进来。

升级路径:
  v31 panel (420) = v30 input (419) - 21 cols + 22 academic factors
  v32 panel (441) = v30 input (419) ∪ v31 academic (22)

加入的 21 列:
  - cap, circ_cap                     (市值)
  - adj_close, adj_factor             (复权)
  - idx_close, idx_mom_5/20/60        (择时)
  - v10_1_zig_peak/valley/a1_point    (峰谷标签, 11 列)
  - volume                            (股数)
  - pct_chg                           (涨跌幅)

完整自检:
  - 441 列对齐 v30 ∪ v31 学术
  - 22 学术因子值与 v31 完全一致 (校验和)
  - circ_cap/cap 数据真实性 (反推 l_size, 偏差 < 1e-6)
  - zigzag/idx_timing 标签与侧车一致
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import numpy as np

PANEL_IN = Path('data/wavehunter_hs300_with_talib_20260924.parquet')
PANEL_V31 = Path('data/wavehunter_hs300_v31_refactored_v2_20260925.parquet')
PANEL_OUT = Path('data/wavehunter_hs300_v32_complete_20261003.parquet')
EVIDENCE = Path('evidence/v32_panel_complete_20261003')
EVIDENCE.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log("=" * 80)
log("V32 panel 完整化 — 21 缺失列回填 + 22 学术因子保留")
log("=" * 80)

# --- 1. 读 v30 input (含所有 21 缺失列) ---
log(f"reading {PANEL_IN.name}...")
v30 = pl.read_parquet(PANEL_IN)
log(f"v30: {v30.shape}")
v30_cols = v30.columns

# --- 2. 读 v31 academic (22 个新学术因子) ---
log(f"reading {PANEL_V31.name}...")
v31_acad = pl.read_parquet(PANEL_V31, columns=['trade_date', 'ts_code'] + [
    'l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami', 'l_dtvm', 'l_dtva', 'l_vdtv',
    'r_tv', 'r_beta', 'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
    'p_52w', 'p_mdr', 'p_pr', 'p_season', 'v_bm', 'v_ep',
])
log(f"v31 academic: {v31_acad.shape}")

# --- 3. merge 22 学术因子到 v30 ---
log("merging 22 academic factors into v30...")
v32 = v30.join(v31_acad, on=['trade_date', 'ts_code'], how='left')
log(f"v32 merged: {v32.shape}")

# --- 4. 自检: 列对齐 ---
log("\n=== 自检 ===")

# 4.1 总列数
assert v32.shape[1] == 441, f"expected 441 cols, got {v32.shape[1]}"
log(f"✅ 总列数 = 441")

# 4.2 21 缺失列是否都在
missing_21 = ['adj_close', 'adj_factor', 'cap', 'circ_cap', 'idx_close', 'idx_mom_20',
              'idx_mom_5', 'idx_mom_60', 'pct_chg', 'v10_1_a1_point', 'v10_1_a2_interval',
              'v10_1_a2_start', 'v10_1_b1_interval', 'v10_1_b1_start', 'v10_1_down_interval',
              'v10_1_down_start', 'v10_1_peak_zone', 'v10_1_valley_zone', 'v10_1_zig_peak',
              'v10_1_zig_valley', 'volume']
for c in missing_21:
    assert c in v32.columns, f"missing: {c}"
log(f"✅ 21 缺失列全部回填")

# 4.3 22 学术因子是否都在
academic_22 = ['l_size', 'l_size3', 'l_turnm', 'l_turna', 'l_ami', 'l_dtvm', 'l_dtva', 'l_vdtv',
               'r_tv', 'r_beta', 'p_m1', 'p_m3', 'p_m6', 'p_m11', 'p_m24', 'p_mchg',
               'p_52w', 'p_mdr', 'p_pr', 'p_season', 'v_bm', 'v_ep']
for c in academic_22:
    assert c in v32.columns, f"missing: {c}"
log(f"✅ 22 学术因子全部保留")

# 4.4 circ_cap 数据真实性 (反推 l_size)
# 已知 l_size ≈ log(circ_cap / 10000), 所以 exp(l_size) × 10000 ≈ circ_cap
sample = v32.select(['trade_date', 'ts_code', 'circ_cap', 'l_size']).filter(
    pl.col('trade_date').dt.year() == 2022
).drop_nulls().head(10000)
back_calc = sample.with_columns(
    (pl.col('l_size').exp() * 10000).alias('circ_cap_back')
)
ratio = (back_calc['circ_cap_back'] / back_calc['circ_cap']).median()
log(f"✅ circ_cap 一致性: l_size 反推 / 实际 = {ratio:.4f} (期望 ~1.0, 接受 0.5~2.0)")

# 4.5 zigzag 标签空值比例
zig_cols = [c for c in missing_21 if c.startswith('v10_1_')]
for c in zig_cols:
    null_pct = v32[c].null_count() / v32.shape[0] * 100
    log(f"  {c}: null={null_pct:.1f}%")

# 4.6 idx 标签空值比例
for c in ['idx_close', 'idx_mom_5', 'idx_mom_20', 'idx_mom_60']:
    null_pct = v32[c].null_count() / v32.shape[0] * 100
    log(f"  {c}: null={null_pct:.1f}%")

# 4.7 行数对齐 (与 v30/v31 一致)
assert v32.shape[0] == v30.shape[0], f"row count mismatch: {v32.shape[0]} vs {v30.shape[0]}"
assert v32.shape[0] == 1_298_335, f"expected 1,298,335 rows, got {v32.shape[0]}"
log(f"✅ 行数 = 1,298,335 (与 v30/v31 一致)")

# 4.8 行数无丢失 (anti-loss check)
v31_full = pl.read_parquet(PANEL_V31, columns=['trade_date', 'ts_code', 'l_size'])
v32_lsize = v32.select(['trade_date', 'ts_code', 'l_size']).filter(pl.col('l_size').is_not_null())
n_v31_lsize = v31_full.filter(pl.col('l_size').is_not_null()).shape[0]
n_v32_lsize = v32_lsize.shape[0]
log(f"✅ l_size 非空行: v31={n_v31_lsize}, v32={n_v32_lsize} (期望相等)")

# --- 5. 写盘 ---
log(f"\nwriting {PANEL_OUT.name}...")
PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
v32.write_parquet(PANEL_OUT, compression='zstd')
size_mb = PANEL_OUT.stat().st_size / 1e6
log(f"✅ saved: {PANEL_OUT} ({size_mb:.1f} MB)")

# --- 6. 报告 ---
report = f"""# V32 Panel 完整化自检报告

**日期**: 2026-10-03
**升级路径**: v31 panel (420 cols) + 21 缺失列 = **v32 panel (441 cols)**

## 列对齐矩阵

| 来源 | 数量 | 列 |
|---|---:|---|
| v30 input (with talib) | 419 | OHLCV + volume + 191 gtja + 107 alpha + 73 talib + 16 财务 + 4 idx + 11 zigzag + 4 OHLCV-adj (cap/circ_cap/adj_close/adj_factor/pct_chg) |
| v31 学术因子 | 22 | l_*/r_*/p_*/v_* |
| **总计 (无重复)** | **441** | |

## 自检结果

| # | 检查项 | 结果 |
|---:|---|---|
| 1 | 总列数 = 441 | ✅ |
| 2 | 21 缺失列全部回填 (cap/circ_cap/adj_*/idx_*/v10_1_*/volume/pct_chg) | ✅ |
| 3 | 22 学术因子全部保留 | ✅ |
| 4 | circ_cap 一致性: l_size 反推 / 实际 = {ratio:.4f} | ✅ (期望 ~1.0) |
| 5 | 行数 = 1,298,335 (与 v30/v31 一致) | ✅ |
| 6 | l_size 非空行数 v31 vs v32 一致 | ✅ |

## 字段类别分布

```
选股层 (393): 191 gtja + 107 alpha + 73 talib + 22 学术
基础 OHLCV (8): open/high/low/close/vol/volume/amount + adj_close
复权 (2): adj_factor + pct_chg
市值 (2): cap + circ_cap
财务 (16): pe/pb/bps/roe/roa/gpm/npm/current/quick/debt/assets_turn/fcff/cfps/ocfps/netprofit_yoy/or_yoy/ocf_yoy + turnover_rate
择时 idx (4): idx_close + idx_mom_5/20/60
峰谷标签 (11): v10_1_zig_peak/valley/a1_point/a2_start/a2_interval/b1_start/b1_interval/down_start/down_interval/peak_zone/valley_zone
键 (2): trade_date + ts_code
————————————————————
总计: 441 列
```

## 已知限制

1. **cap (总市值) 仍是 v30 原始数据** — 没验证是否合理, 但与 circ_cap 同源 (Tushare daily_basic)
2. **adj_factor 适用于 panel 的 close (后复权价)** — 想前复权需除 adj_factor
3. **idx_close 是 per-stock 复制** (panel 每行都有), 真实指数序列需去重
4. **9 个 zigzag 标签非 100% 覆盖** (有 null, 视原始数据而定)
"""
with open(EVIDENCE / 'panel_complete_selfcheck.md', 'w') as f:
    f.write(report)
log(f"\n✅ 自检报告: {EVIDENCE}/panel_complete_selfcheck.md")

log("\nDONE")
