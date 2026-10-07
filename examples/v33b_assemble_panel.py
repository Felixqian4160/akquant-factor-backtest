"""V33b: 组装 V33 panel = V32 (441 cols) + 17 新因子 (来自 v33a)

组装方式: left join (maintain_order='left', polars 1.x 左连接保序), 
随后双断言校验行序 + 行数, 防止静默错位。

必须 foreground 运行 (gateway cgroup 无内存限制; background worker scope ~4GB 会 OOM).
峰值内存 ~11GB (full 5GB + out 6GB).

输出: data/wavehunter_hs300_v33_with_new_factors_20261003.parquet (458 cols)
"""
from __future__ import annotations

import time
from pathlib import Path

import polars as pl

PANEL_IN = Path('data/wavehunter_hs300_v32_complete_20261003.parquet')
FACTORS_IN = Path('evidence/v33_new_factors_20261003/new17_factors.parquet')
PANEL_OUT = Path('data/wavehunter_hs300_v33_with_new_factors_20261003.parquet')

NEW17 = ['winner_ratio', 'efficiency_ratio', 'fractal_dimension',
         'alpha191_040', 'alpha191_095', 'mom12m_jt', 'maxret_bcw',
         'accruals_sloan', 'idiovola_clmx', 'gp_novymarx',
         'overnight_intraday_spread', 'skew21_lottery', 'pvcorr_21',
         'kurt21_returns', 'coskew60', 'hl_52w_disposition', 'resmom_6m']


def log(x):
    print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)


log("=" * 80)
log("V33b: 组装 panel (V32 441 + 17 = 458 cols)")
log("=" * 80)

# 1. 读
log(f"reading {PANEL_IN.name}...")
t0 = time.time()
full = pl.read_parquet(PANEL_IN)
log(f"  full: {full.shape} in {time.time()-t0:.1f}s")

log(f"reading {FACTORS_IN.name}...")
fac = pl.read_parquet(FACTORS_IN)
log(f"  fac: {fac.shape}")

# 2. 前提检查
assert fac.height == full.height, f"height mismatch: {fac.height} vs {full.height}"
assert not fac.select(['trade_date', 'ts_code']).is_duplicated().any(), "fac keys duplicated!"
for c in NEW17:
    assert c in fac.columns and c not in full.columns, f"col issue: {c}"

# 3. join (保序)
log("joining (maintain_order='left')...")
out = full.join(fac, on=['trade_date', 'ts_code'], how='left', maintain_order='left')
log(f"  out: {out.shape}")

# 4. 行序 + 行数双断言
assert out.height == full.height, f"row count changed: {out.height} vs {full.height}"
assert out['trade_date'].equals(full['trade_date']), "ROW ORDER BROKEN (trade_date)"
assert out['ts_code'].equals(full['ts_code']), "ROW ORDER BROKEN (ts_code)"
log("✅ 行序 + 行数断言通过")

# 5. 列数
assert out.shape[1] == 441 + 17, f"expected 458 cols, got {out.shape[1]}"
log(f"✅ 列数 = {out.shape[1]}")

# 6. null 传递一致性 (fac nulls == out nulls)
for c in NEW17:
    assert out[c].null_count() == fac[c].null_count(), f"null mismatch on {c}"
    assert out[c].null_count() == fac[c].null_count(), f"null mismatch {c}"
log("✅ 17 因子 null 传递一致")

# 7. 与 V32 原始数据一致性抽查 (3 列 x 3 行)
chk = full.select(['close', 'l_size', 'cap']).head(3).to_dicts()
chk2 = out.select(['close', 'l_size', 'cap']).head(3).to_dicts()
assert chk == chk2, "original column values changed!"
log("✅ 原始列值未变 (抽样)")

# 8. 写盘
log(f"writing {PANEL_OUT.name}...")
t0 = time.time()
out.write_parquet(PANEL_OUT, compression='zstd')
log(f"✅ saved: {PANEL_OUT} ({PANEL_OUT.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")

# 9. 落盘后复检
schema = pl.scan_parquet(PANEL_OUT).collect_schema().names()
assert len(schema) == 458, f"reload col count: {len(schema)}"
assert all(c in schema for c in NEW17)
log(f"✅ 复检: 落盘 458 列, 17 新因子存在")

log("DONE")
