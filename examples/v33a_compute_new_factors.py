"""V33a: 计算 17 个新因子 (slim 列模式 — 规避 4GB worker cgroup 限制)

背景: v33_add_new_factors_v2.py 的 main() 在 background worker scope 中
被 cgroup OOM kill (anon-rss 4.16GB > ~4GB scope MemoryMax), 因为宽表 (441列)
排序 + 中间列过多。本脚本改为:
  - 只加载 11 列 (trade_date/ts_code/open/high/low/close/vol/amount/bps/netprofit_yoy/grossprofit_margin)
  - 计算 17 因子 → 写 evidence/v33_new_factors_20261003/new17_factors.parquet
  - 组装全量 V33 panel 由 v33b_assemble_panel.py 负责 (foreground 无限制)

输出: new17_factors.parquet (2 keys + 17 factors) + 自检报告
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
import v33_add_new_factors_v2 as m

PANEL_IN = Path('data/wavehunter_hs300_v32_complete_20261003.parquet')
EVIDENCE = Path('evidence/v33_new_factors_20261003')
FACTORS_OUT = EVIDENCE / 'new17_factors.parquet'

NEED = ['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
        'bps', 'netprofit_yoy', 'grossprofit_margin']


def log(x):
    print(f"[{time.strftime('%H:%M:%S')}] {x}", flush=True)


log("=" * 80)
log("V33a: 计算 17 个新因子 (slim 模式)")
log("=" * 80)

# 1. slim 加载
log(f"reading {len(NEED)} cols from {PANEL_IN.name}...")
t0 = time.time()
slim = pl.read_parquet(PANEL_IN, columns=NEED)
log(f"  slim: {slim.shape} in {time.time()-t0:.1f}s")

# 2. 排序 + 市场列
slim = slim.sort(['ts_code', 'trade_date'])
slim = m.add_market_cols(slim)

# 3. 17 因子
log("\n=== 计算 17 个新因子 ===")
series_map = {}
t0 = time.time()
for fid, fn in m.NEW_FACTOR_REGISTRY.items():
    t1 = time.time()
    s = fn(slim)
    series_map[fid] = s
    log(f"  {fid:28s} {time.time()-t1:6.1f}s  nulls={s.null_count():,d}")
log(f"total compute: {time.time()-t0:.1f}s")

# 4. 组装 factors frame + sanitize
fac = slim.select(['trade_date', 'ts_code']).with_columns(
    [s.alias(fid) for fid, s in series_map.items()])
fac = fac.with_columns([
    pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None)
    .otherwise(pl.col(c).clip(-1e6, 1e6)).alias(c)
    for c in m.NEW_FACTOR_REGISTRY
])
log(f"\nfac frame: {fac.shape}")

# 5. 自检: null 分布
log("\n=== null 分布 ===")
null_stats = {}
for fid in m.NEW_FACTOR_REGISTRY:
    n = fac[fid].null_count()
    pct = n / fac.height * 100
    null_stats[fid] = pct
    log(f"  {fid:28s}: {n:8,d} ({pct:5.1f}%)")

# 6. 独立 pandas 复算 (000001.SZ 全序列, 最后 200 行, 4 个因子)
log("\n=== pandas 复算对照 (000001.SZ, 最后200行) ===")
import pandas as pd
sub = (slim.filter(pl.col('ts_code') == '000001.SZ').sort('trade_date')
       .select(['trade_date', 'high', 'low', 'close', 'amount']))
facsub = fac.filter(pl.col('ts_code') == '000001.SZ').sort('trade_date')
assert sub.height == facsub.height
one = sub.hstack(facsub.select(
    ['efficiency_ratio', 'alpha191_095', 'hl_52w_disposition', 'winner_ratio'])).to_pandas()

checks = {
    "efficiency_ratio": (one["close"].diff(20).abs()
                         / (one["close"].diff().abs().rolling(20).sum() + 1e-12)).values,
    "alpha191_095": one["amount"].rolling(20).std().values,
    "hl_52w_disposition": ((one["close"] - one["low"].rolling(252).min())
                           / (one["high"].rolling(252).max() - one["low"].rolling(252).min() + 1e-9)).values,
    "winner_ratio": np.array([
        np.mean([(one["close"].iloc[t - i] < one["close"].iloc[t]) for i in range(1, 31)])
        if t >= 30 else np.nan for t in range(len(one))]),
}
verify = {}
for fid, pd_vals in checks.items():
    pl_vals = one[fid].values
    tail = slice(-200, None)
    pd_t, pl_t = pd_vals[tail], pl_vals[tail]
    mask = ~(pd.isna(pd_t) | pd.isna(pl_t))
    if mask.sum() > 0:
        a = pd_t[mask].astype(float); b = pl_t[mask].astype(float)
        max_err = float((np.abs(a - b) / (np.abs(a) + 1e-9)).max())
        verify[fid] = max_err
        log(f"  {fid:22s}: max rel err = {max_err:.2e} (n={int(mask.sum())})")
    else:
        verify[fid] = None
        log(f"  {fid:22s}: no valid overlap")

# 7. 写盘
EVIDENCE.mkdir(parents=True, exist_ok=True)
fac.write_parquet(FACTORS_OUT)
log(f"\n✅ saved: {FACTORS_OUT} ({FACTORS_OUT.stat().st_size/1e6:.1f} MB)")
log(f"   shape: {fac.shape}")

# 8. 报告
report = "# V33a 新因子计算报告\n\n"
report += f"**slim 输入**: {len(NEED)} cols; **输出**: {fac.shape[0]:,} 行 × {fac.shape[1]} 列\n\n"
report += "## null 分布\n\n| 因子 | null% |\n|---|---:|\n"
for fid, pct in null_stats.items():
    report += f"| {fid} | {pct:.1f}% |\n"
report += "\n## pandas 复算对照\n\n| 因子 | max rel err |\n|---|---:|\n"
for fid, err in verify.items():
    report += f"| {fid} | {'N/A' if err is None else f'{err:.2e}'} |\n"
report += """
## 说明

- v33_add_new_factors_v2.py 的 main() 在 background worker scope 被 cgroup OOM kill
  (scope MemoryMax ~4GB; 宽表排序超限)。本脚本 slim 模式重跑, 组装由 v33b 负责。
- 限制: winner_ratio 为 30d percentile 简化版; coskew60/resmom_6m 用等权市场; accruals 用 bps 代理。
"""
(EVIDENCE / 'v33a_compute_report.md').write_text(report)
log("DONE")
