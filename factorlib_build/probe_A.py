"""Probe set for A (recompute):
1. Which adjusted series is continuous at dividend events (hfq vs qfq)?
2. Fidelity runtimes (calibration for chunking).
3. Disk space.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")

print("=== fidelity runtimes ===")
prog = BASE / "factorlib_build" / "fidelity_progress.jsonl"
recs = [json.loads(l) for l in prog.read_text().splitlines() if l.strip()]
times = sorted((r.get("runtime_s", -1), r["module"]) for r in recs)
for t, m in times:
    print(f"  {m:<28} {t}s")
print(f"  TOTAL: {sum(t for t, _ in times if t > 0):.1f}s for {len(times)} factors")

print()
print("=== dividend continuity test: 000001.SZ ===")
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"
df = (
    pl.read_parquet(FSDB, columns=["ts_code", "trade_date", "close", "close_hfq", "close_qfq"])
    .filter(pl.col("ts_code") == "000001.SZ")
    .sort("trade_date")
    .with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_raw"),
        (pl.col("close_hfq") / pl.col("close_hfq").shift(1) - 1).alias("ret_hfq"),
        (pl.col("close_qfq") / pl.col("close_qfq").shift(1) - 1).alias("ret_qfq"),
    ])
)
d15 = df.filter(pl.col("trade_date") >= "2015-01-01")
stats = d15.select([
    pl.col("ret_raw").std().alias("std_raw"),
    pl.col("ret_hfq").std().alias("std_hfq"),
    pl.col("ret_qfq").std().alias("std_qfq"),
]).row(0, named=True)
print("2015+ return std:")
print(f"  raw={stats['std_raw']:.5f}  hfq={stats['std_hfq']:.5f}  qfq={stats['std_qfq']:.5f}")

# count extreme returns
for col in ("ret_raw", "ret_hfq", "ret_qfq"):
    n = d15.filter(pl.col(col).abs() > 0.10).height
    print(f"  |{col}| > 10%: {n} days")

# biggest raw DROPS (dividend gaps are drops)
print()
print("top raw drops (likely ex-div) and counterparts:")
big = df.filter(pl.col("ret_raw") < -0.05).sort("ret_raw").head(10)
print(big.select(["trade_date", "close", "ret_raw", "ret_hfq", "ret_qfq"]))
