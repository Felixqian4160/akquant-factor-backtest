"""Comprehensive cross-check: all 412 v4 factors vs v34 counterparts.

For each factor: per-date cross-sectional Spearman rank correlation on ~30 sample
dates spread over the common history, then average. Low-agreement factors are
flagged for review.

Output: cross_check_all.csv + console summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE / "factorlib_build"))
import v4_build as vb  # noqa: E402

V4 = BASE / "data" / "wavehunter_hs300_fsdb_v4_20261009_235409.parquet"
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
OUT = BASE / "factorlib_build" / "cross_check_all.csv"

items = vb.master_list()
total = len(items)
print(f"factors to check: {total}")

# sample dates from v4 (Date type)
k4 = pl.read_parquet(V4, columns=["trade_date"]).unique()
k4 = k4.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"))
dates = k4["d"].unique().sort().to_list()
sample_dates = [dates[int(i)] for i in np.linspace(0, len(dates) - 1, 30)]
print("sample dates:", sample_dates[0], "...", sample_dates[-1], f"({len(sample_dates)})")

results = []
for i, (family, module, col) in enumerate(items, 1):
    try:
        s4 = (
            pl.read_parquet(V4, columns=["ts_code", "trade_date", col])
            .with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"))
            .drop("trade_date")
            .filter(pl.col("d").is_in(sample_dates))
        )
        s34 = (
            pl.read_parquet(V34, columns=["ts_code", "trade_date", col])
            .with_columns(pl.col("trade_date").dt.date().alias("d"))
            .drop("trade_date")
            .filter(pl.col("d").is_in(sample_dates))
        )
    except Exception as exc:  # noqa: BLE001
        results.append({"factor": col, "family": family, "mean_rho": None, "n_dates": 0,
                        "note": f"read error: {exc}"[:120]})
        continue

    j = s4.join(s34, on=["ts_code", "d"], how="inner", suffix="_34")
    rhos = []
    for d in sample_dates:
        sub = j.filter(pl.col("d") == d)
        if sub.height < 50:
            continue
        a = sub[col].to_numpy().astype(float)
        b = sub[f"{col}_34"].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50:
            continue
        ra = a[ok].argsort().argsort()
        rb = b[ok].argsort().argsort()
        if ra.std() == 0 or rb.std() == 0:
            continue
        rhos.append(np.corrcoef(ra, rb)[0, 1])
    mean_rho = float(np.mean(rhos)) if rhos else None
    results.append({"factor": col, "family": family, "mean_rho": mean_rho, "n_dates": len(rhos)})
    if i % 50 == 0:
        print(f"  {i}/{total}...", flush=True)

res = pl.DataFrame(results)
res.write_csv(OUT)

valid = res.filter(pl.col("mean_rho").is_not_null())
print()
print("=== distribution of mean per-date rank corr (v4 vs v34) ===")
bins = [
    ("rho >= 0.99", valid.filter(pl.col("mean_rho") >= 0.99).height),
    ("0.95 <= rho < 0.99", valid.filter((pl.col("mean_rho") >= 0.95) & (pl.col("mean_rho") < 0.99)).height),
    ("0.90 <= rho < 0.95", valid.filter((pl.col("mean_rho") >= 0.90) & (pl.col("mean_rho") < 0.95)).height),
    ("0.80 <= rho < 0.90", valid.filter((pl.col("mean_rho") >= 0.80) & (pl.col("mean_rho") < 0.90)).height),
    ("rho < 0.80", valid.filter(pl.col("mean_rho") < 0.80).height),
]
for label, n in bins:
    print(f"  {label:<22} {n}")
n_null = res.height - valid.height
print(f"  (no valid rho: {n_null})")
print(f"  total valid: {valid.height}/{total}")

print()
print("=== factors with rho < 0.90 ===")
low = valid.filter(pl.col("mean_rho") < 0.90).sort("mean_rho")
print(low)
