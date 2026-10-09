"""Post-build audit for fsdb v4:
1. gtja_gtja_030 investigate (const flag)
2. Cross-check v4 recomputed factors vs v34 original factors on common stock-dates
   (rank correlation in cross-section) for a representative set.
3. Null-rate summary per family.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
V4 = BASE / "data" / "wavehunter_hs300_fsdb_v4_20261009_234301.parquet"
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
CACHE = BASE / "factorlib_build" / "_v4_cache"

print("=== 1. gtja_gtja_030 ===")
g30 = pl.read_parquet(V4, columns=["ts_code", "trade_date", "gtja_gtja_030"])
print("nulls:", g30["gtja_gtja_030"].null_count(), "/", g30.height)
print("unique:", g30["gtja_gtja_030"].n_unique())
print("sample non-null:", g30.filter(pl.col("gtja_gtja_030").is_not_null()).head(5))

print()
print("=== 2. cross-check vs v34 (rank corr on common dates) ===")
CHECK_FACTORS = [
    "alpha_alpha001", "alpha_alpha002", "gtja_gtja_001", "gtja_gtja_010",
    "talib_RSI", "talib_MACD_0", "talib_SMA",
    "l_ami", "r_tv", "p_m6", "efficiency_ratio", "mom12m_jt",
]
# common dates: v34 ends 2026-08-27, fsdb ends 2026-09-30; use 2024 dates
v4 = pl.read_parquet(V4, columns=["ts_code", "trade_date"] + CHECK_FACTORS)
v34 = pl.read_parquet(V34, columns=["ts_code", "trade_date"] + CHECK_FACTORS)
# normalize date types: v4 string 'YYYYMMDD', v34 datetime
v4 = v4.with_columns(
    pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d")
).drop("trade_date")
v34 = v34.with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")

joined = v4.join(v34, on=["ts_code", "d"], how="inner", suffix="_34")
print("joined rows:", joined.height)

# sample a few dates
dates = joined["d"].unique().sort().to_list()
sample_dates = [dates[i] for i in np.linspace(0, len(dates) - 1, 30).astype(int)]
sub = joined.filter(pl.col("d").is_in(sample_dates))

rows = []
for fac in CHECK_FACTORS:
    a = sub[fac].to_numpy().astype(float)
    b = sub[f"{fac}_34"].to_numpy().astype(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 1000:
        rows.append((fac, float("nan"), int(ok.sum())))
        continue
    ra = a[ok].argsort().argsort()
    rb = b[ok].argsort().argsort()
    cc = float(np.corrcoef(ra, rb)[0, 1])
    rows.append((fac, cc, int(ok.sum())))

print(f"{'factor':<34} {'rank_corr':>10} {'n':>10}")
for fac, cc, n in rows:
    print(f"{fac:<34} {cc:>10.4f} {n:>10}")

print()
print("=== 3. null-rate summary per family ===")
audit = pl.read_csv(CACHE / "v4_factor_audit.csv")
audit = audit.with_columns(
    pl.when(pl.col("factor").str.starts_with("alpha_")).then(pl.lit("alpha"))
    .when(pl.col("factor").str.starts_with("gtja_")).then(pl.lit("gtja"))
    .when(pl.col("factor").str.starts_with("talib_")).then(pl.lit("talib"))
    .otherwise(pl.lit("academic/github")).alias("family")
)
summary = audit.group_by("family").agg([
    pl.len().alias("n"),
    pl.col("null_rate").mean().round(4).alias("mean_null_rate"),
    pl.col("null_rate").max().round(4).alias("max_null_rate"),
]).sort("family")
print(summary)
print()
print("factors with null_rate > 0.2:")
print(audit.filter(pl.col("null_rate") > 0.2).select(["factor", "null_rate"]).head(30))
