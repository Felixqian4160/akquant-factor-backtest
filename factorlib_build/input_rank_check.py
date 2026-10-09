"""Cross-source input-level agreement check: fsdb v4 vs Tushare v34.

For matched (stock, date) rows, compute the per-date cross-sectional Spearman
rank correlation between the two sources for adjusted close / volume / amount,
then report the median across dates. This quantifies how much of the residual
factor divergence (55 of 405 comparable factors <0.90 rank corr) is explained
by raw input differences between the stockdb and Tushare data sources.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")

V4 = BASE / "data" / "wavehunter_hs300_fsdb_v4_20261009_235409.parquet"
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"

a = pl.read_parquet(
    V4, columns=["ts_code", "trade_date", "close_hfq", "cum_latest", "volume", "amount"]
).select([
    pl.col("ts_code").alias("code"),
    pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"),
    (pl.col("close_hfq") * pl.col("cum_latest")).alias("adj_close"),
    (pl.col("volume") / 100.0).alias("vol_lots"),
    (pl.col("amount") / 1000.0).alias("amount_k"),
])
print("v4 side:", a.shape)

b = pl.read_parquet(
    V34, columns=["ts_code", "trade_date", "adj_close", "vol", "amount"]
).select([
    pl.col("ts_code").alias("code"),
    pl.col("trade_date").dt.date().alias("d"),
    pl.col("adj_close").alias("adj_close_34"),
    pl.col("vol").alias("vol_34"),
    pl.col("amount").alias("amt_34"),
])
print("v34 side:", b.shape)

j = a.join(b, on=["code", "d"], how="inner")
print("joined:", j.shape)
print()

for x, y in (("adj_close", "adj_close_34"), ("vol_lots", "vol_34"), ("amount_k", "amt_34")):
    g = (
        j.filter(pl.col(x).is_not_null() & pl.col(y).is_not_null())
        .group_by("d")
        .agg(pl.corr(pl.col(x).rank(), pl.col(y).rank()).alias("rho"))
        .drop_nulls()
    )
    med = g["rho"].median()
    med_pearson = (
        j.filter(pl.col(x).is_not_null() & pl.col(y).is_not_null())
        .group_by("d")
        .agg(pl.corr(pl.col(x), pl.col(y)).alias("r"))
        .drop_nulls()["r"].median()
    )
    rr = (
        j.filter((pl.col(y) > 0) & pl.col(x).is_not_null())
        .select((pl.col(x) / pl.col(y)).alias("ratio"))
        .drop_nulls()["ratio"]
    )
    print(
        f"{x:<10} rank-rho median={med:.4f}  pearson median={med_pearson:.4f}  "
        f"ratio med={rr.median():.4f} p05={rr.quantile(0.05):.4f} p95={rr.quantile(0.95):.4f}  "
        f"n_dates={g.height}"
    )
