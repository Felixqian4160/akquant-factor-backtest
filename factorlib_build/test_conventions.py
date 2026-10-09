"""Empirical test: listing-anchored (B) vs latest-anchored (A) price series for v4.

v34 factors were computed on raw * adj_factor_tushare (listing-anchored).
fsdb *_hfq columns are latest-anchored (cum_t/cum_latest, = raw at the newest date).
Level-dependent factors' cross-sectional ranks can differ between conventions.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(BASE))

V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"

# ---- 0. unit ratios ----
v34u = pl.read_parquet(V34, columns=["ts_code", "trade_date", "vol", "amount"])
fsu = pl.read_parquet(FSDB, columns=["ts_code", "trade_date", "volume", "amount", "cum_latest"])
v34u = v34u.with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")
fsu = fsu.with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d")).drop("trade_date")
j = v34u.join(fsu, on=["ts_code", "d"], how="inner", suffix="_f").filter(
    (pl.col("d") >= pl.date(2015, 1, 1)) & (pl.col("d") <= pl.date(2024, 12, 31))
)
print("=== units ===")
print("vol/volume ratio median:", j.select((pl.col("vol") / pl.col("volume")).median()).item())
print("amount/amount_f ratio median:", j.select((pl.col("amount") / pl.col("amount_f")).median()).item())

per_stock = fsu.group_by("ts_code").agg(pl.col("cum_latest").n_unique().alias("n"))
print("cum_latest unique-per-stock max:", per_stock["n"].max())


def build_frame(anchor: str) -> pl.DataFrame:
    f = (
        pl.read_parquet(
            FSDB,
            columns=["ts_code", "trade_date", "open_hfq", "high_hfq", "low_hfq", "close_hfq",
                     "adj_factor_hfq", "cum_latest", "volume", "amount", "pb", "close",
                     "float_mv", "total_mv"],
        )
        .with_columns(pl.col("trade_date").str.strptime(pl.Date, "%Y%m%d").alias("d"))
        .drop("trade_date")
        .sort(["ts_code", "d"])
    )
    mult = pl.lit(1.0) if anchor == "latest" else pl.col("cum_latest")
    frame = f.select([
        pl.col("ts_code").alias("stock_code"),
        pl.col("d").alias("trade_date"),
        (pl.col("open_hfq") * mult).alias("open"),
        (pl.col("high_hfq") * mult).alias("high"),
        (pl.col("low_hfq") * mult).alias("low"),
        (pl.col("close_hfq") * mult).alias("close"),
        (pl.col("volume") / 100.0).alias("volume"),
        (pl.col("volume") / 100.0).alias("vol"),
        (pl.col("amount") / 1000.0).alias("amount"),
        (((pl.col("amount") / 1000.0) * 10.0 / (pl.col("volume") / 100.0).clip(lower_bound=1))
         * pl.col("adj_factor_hfq") * mult).alias("vwap"),
        # bps = book value per share, anchored on the same scale as `close`:
        (pl.col("close_hfq") / pl.col("pb") * mult).alias("bps"),
        (pl.col("float_mv") / 1e4).alias("circ_cap"),
        (pl.col("total_mv") / 1e4).alias("cap"),
    ])
    frame = frame.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1.0).fill_null(0.0).alias("returns")
    )
    for w in (5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180):
        frame = frame.with_columns(
            pl.col("volume").rolling_mean(w, min_samples=1).over("stock_code").alias(f"adv{w}")
        )
    return frame


FRAMES = {"A_latest": build_frame("latest"), "B_listing": build_frame("listing")}

v34s = pl.read_parquet(
    V34, columns=["ts_code", "trade_date", "talib_SMA", "talib_MACD_0", "gtja_gtja_010", "v_bm", "alpha_alpha001"]
).with_columns(pl.col("trade_date").dt.date().alias("trade_date"))

CASES = [
    ("talib", "talib_SMA", "talib_SMA"),
    ("talib", "talib_MACD_0", "talib_MACD_0"),
    ("gtja", "gtja_010", "gtja_gtja_010"),
    ("academic", "v_bm", "v_bm"),
    ("alpha", "alpha001", "alpha_alpha001"),
]

for cand, frame in FRAMES.items():
    print()
    print(f"=== candidate {cand} ===")
    for family, module, col34 in CASES:
        mod = importlib.import_module(f"factorlib.{family}.{module}")
        vals = mod.compute(frame)
        fr = frame.select(["stock_code", "trade_date"]).with_columns(vals.alias("v"))
        jj = fr.rename({"stock_code": "ts_code"}).join(
            v34s.select(["ts_code", "trade_date", col34]),
            on=["ts_code", "trade_date"], how="inner",
        ).filter(pl.col("trade_date") >= pl.date(2020, 1, 1))
        a = jj["v"].to_numpy().astype(float)
        b = jj[col34].to_numpy().astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 1000:
            print(f"  {col34:<24} insufficient n={int(ok.sum())}")
            continue
        ra = a[ok].argsort().argsort()
        rb = b[ok].argsort().argsort()
        print(f"  {col34:<24} rank_corr={np.corrcoef(ra, rb)[0,1]:.4f} n={int(ok.sum())}")
