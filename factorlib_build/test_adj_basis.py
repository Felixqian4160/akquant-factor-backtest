"""Empirical test: which fsdb adjusted series is economically correct?

A correct adjusted series removes ex-dividend jumps: its daily returns at
corporate-action dates stay small while raw returns spike. An inversely
adjusted series would AMPLIFY jumps. Compare return dispersion at events.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"
EVENTS = Path(
    "/home/felix/.hermes/profiles/buffett/cache/scratch/fsdb_eval/fsdb_panel_build/"
    "stockdb_adj/hs300_adj_events.parquet"
)

CODE = "000001.SZ"

df = (
    pl.read_parquet(
        FSDB,
        columns=["ts_code", "trade_date", "close", "close_hfq", "close_qfq",
                 "adj_factor_hfq", "cum_t", "cum_latest"],
    )
    .filter(pl.col("ts_code") == CODE)
    .sort("trade_date")
)
print("rows:", df.height, df["trade_date"].min(), "->", df["trade_date"].max())

df = df.with_columns([
    (pl.col("close") / pl.col("close").shift(1) - 1).alias("ret_raw"),
    (pl.col("close_hfq") / pl.col("close_hfq").shift(1) - 1).alias("ret_hfq"),
    (pl.col("close_qfq") / pl.col("close_qfq").shift(1) - 1).alias("ret_qfq"),
])

print()
print("=== daily return std (2015+) ===")
d15 = df.filter(pl.col("trade_date") >= "2015-01-01")
print(d15.select([
    pl.col("ret_raw").std().alias("std_raw"),
    pl.col("ret_hfq").std().alias("std_hfq"),
    pl.col("ret_qfq").std().alias("std_qfq"),
]))

print()
print("=== biggest raw drops and their returns in each series ===")
big = df.filter(pl.col("ret_raw").abs() > 0.20).sort(pl.col("ret_raw").abs(), descending=True)
print("count of |raw ret| > 20%:", big.height)
print(big.select(["trade_date", "close", "ret_raw", "ret_hfq", "ret_qfq"]).head(15))

print()
print("=== events file sample for this code ===")
ev = pl.read_parquet(EVENTS)
print("columns:", ev.columns)
sample = ev.filter(pl.col("code") == CODE.replace(".SZ", "")) if "code" in ev.columns else ev.head(3)
if sample.height == 0:
    # try alternative key naming
    for cand in ("ts_code", "symbol", "stock_code"):
        if cand in ev.columns:
            sample = ev.filter(pl.col(cand) == CODE)
            break
print(sample.sort("date").tail(20) if "date" in sample.columns else sample.head(20))
