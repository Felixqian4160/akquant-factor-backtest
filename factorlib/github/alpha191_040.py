"""alpha191_040 — standalone GitHub/academic factor formula.

Canonical source: /media/felix/f/quant/akquant-factor-backtest/examples/v34_build_part1_factors.py
The function body is copied from the canonical v34 formula source and adapted
only for the factorlib panel key ``stock_code``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from factorlib._ops.github_ops import prepare as _prepare

CLOSE = pl.col("close")

def compute(panel):
    d = _prepare(panel)
    w = d.with_columns([
        (pl.col("vol") * (CLOSE > CLOSE.shift(1).over("stock_code")).cast(pl.Float64)).alias("_up"),
        (pl.col("vol") * (CLOSE <= CLOSE.shift(1).over("stock_code")).cast(pl.Float64)).alias("_dn"),
    ])
    w = w.with_columns([
        pl.col("_up").rolling_sum(26).over("stock_code").alias("_ups"),
        pl.col("_dn").rolling_sum(26).over("stock_code").alias("_dns"),
    ])
    return w.with_columns((pl.col("_ups") / (pl.col("_dns") + 1e-9)).alias("alpha191_040"))["alpha191_040"]
