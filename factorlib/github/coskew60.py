"""coskew60 — standalone GitHub/academic factor formula.

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
        (CLOSE / CLOSE.shift(1).over("stock_code") - 1.0).alias("_ret"),
        (pl.col("_mkt_ret") ** 2).alias("_m2"),
    ])
    w = w.with_columns((pl.col("_ret") * pl.col("_m2")).alias("_rm2"))
    w = w.with_columns([
        pl.col("_rm2").rolling_mean(60).over("stock_code").alias("_erm2"),
        pl.col("_m2").rolling_mean(60).over("stock_code").alias("_em2"),
    ])
    return w.with_columns((pl.col("_erm2") / (pl.col("_em2") + 1e-9)).alias("coskew60"))["coskew60"]
