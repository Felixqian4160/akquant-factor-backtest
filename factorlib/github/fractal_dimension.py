"""fractal_dimension — standalone GitHub/academic factor formula.

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
    w = d.with_columns((pl.col("high").rolling_max(20).over("stock_code") - pl.col("low").rolling_min(20).over("stock_code")).alias("_h_range"))
    w = w.with_columns(((1.0/20)**2 + (CLOSE.diff(1).over("stock_code") / (pl.col("_h_range") + 1e-9))**2).sqrt().alias("_leg"))
    w = w.with_columns(pl.col("_leg").rolling_sum(20).over("stock_code").alias("_L"))
    w = w.with_columns((1 + (pl.col("_L").log() + np.log(2)) / np.log(40.0)).alias("fractal_dimension"))
    return w["fractal_dimension"]
