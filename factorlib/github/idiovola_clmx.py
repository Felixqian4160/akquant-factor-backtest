"""idiovola_clmx — standalone GitHub/academic factor formula.

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
    w = d.with_columns((CLOSE / CLOSE.shift(1).over("stock_code") - 1.0).abs().alias("_absret"))
    w = w.with_columns(pl.col("_absret").shift(1).over("stock_code").alias("_absret_lag"))
    return w.with_columns(pl.col("_absret_lag").rolling_std(60).over("stock_code").alias("idiovola_clmx"))["idiovola_clmx"]
