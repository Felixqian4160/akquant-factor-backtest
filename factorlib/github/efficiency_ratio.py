"""efficiency_ratio — standalone GitHub/academic factor formula.

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
    num = CLOSE.diff(20).abs().over("stock_code")
    den = CLOSE.diff(1).abs().rolling_sum(20).over("stock_code")
    return d.select((num / (den + 1e-12)).alias("efficiency_ratio"))["efficiency_ratio"]
