"""hl_52w_disposition — standalone GitHub/academic factor formula.

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
    return d.select(((CLOSE - pl.col("low").rolling_min(252).over("stock_code"))
                     / (pl.col("high").rolling_max(252).over("stock_code") - pl.col("low").rolling_min(252).over("stock_code") + 1e-9)).alias("hl_52w_disposition"))["hl_52w_disposition"]
