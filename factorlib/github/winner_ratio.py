"""winner_ratio — standalone GitHub/academic factor formula.

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
    parts = [((CLOSE.shift(i).over("stock_code")) < CLOSE).cast(pl.Float64).fill_null(0.0) for i in range(1, 31)]
    return d.select(pl.when(CLOSE.is_null()).then(None).otherwise(pl.sum_horizontal(parts) / 30.0).alias("winner_ratio"))["winner_ratio"]
