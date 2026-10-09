"""gtja_189 — standalone gtja factor.

GTJA #189 — MEAN(|C - MEAN(C,6)|, 6). Mean abs deviation from MA6.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_189 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #189 — MEAN(|C - MEAN(C,6)|, 6). Mean abs deviation from MA6."""
    expr = mean((pl.col('close') - mean(pl.col('close'), 6)).abs(), 6).alias('gtja_189')
    return panel.select(expr).to_series()
