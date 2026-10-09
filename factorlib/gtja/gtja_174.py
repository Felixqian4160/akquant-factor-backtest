"""gtja_174 — standalone gtja factor.

GTJA #174 — SMA(C>prev_C ? STD(C,20) : 0, 20, 1). Up-day vol EWMA.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_174 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #174 — SMA(C>prev_C ? STD(C,20) : 0, 20, 1). Up-day vol EWMA."""
    pc = delay(pl.col('close'), 1)
    s = std_(pl.col('close'), 20)
    masked = pl.when(pc.is_null() | s.is_null()).then(None).when(pl.col('close') > pc).then(s).otherwise(0.0)
    expr = sma(masked, 20, 1).alias('gtja_174')
    return panel.select(expr).to_series()
