"""gtja_177 — standalone gtja factor.

GTJA #177 — (20 - HIGHDAY(HIGH,20)) / 20 * 100. Recency-of-high.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_177 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #177 — (20 - HIGHDAY(HIGH,20)) / 20 * 100. Recency-of-high."""
    expr = ((20.0 - highday(pl.col('high'), 20)) / 20.0 * 100.0).alias('gtja_177')
    return panel.select(expr).to_series()
