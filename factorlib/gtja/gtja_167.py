"""gtja_167 — standalone gtja factor.

GTJA #167 — SUM(MAX(C-prev_C,0), 12). 12-day cumulative up-move.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_167 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #167 — SUM(MAX(C-prev_C,0), 12). 12-day cumulative up-move."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    expr = sum_(pl.when(dc > 0).then(dc).otherwise(0.0), 12).alias('gtja_167')
    return panel.select(expr).to_series()
