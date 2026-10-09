"""gtja_175 — standalone gtja factor.

GTJA #175 — 6-day mean true range.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_175 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #175 — 6-day mean true range."""
    pc = delay(pl.col('close'), 1)
    a = pl.col('high') - pl.col('low')
    b = (pc - pl.col('high')).abs()
    c = (pc - pl.col('low')).abs()
    tr = pl.max_horizontal([pl.max_horizontal([a, b]), c])
    expr = mean(tr, 6).alias('gtja_175')
    return panel.select(expr).to_series()
