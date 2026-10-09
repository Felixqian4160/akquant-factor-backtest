"""gtja_171 — standalone gtja factor.

GTJA #171 — -1 * (L-C) * O^5 / ((C-H)*C^5).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_171 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #171 — -1 * (L-C) * O^5 / ((C-H)*C^5)."""
    expr = (-1.0 * (pl.col('low') - pl.col('close')) * pl.col('open').pow(5) / ((pl.col('close') - pl.col('high') + 1e-07) * pl.col('close').pow(5))).alias('gtja_171')
    return panel.select(expr).to_series()
