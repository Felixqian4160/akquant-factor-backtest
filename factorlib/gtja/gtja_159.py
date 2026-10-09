"""gtja_159 — standalone gtja factor.

GTJA #159 — Three-window cumulative range-position oscillator.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_159 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #159 — Three-window cumulative range-position oscillator."""
    pc = delay(pl.col('close'), 1)
    p2 = pl.min_horizontal([pl.col('low'), pc])
    p3 = pl.max_horizontal([pl.col('high'), pc])
    p1 = p3 - p2
    df = panel.with_columns([p1.alias('__p1'), p2.alias('__p2'), p3.alias('__p3')])
    a = (pl.col('close') - sum_(pl.col('__p2'), 6)) / sum_(pl.col('__p1'), 6) * 288.0
    b = (pl.col('close') - sum_(pl.col('__p2'), 12)) / sum_(pl.col('__p3') - pl.col('__p2'), 12) * 144.0
    c = (pl.col('close') - sum_(pl.col('__p2'), 24)) / sum_(pl.col('__p1'), 24) * 144.0
    expr = ((a + b + c) * 100.0 / 504.0).alias('gtja_159')
    return df.select(expr).to_series()
