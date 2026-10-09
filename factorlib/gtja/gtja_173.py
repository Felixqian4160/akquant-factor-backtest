"""gtja_173 — standalone gtja factor.

GTJA #173 — 3*SMA(C,13,2) - 2*SMA^2(C,13,2) + SMA^3(LOG(C),13,2).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_173 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #173 — 3*SMA(C,13,2) - 2*SMA^2(C,13,2) + SMA^3(LOG(C),13,2)."""
    ma = sma(pl.col('close'), 13, 2)
    expr = (3.0 * ma - 2.0 * sma(ma, 13, 2) + sma(sma(log_(pl.col('close')), 13, 2), 13, 2)).alias('gtja_173')
    return panel.select(expr).to_series()
