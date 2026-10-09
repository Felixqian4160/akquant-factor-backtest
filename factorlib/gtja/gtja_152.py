"""gtja_152 — standalone gtja factor.

GTJA #152 — DEA-style triple-EWMA differential.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_152 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #152 — DEA-style triple-EWMA differential."""
    inner = sma(delay(pl.col('close') / delay(pl.col('close'), 9), 1), 9, 1)
    part = delay(inner, 1)
    expr = sma(mean(part, 12) - mean(part, 26), 9, 1).alias('gtja_152')
    return panel.select(expr).to_series()
