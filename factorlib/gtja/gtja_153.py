"""gtja_153 — standalone gtja factor.

GTJA #153 — (MA3+MA6+MA12+MA24)/4 — multi-MA average.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_153 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #153 — (MA3+MA6+MA12+MA24)/4 — multi-MA average."""
    expr = ((mean(pl.col('close'), 3) + mean(pl.col('close'), 6) + mean(pl.col('close'), 12) + mean(pl.col('close'), 24)) / 4.0).alias('gtja_153')
    return panel.select(expr).to_series()
