"""gtja_109 — standalone gtja factor.

GTJA #109 — SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2).

Range-relative trend oscillator.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_109 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #109 — SMA(H-L,10,2) / SMA(SMA(H-L,10,2),10,2).

    Range-relative trend oscillator.
    """
    hl = pl.col('high') - pl.col('low')
    inner = sma(hl, 10, 2)
    expr = (inner / sma(inner, 10, 2)).alias('gtja_109')
    return panel.select(expr).to_series()
