"""gtja_188 — standalone gtja factor.

GTJA #188 — ((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100. Range deviation.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_188 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #188 — ((H-L) - SMA(H-L,11,2)) / SMA(H-L,11,2) * 100. Range deviation."""
    rng = pl.col('high') - pl.col('low')
    s = sma(rng, 11, 2)
    expr = ((rng - s) / s * 100.0).alias('gtja_188')
    return panel.select(expr).to_series()
