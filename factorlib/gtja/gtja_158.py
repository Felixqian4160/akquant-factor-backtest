"""gtja_158 — standalone gtja factor.

GTJA #158 — ((H - SMA(C,15,2)) - (L - SMA(C,15,2))) / C — high-low spread / close.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_158 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #158 — ((H - SMA(C,15,2)) - (L - SMA(C,15,2))) / C — high-low spread / close."""
    s = sma(pl.col('close'), 15, 2)
    expr = ((pl.col('high') - s - (pl.col('low') - s)) / pl.col('close')).alias('gtja_158')
    return panel.select(expr).to_series()
