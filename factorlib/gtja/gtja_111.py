"""gtja_111 — standalone gtja factor.

GTJA #111 — VOL * intra-day position SMA differential.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_111 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #111 — VOL * intra-day position SMA differential."""
    rng = pl.col('high') - pl.col('low')
    pos = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / rng
    weighted = pl.col('volume') * pos
    expr = (sma(weighted, 11, 2) - sma(weighted, 4, 2)).alias('gtja_111')
    return panel.select(expr).to_series()
