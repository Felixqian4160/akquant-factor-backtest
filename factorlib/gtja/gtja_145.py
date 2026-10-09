"""gtja_145 — standalone gtja factor.

GTJA #145 — (MEAN(V,9) - MEAN(V,26)) / MEAN(V,12) * 100.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_145 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #145 — (MEAN(V,9) - MEAN(V,26)) / MEAN(V,12) * 100."""
    expr = ((mean(pl.col('volume'), 9) - mean(pl.col('volume'), 26)) / mean(pl.col('volume'), 12) * 100.0).alias('gtja_145')
    return panel.select(expr).to_series()
