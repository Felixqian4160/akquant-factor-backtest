"""gtja_139 — standalone gtja factor.

GTJA #139 — -1 * CORR(OPEN, VOLUME, 10).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_139 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #139 — -1 * CORR(OPEN, VOLUME, 10)."""
    expr = (-1.0 * corr(pl.col('open'), pl.col('volume'), 10)).alias('gtja_139')
    return panel.select(expr).to_series()
