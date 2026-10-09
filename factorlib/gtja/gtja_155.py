"""gtja_155 — standalone gtja factor.

GTJA #155 — MACD-style on volume.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_155 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #155 — MACD-style on volume."""
    macd = sma(pl.col('volume'), 13, 2) - sma(pl.col('volume'), 27, 2)
    signal = sma(macd, 10, 2)
    expr = (macd - signal).alias('gtja_155')
    return panel.select(expr).to_series()
