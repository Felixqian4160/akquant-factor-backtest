"""gtja_106 — standalone gtja factor.

GTJA #106 — CLOSE - DELAY(CLOSE, 20). Pure 20-day momentum.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_106 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #106 — CLOSE - DELAY(CLOSE, 20). Pure 20-day momentum."""
    expr = (pl.col('close') - delay(pl.col('close'), 20)).alias('gtja_106')
    return panel.select(expr).to_series()
