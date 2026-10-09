"""gtja_112 — standalone gtja factor.

GTJA #112 — Up-vs-down 12-day cumulative move ratio (CMO-style).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_112 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #112 — Up-vs-down 12-day cumulative move ratio (CMO-style)."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    up = pl.when(dc > 0).then(dc).otherwise(0.0)
    dn = pl.when(dc < 0).then(-dc).otherwise(0.0)
    sup = sum_(up, 12)
    sdn = sum_(dn, 12)
    expr = ((sup - sdn) / (sup + sdn) * 100.0).alias('gtja_112')
    return panel.select(expr).to_series()
