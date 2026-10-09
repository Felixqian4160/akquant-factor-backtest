"""gtja_163 — standalone gtja factor.

GTJA #163 — RANK(-RET * MEAN(V,20) * VWAP * (HIGH - CLOSE)).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_163 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #163 — RANK(-RET * MEAN(V,20) * VWAP * (HIGH - CLOSE))."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    inner = -1.0 * ret * mean(pl.col('volume'), 20) * pl.col('vwap') * (pl.col('high') - pl.col('close'))
    df = panel.with_columns(inner.alias('__i'))
    return df.select(rank(pl.col('__i')).alias('gtja_163')).to_series()
