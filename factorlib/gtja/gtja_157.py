"""gtja_157 — standalone gtja factor.

GTJA #157 — TS_MIN of triple-rank-log + tsrank of delay(-ret,6).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_157 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #157 — TS_MIN of triple-rank-log + tsrank of delay(-ret,6)."""
    df = panel.with_columns([delta(pl.col('close') - 1.0, 5).alias('__d'), (pl.col('close') / delay(pl.col('close'), 1) - 1.0).alias('__ret')])
    df = df.with_columns([rank(-1.0 * rank(pl.col('__d'))).alias('__rd')])
    df = df.with_columns(rank(pl.col('__rd')).alias('__rrd'))
    df = df.with_columns(ts_min(pl.col('__rrd'), 2).alias('__tm'))
    df = df.with_columns(sum_(pl.col('__tm'), 1).alias('__sum'))
    df = df.with_columns(rank(rank(log_(pl.col('__sum')))).alias('__lhs'))
    df = df.with_columns([ts_min(pl.col('__lhs'), 5).alias('__lhs_min'), ts_rank(delay(-1.0 * pl.col('__ret'), 6), 5).alias('__rhs')])
    expr = (pl.col('__lhs_min') + pl.col('__rhs')).alias('gtja_157')
    return df.select(expr).to_series()
