"""gtja_146 — standalone gtja factor.

GTJA #146 — Daic115 variant: mean(part_ma,20) * part_ma / SMA(part-part_ma)^2.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_146 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #146 — Daic115 variant: mean(part_ma,20) * part_ma / SMA(part-part_ma)^2."""
    pc = delay(pl.col('close'), 1)
    part = (pl.col('close') - pc) / pc
    df = panel.with_columns(part.alias('__p'))
    df = df.with_columns(sma(pl.col('__p'), 61, 2).alias('__sp'))
    df = df.with_columns((pl.col('__p') - pl.col('__sp')).alias('__pm'))
    df = df.with_columns([mean(pl.col('__pm'), 20).alias('__m'), sma((pl.col('__p') - pl.col('__pm')).pow(2), 61, 2).alias('__s')])
    expr = (pl.col('__m') * pl.col('__pm') / pl.col('__s')).alias('gtja_146')
    return df.select(expr).to_series()
