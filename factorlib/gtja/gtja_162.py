"""gtja_162 — standalone gtja factor.

GTJA #162 — RSI-style normalised: (RSI - min(RSI,12)) / (max(RSI,12) - min(RSI,12)).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_162 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #162 — RSI-style normalised: (RSI - min(RSI,12)) / (max(RSI,12) - min(RSI,12))."""
    dc = pl.col('close') - delay(pl.col('close'), 1)
    p2 = sma(pl.when(dc > 0).then(dc).otherwise(0.0), 12, 1)
    p3 = sma(dc.abs(), 12, 1)
    rsi = p2 / p3 * 100.0
    df = panel.with_columns(rsi.alias('__rsi'))
    df = df.with_columns([ts_min(pl.col('__rsi'), 12).alias('__rmin'), ts_max(pl.col('__rsi'), 12).alias('__rmax')])
    expr = ((pl.col('__rsi') - pl.col('__rmin')) / (pl.col('__rmax') - pl.col('__rmin'))).alias('gtja_162')
    return df.select(expr).to_series()
