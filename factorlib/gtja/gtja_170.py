"""gtja_170 — standalone gtja factor.

GTJA #170 — Weighted multi-rank composite.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_170 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #170 — Weighted multi-rank composite."""
    df = panel.with_columns(rank(1.0 / pl.col('close')).alias('__r1'))
    df = df.with_columns(rank(pl.col('high') - pl.col('close')).alias('__r2'))
    a = pl.col('__r1') * pl.col('volume') / mean(pl.col('volume'), 20)
    b = pl.col('high') * pl.col('__r2') / (sum_(pl.col('high'), 5) / 5.0)
    df = df.with_columns((pl.col('vwap') - delay(pl.col('vwap'), 5)).alias('__dvwap5'))
    df = df.with_columns(rank(pl.col('__dvwap5')).alias('__r3'))
    expr = (a * b - pl.col('__r3')).alias('gtja_170')
    return df.select(expr).to_series()
