"""gtja_141 — standalone gtja factor.

GTJA #141 — RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) * -1.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_141 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #141 — RANK(CORR(RANK(HIGH), RANK(MEAN(VOLUME,15)), 9)) * -1."""
    df = panel.with_columns([mean(pl.col('volume'), 15).alias('__mv15'), rank(pl.col('high')).alias('__rh')])
    df = df.with_columns(rank(pl.col('__mv15')).alias('__rmv'))
    df = df.with_columns(corr(pl.col('__rh'), pl.col('__rmv'), 9).alias('__c'))
    df = df.with_columns(rank(pl.col('__c')).alias('__r'))
    return df.select((pl.col('__r') * -1.0).alias('gtja_141')).to_series()
