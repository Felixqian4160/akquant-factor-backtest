"""gtja_179 — standalone gtja factor.

GTJA #179 — RANK(CORR(VWAP,V,4)) * RANK(CORR(RANK(LOW), RANK(MEAN(V,50)), 12)).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_179 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #179 — RANK(CORR(VWAP,V,4)) * RANK(CORR(RANK(LOW), RANK(MEAN(V,50)), 12))."""
    df = panel.with_columns([corr(pl.col('vwap'), pl.col('volume'), 4).alias('__c1'), mean(pl.col('volume'), 50).alias('__mv50'), rank(pl.col('low')).alias('__rl')])
    df = df.with_columns(rank(pl.col('__mv50')).alias('__rmv'))
    df = df.with_columns(corr(pl.col('__rl'), pl.col('__rmv'), 12).alias('__c2'))
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    return df.select((pl.col('__r1') * pl.col('__r2')).alias('gtja_179')).to_series()
