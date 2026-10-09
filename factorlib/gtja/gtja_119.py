"""gtja_119 — standalone gtja factor.

GTJA #119 — Decay-linear of corrs and tsranks, V→Liquidity.

Guotai Junan Formula
--------------------
    RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5),26), 5), 7)) -
    RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)),21), 9), 7), 8))

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_119 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #119 — Decay-linear of corrs and tsranks, V→Liquidity.

    Guotai Junan Formula
    --------------------
        RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5),26), 5), 7)) -
        RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)),21), 9), 7), 8))
    """
    df = panel.with_columns([mean(pl.col('volume'), 5).alias('__mv5'), mean(pl.col('volume'), 15).alias('__mv15')])
    df = df.with_columns([sum_(pl.col('__mv5'), 26).alias('__smv5_26'), rank(pl.col('open')).alias('__ro'), rank(pl.col('__mv15')).alias('__rmv')])
    df = df.with_columns([corr(pl.col('vwap'), pl.col('__smv5_26'), 5).alias('__c1')])
    df = df.with_columns([corr(pl.col('__ro'), pl.col('__rmv'), 21).alias('__c2')])
    df = df.with_columns([ts_min(pl.col('__c2'), 9).alias('__m')])
    df = df.with_columns([ts_rank(pl.col('__m'), 7).alias('__tr')])
    df = df.with_columns([decay_linear(pl.col('__c1'), 7).alias('__dl1'), decay_linear(pl.col('__tr'), 8).alias('__dl2')])
    df = df.with_columns([rank(pl.col('__dl1')).alias('__r1'), rank(pl.col('__dl2')).alias('__r2')])
    expr = (pl.col('__r1') - pl.col('__r2')).alias('gtja_119')
    return df.select(expr).to_series()
