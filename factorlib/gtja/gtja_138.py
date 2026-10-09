"""gtja_138 — standalone gtja factor.

GTJA #138 — Decay-linear delta of (0.7L+0.3VWAP) minus tsrank-cascade.

Guotai Junan Formula
--------------------
    (RANK(DECAYLINEAR(DELTA(0.7L + 0.3VWAP, 3), 20)) -
     TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8),
                                    TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_138 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #138 — Decay-linear delta of (0.7L+0.3VWAP) minus tsrank-cascade.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(0.7L + 0.3VWAP, 3), 20)) -
         TSRANK(DECAYLINEAR(TSRANK(CORR(TSRANK(LOW, 8),
                                        TSRANK(MEAN(VOLUME,60), 17), 5), 19), 16), 7)) * -1
    """
    df = panel.with_columns([delta(pl.col('low') * 0.7 + pl.col('vwap') * 0.3, 3).alias('__d'), ts_rank(pl.col('low'), 8).alias('__tl'), ts_rank(mean(pl.col('volume'), 60), 17).alias('__tv')])
    df = df.with_columns(corr(pl.col('__tl'), pl.col('__tv'), 5).alias('__c'))
    df = df.with_columns(ts_rank(pl.col('__c'), 19).alias('__tc'))
    df = df.with_columns(decay_linear(pl.col('__tc'), 16).alias('__dlt'))
    df = df.with_columns(decay_linear(pl.col('__d'), 20).alias('__dld'))
    df = df.with_columns([rank(pl.col('__dld')).alias('__r'), ts_rank(pl.col('__dlt'), 7).alias('__tr')])
    expr = ((pl.col('__r') - pl.col('__tr')) * -1.0).alias('gtja_138')
    return df.select(expr).to_series()
