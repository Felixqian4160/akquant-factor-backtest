"""gtja_123 — standalone gtja factor.

GTJA #123 — Binary cross of two corr-ranks (Daic115 style with NaN map).

Guotai Junan Formula
--------------------
    (RANK(CORR(SUM((H+L)/2, 20), SUM(MEAN(V,60), 20), 9))
     < RANK(CORR(LOW, VOLUME, 6))) * -1

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_123 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #123 — Binary cross of two corr-ranks (Daic115 style with NaN map).

    Guotai Junan Formula
    --------------------
        (RANK(CORR(SUM((H+L)/2, 20), SUM(MEAN(V,60), 20), 9))
         < RANK(CORR(LOW, VOLUME, 6))) * -1
    """
    df = panel.with_columns([corr(sum_((pl.col('high') + pl.col('low')) / 2.0, 20), sum_(mean(pl.col('volume'), 60), 20), 9).alias('__c1'), corr(pl.col('low'), pl.col('volume'), 6).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    expr = pl.when(pl.col('__r1').is_null() | pl.col('__r2').is_null()).then(None).otherwise((pl.col('__r1') < pl.col('__r2')).cast(pl.Float64) * -1.0).alias('gtja_123')
    return df.select(expr).to_series()
