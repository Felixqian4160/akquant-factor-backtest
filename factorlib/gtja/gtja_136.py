"""gtja_136 — standalone gtja factor.

GTJA #136 — -RANK(DELTA(RET,3)) * CORR(OPEN, VOLUME, 10).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_136 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #136 — -RANK(DELTA(RET,3)) * CORR(OPEN, VOLUME, 10)."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    df = panel.with_columns([delta(ret, 3).alias('__dr'), corr(pl.col('open'), pl.col('volume'), 10).alias('__c')])
    df = df.with_columns(rank(pl.col('__dr')).alias('__r'))
    return df.select((-1.0 * pl.col('__r') * pl.col('__c')).alias('gtja_136')).to_series()
