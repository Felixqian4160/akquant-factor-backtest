"""gtja_140 — standalone gtja factor.

GTJA #140 — MIN of two decay-linear ranks.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_140 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #140 — MIN of two decay-linear ranks."""
    df = panel.with_columns([(rank(pl.col('open')) + rank(pl.col('low')) - rank(pl.col('high')) - rank(pl.col('close'))).alias('__a'), ts_rank(pl.col('close'), 8).alias('__tc'), ts_rank(mean(pl.col('volume'), 60), 20).alias('__tv')])
    df = df.with_columns(corr(pl.col('__tc'), pl.col('__tv'), 8).alias('__c'))
    df = df.with_columns(decay_linear(pl.col('__c'), 7).alias('__dlc'))
    df = df.with_columns(decay_linear(pl.col('__a'), 8).alias('__dla'))
    df = df.with_columns([rank(pl.col('__dla')).alias('__r1'), ts_rank(pl.col('__dlc'), 3).alias('__r2')])
    expr = pl.min_horizontal([pl.col('__r1'), pl.col('__r2')]).alias('gtja_140')
    return df.select(expr).to_series()
