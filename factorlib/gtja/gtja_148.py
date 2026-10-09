"""gtja_148 — standalone gtja factor.

GTJA #148 — (RANK(CORR(OPEN, SUM(MEAN(V,60),9), 6)) < RANK(OPEN-TSMIN(OPEN,14))) * -1.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_148 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #148 — (RANK(CORR(OPEN, SUM(MEAN(V,60),9), 6)) < RANK(OPEN-TSMIN(OPEN,14))) * -1."""
    df = panel.with_columns([corr(pl.col('open'), sum_(mean(pl.col('volume'), 60), 9), 6).alias('__c'), (pl.col('open') - ts_min(pl.col('open'), 14)).alias('__d')])
    df = df.with_columns([rank(pl.col('__c')).alias('__r1'), rank(pl.col('__d')).alias('__r2')])
    expr = pl.when(pl.col('__r1').is_null() | pl.col('__r2').is_null()).then(None).otherwise((pl.col('__r1') < pl.col('__r2')).cast(pl.Float64) * -1.0).alias('gtja_148')
    return df.select(expr).to_series()
