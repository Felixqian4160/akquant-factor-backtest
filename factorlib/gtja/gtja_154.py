"""gtja_154 — standalone gtja factor.

GTJA #154 — Sign indicator: -1/0 of (vwap-min(vwap,16)) < CORR(vwap, MA(V,180), 18).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_154 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #154 — Sign indicator: -1/0 of (vwap-min(vwap,16)) < CORR(vwap, MA(V,180), 18)."""
    df = panel.with_columns([(pl.col('vwap') - ts_min(pl.col('vwap'), 16)).alias('__a'), corr(pl.col('vwap'), mean(pl.col('volume'), 180), 18).alias('__c')])
    expr = pl.when(pl.col('__a').is_null() | pl.col('__c').is_null()).then(None).otherwise(pl.when(pl.col('__a') < pl.col('__c')).then(1.0).otherwise(-1.0)).alias('gtja_154')
    return df.select(expr).to_series()
