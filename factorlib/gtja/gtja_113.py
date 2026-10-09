"""gtja_113 — standalone gtja factor.

GTJA #113 — -1 * RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(C,V,2) * RANK(CORR(SUM(C,5), SUM(C,20),2)).

NB: Daic115 broadcasts the inner ``corr_period=20`` rather than the
paper's ``2`` for the second corr; we follow Daic115 (parity goal).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_113 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #113 — -1 * RANK(SUM(DELAY(CLOSE,5),20)/20) * CORR(C,V,2) * RANK(CORR(SUM(C,5), SUM(C,20),2)).

    NB: Daic115 broadcasts the inner ``corr_period=20`` rather than the
    paper's ``2`` for the second corr; we follow Daic115 (parity goal).
    """
    df = panel.with_columns([(sum_(delay(pl.col('close'), 5), 20) / 20.0).alias('__a'), corr(pl.col('close'), pl.col('volume'), 20).alias('__c1'), corr(sum_(pl.col('close'), 5), sum_(pl.col('close'), 20), 20).alias('__c2')])
    df = df.with_columns([rank(pl.col('__a')).alias('__ra'), rank(pl.col('__c2')).alias('__rc2')])
    return df.select((-1.0 * pl.col('__ra') * pl.col('__c1') * pl.col('__rc2')).alias('gtja_113')).to_series()
