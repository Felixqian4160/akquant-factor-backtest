"""gtja_184 — standalone gtja factor.

GTJA #184 — RANK(CORR(DELAY(O-C,1), C, 200)) + RANK(O-C).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_184 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #184 — RANK(CORR(DELAY(O-C,1), C, 200)) + RANK(O-C)."""
    df = panel.with_columns([delay(pl.col('open') - pl.col('close'), 1).alias('__d'), (pl.col('open') - pl.col('close')).alias('__oc')])
    df = df.with_columns(corr(pl.col('__d'), pl.col('close'), 200).alias('__c'))
    df = df.with_columns([rank(pl.col('__c')).alias('__r1'), rank(pl.col('__oc')).alias('__r2')])
    return df.select((pl.col('__r1') + pl.col('__r2')).alias('gtja_184')).to_series()
