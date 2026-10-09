"""gtja_104 — standalone gtja factor.

GTJA #104 — -1 * DELTA(CORR(HIGH,VOL,5),5) * RANK(STD(CLOSE,20)).

Guotai Junan Formula
--------------------
    -1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_104 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #104 — -1 * DELTA(CORR(HIGH,VOL,5),5) * RANK(STD(CLOSE,20)).

    Guotai Junan Formula
    --------------------
        -1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20)))
    """
    df = panel.with_columns([corr(pl.col('high'), pl.col('volume'), 5).alias('__c'), std_(pl.col('close'), 20).alias('__s')])
    df = df.with_columns([delta(pl.col('__c'), 5).alias('__dc'), rank(pl.col('__s')).alias('__rs')])
    return df.select((-1.0 * pl.col('__dc') * pl.col('__rs')).alias('gtja_104')).to_series()
