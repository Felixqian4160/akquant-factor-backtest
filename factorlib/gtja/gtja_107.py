"""gtja_107 — standalone gtja factor.

GTJA #107 — Triple-rank gap product across O-H/O-C/O-L.

Guotai Junan Formula
--------------------
    ((-1 * RANK(OPEN - DELAY(HIGH, 1))) *
      RANK(OPEN - DELAY(CLOSE, 1))) *
      RANK(OPEN - DELAY(LOW, 1))

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_107 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #107 — Triple-rank gap product across O-H/O-C/O-L.

    Guotai Junan Formula
    --------------------
        ((-1 * RANK(OPEN - DELAY(HIGH, 1))) *
          RANK(OPEN - DELAY(CLOSE, 1))) *
          RANK(OPEN - DELAY(LOW, 1))
    """
    df = panel.with_columns([(pl.col('open') - delay(pl.col('high'), 1)).alias('__a'), (pl.col('open') - delay(pl.col('close'), 1)).alias('__b'), (pl.col('open') - delay(pl.col('low'), 1)).alias('__c')])
    return df.select((-1.0 * rank(pl.col('__a')) * rank(pl.col('__b')) * rank(pl.col('__c'))).alias('gtja_107')).to_series()
