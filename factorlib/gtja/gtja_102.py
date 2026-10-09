"""gtja_102 — standalone gtja factor.

GTJA #102 — Volume RSI: SMA(MAX(dV,0))/SMA(|dV|).

Guotai Junan Formula
--------------------
    SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/
    SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_102 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #102 — Volume RSI: SMA(MAX(dV,0))/SMA(|dV|).

    Guotai Junan Formula
    --------------------
        SMA(MAX(VOLUME-DELAY(VOLUME,1),0),6,1)/
        SMA(ABS(VOLUME-DELAY(VOLUME,1)),6,1)*100
    """
    dv = pl.col('volume') - delay(pl.col('volume'), 1)
    pos = pl.when(dv.is_null()).then(None).when(dv > 0).then(dv).otherwise(0.0)
    num = sma(pos, 6, 1)
    den = sma(dv.abs(), 6, 1)
    expr = (num / den * 100.0).alias('gtja_102')
    return panel.select(expr).to_series()
