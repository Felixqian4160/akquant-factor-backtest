"""gtja_022 — standalone gtja factor.

GTJA Alpha #022 — EWMA(12,1) of (close mean-detrend - lag3).

Guotai Junan Formula
--------------------
    SMA((C - MEAN(C,6))/MEAN(C,6) -
         DELAY((C - MEAN(C,6))/MEAN(C,6), 3), 12, 1)

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_022 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #022 — EWMA(12,1) of (close mean-detrend - lag3).

    Guotai Junan Formula
    --------------------
        SMA((C - MEAN(C,6))/MEAN(C,6) -
             DELAY((C - MEAN(C,6))/MEAN(C,6), 3), 12, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    val_mean = mean(vwap, 6)
    detrend = (vwap - val_mean) / val_mean
    diff = detrend - delay(detrend, 3)
    return panel.select(sma(diff, 12, 1).alias('gtja_022').cast(pl.Float64)).to_series()
