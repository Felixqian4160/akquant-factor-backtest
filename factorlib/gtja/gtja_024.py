"""gtja_024 — standalone gtja factor.

GTJA Alpha #024 — EWMA(5,1) of 5d price change.

Guotai Junan Formula
--------------------
    SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_024 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #024 — EWMA(5,1) of 5d price change.

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 5), 5, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    diff = vwap - delay(vwap, 5)
    return panel.select(sma(diff, 5, 1).alias('gtja_024').cast(pl.Float64)).to_series()
