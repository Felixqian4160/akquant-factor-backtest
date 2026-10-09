"""gtja_089 — standalone gtja factor.

GTJA Alpha #089 — MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27)).

Guotai Junan Formula
--------------------
    2 * (SMA(C, 13, 2) - SMA(C, 27, 2) -
         SMA(SMA(C, 13, 2) - SMA(C, 27, 2), 10, 2))

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_089 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #089 — MACD-style oscillator: 2*(SMA13 - SMA27 - SMA10(SMA13-SMA27)).

    Guotai Junan Formula
    --------------------
        2 * (SMA(C, 13, 2) - SMA(C, 27, 2) -
             SMA(SMA(C, 13, 2) - SMA(C, 27, 2), 10, 2))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    ma_short = sma(c, 13, 2)
    ma_long = sma(c, 27, 2)
    diff = ma_short - ma_long
    return panel.select((2.0 * (ma_short - ma_long - sma(diff, 10, 2))).alias('gtja_089').cast(pl.Float64)).to_series()
