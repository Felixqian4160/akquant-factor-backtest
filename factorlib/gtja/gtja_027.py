"""gtja_027 — standalone gtja factor.

GTJA Alpha #027 — EWMA(12,1) of 3d+6d % momentum sum.

Guotai Junan Formula
--------------------
    WMA(((C/DELAY(C,3) - 1)*100 + (C/DELAY(C,6) - 1)*100), 12)

Daic115 substitutes WMA with SMA(.,12,1) — we follow that.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_027 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #027 — EWMA(12,1) of 3d+6d % momentum sum.

    Guotai Junan Formula
    --------------------
        WMA(((C/DELAY(C,3) - 1)*100 + (C/DELAY(C,6) - 1)*100), 12)

    Daic115 substitutes WMA with SMA(.,12,1) — we follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    short = (vwap / delay(vwap, 3) - 1.0) * 100.0
    long_ = (vwap / delay(vwap, 6) - 1.0) * 100.0
    return panel.select(sma(short + long_, 12, 1).alias('gtja_027').cast(pl.Float64)).to_series()
