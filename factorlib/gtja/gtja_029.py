"""gtja_029 — standalone gtja factor.

GTJA Alpha #029 — 6d % change × log(volume).

Guotai Junan Formula
--------------------
    (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME

Daic115 uses ``log(volume)`` rather than raw volume — we follow.

Required panel columns: ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_029 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #029 — 6d % change × log(volume).

    Guotai Junan Formula
    --------------------
        (CLOSE - DELAY(CLOSE, 6)) / DELAY(CLOSE, 6) * VOLUME

    Daic115 uses ``log(volume)`` rather than raw volume — we follow.

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    expr = (vwap - delay(vwap, 6)) / delay(vwap, 6) * pl.col('volume').log()
    return panel.select(expr.alias('gtja_029').cast(pl.Float64)).to_series()
