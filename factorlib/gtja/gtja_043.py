"""gtja_043 — standalone gtja factor.

GTJA Alpha #043 — 6d signed-volume sum (vwap-anchored direction).

Guotai Junan Formula
--------------------
    SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 6)

Required panel columns: ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_043 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #043 — 6d signed-volume sum (vwap-anchored direction).

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C,1) ? V : (C < DELAY(C,1) ? -V : 0)), 6)

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    signed_vol = ifelse(cond, pl.col('volume'), -pl.col('volume'))
    return panel.select(sum_(signed_vol, 6).alias('gtja_043').cast(pl.Float64)).to_series()
