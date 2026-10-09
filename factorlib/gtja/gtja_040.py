"""gtja_040 — standalone gtja factor.

GTJA Alpha #040 — 26d up/down volume ratio × 100.

Guotai Junan Formula
--------------------
    SUM((C > DELAY(C, 1) ? VOLUME : 0), 26) /
    SUM((C <= DELAY(C, 1) ? VOLUME : 0), 26) * 100

Required panel columns: ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_040 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #040 — 26d up/down volume ratio × 100.

    Guotai Junan Formula
    --------------------
        SUM((C > DELAY(C, 1) ? VOLUME : 0), 26) /
        SUM((C <= DELAY(C, 1) ? VOLUME : 0), 26) * 100

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    up = ifelse(cond, pl.col('volume'), 0.0)
    dn = ifelse(~cond, pl.col('volume'), 0.0)
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(expr.alias('gtja_040').cast(pl.Float64)).to_series()
