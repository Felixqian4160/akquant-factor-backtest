"""gtja_059 — standalone gtja factor.

GTJA Alpha #059 — 20d sum of close-vs-extreme conditional flow (vwap).

Guotai Junan Formula
--------------------
    SUM((C = DELAY(C,1) ? 0 : C - (C > DELAY(C,1) ?
         MIN(L, DELAY(C,1)) : MAX(H, DELAY(C,1)))), 20)

Required panel columns: ``vwap``, ``low``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_059 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #059 — 20d sum of close-vs-extreme conditional flow (vwap).

    Guotai Junan Formula
    --------------------
        SUM((C = DELAY(C,1) ? 0 : C - (C > DELAY(C,1) ?
             MIN(L, DELAY(C,1)) : MAX(H, DELAY(C,1)))), 20)

    Required panel columns: ``vwap``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    v = pl.col('vwap')
    v_lag = delay(v, 1)
    pivot = pl.when(v > v_lag).then(pl.min_horizontal(pl.col('low'), v_lag)).otherwise(pl.max_horizontal(pl.col('high'), v_lag))
    inner = pl.when(v != v_lag).then(v - pivot).otherwise(0.0)
    return panel.select(sum_(inner, 20).alias('gtja_059').cast(pl.Float64)).to_series()
