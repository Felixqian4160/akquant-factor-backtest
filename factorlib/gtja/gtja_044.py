"""gtja_044 — standalone gtja factor.

GTJA Alpha #044 — Sum of two TSRANK(DECAYLINEAR(...)) arms.

Guotai Junan Formula
--------------------
    TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4) +
    TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)

Required panel columns: ``low``, ``volume``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_044 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #044 — Sum of two TSRANK(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        TSRANK(DECAYLINEAR(CORR(LOW, MEAN(VOLUME, 10), 7), 6), 4) +
        TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)

    Required panel columns: ``low``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    cor_inner = corr(pl.col('low'), mean(pl.col('volume'), 10), 7)
    arm1 = ts_rank(decay_linear(cor_inner, 6), 4)
    arm2 = ts_rank(decay_linear(delta(pl.col('vwap'), 3), 10), 15)
    return panel.select((arm1 + arm2).alias('gtja_044').cast(pl.Float64)).to_series()
