"""gtja_073 — standalone gtja factor.

GTJA Alpha #073 — -TS_RANK(decay-decay-corr) - RANK(decay-corr-MA30).

Guotai Junan Formula
--------------------
    -1 * TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(C, V, 10), 16), 4), 5) -
    RANK(DECAYLINEAR(CORR(VWAP, MEAN(V, 30), 4), 3))

Required panel columns: ``close``, ``volume``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_073 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #073 — -TS_RANK(decay-decay-corr) - RANK(decay-corr-MA30).

    Guotai Junan Formula
    --------------------
        -1 * TS_RANK(DECAYLINEAR(DECAYLINEAR(CORR(C, V, 10), 16), 4), 5) -
        RANK(DECAYLINEAR(CORR(VWAP, MEAN(V, 30), 4), 3))

    Required panel columns: ``close``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c1 = corr(pl.col('close'), pl.col('volume'), 10)
    arm1_inner = decay_linear(decay_linear(c1, 16), 4)
    arm1 = -1.0 * ts_rank(arm1_inner, 5)
    c2 = corr(pl.col('vwap'), mean(pl.col('volume'), 30), 4)
    arm2_inner = decay_linear(c2, 3)
    staged = panel.with_columns(arm2_inner.alias('__g073_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g073_a2_inner')).alias('__g073_r2'))
    return staged.select((arm1 - pl.col('__g073_r2')).alias('gtja_073').cast(pl.Float64)).to_series()
