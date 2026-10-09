"""gtja_016 — standalone gtja factor.

GTJA Alpha #016 — Negated 5d max of rank(volume,vwap)-corr rank.

Guotai Junan Formula
--------------------
    -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)

Required panel columns: ``volume``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_016 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #016 — Negated 5d max of rank(volume,vwap)-corr rank.

    Guotai Junan Formula
    --------------------
        -1 * TSMAX(RANK(CORR(RANK(VOLUME), RANK(VWAP), 5)), 5)

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('volume')).alias('__g016_rv'), rank(pl.col('vwap')).alias('__g016_rw'))
    staged = staged.with_columns(corr(pl.col('__g016_rv'), pl.col('__g016_rw'), 5).alias('__g016_c'))
    staged = staged.with_columns(rank(pl.col('__g016_c')).alias('__g016_r'))
    return staged.select((-1.0 * ts_max(pl.col('__g016_r'), 5)).alias('gtja_016').cast(pl.Float64)).to_series()
