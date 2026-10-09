"""gtja_036 — standalone gtja factor.

GTJA Alpha #036 — Rank of 2d sum of rank-volume/rank-vwap 6d corr.

Guotai Junan Formula
--------------------
    RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))

Required panel columns: ``volume``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_036 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #036 — Rank of 2d sum of rank-volume/rank-vwap 6d corr.

    Guotai Junan Formula
    --------------------
        RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))

    Required panel columns: ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('volume')).alias('__g036_rv'), rank(pl.col('vwap')).alias('__g036_rw'))
    staged = staged.with_columns(corr(pl.col('__g036_rv'), pl.col('__g036_rw'), 6).alias('__g036_c'))
    staged = staged.with_columns(sum_(pl.col('__g036_c'), 2).alias('__g036_s'))
    return staged.select(rank(pl.col('__g036_s')).alias('gtja_036').cast(pl.Float64)).to_series()
