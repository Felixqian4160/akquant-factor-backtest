"""gtja_032 — standalone gtja factor.

GTJA Alpha #032 — Negated 3d sum of rank(corr(rank-H, rank-Vol, 3)).

Guotai Junan Formula
--------------------
    -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)

Required panel columns: ``high``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_032 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #032 — Negated 3d sum of rank(corr(rank-H, rank-Vol, 3)).

    Guotai Junan Formula
    --------------------
        -1 * SUM(RANK(CORR(RANK(HIGH), RANK(VOLUME), 3)), 3)

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('high')).alias('__g032_rh'), rank(pl.col('volume')).alias('__g032_rv'))
    staged = staged.with_columns(corr(pl.col('__g032_rh'), pl.col('__g032_rv'), 3).alias('__g032_c'))
    staged = staged.with_columns(rank(pl.col('__g032_c')).alias('__g032_rc'))
    return staged.select((-1.0 * sum_(pl.col('__g032_rc'), 3)).alias('gtja_032').cast(pl.Float64)).to_series()
