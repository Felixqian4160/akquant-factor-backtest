"""gtja_091 — standalone gtja factor.

GTJA Alpha #091 — Negated product of two rank arms.

Guotai Junan Formula
--------------------
    -1 * RANK(C - TSMAX(C, 5)) * RANK(CORR(MEAN(V, 40), L, 5))

Required panel columns: ``close``, ``volume``, ``low``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_091 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #091 — Negated product of two rank arms.

    Guotai Junan Formula
    --------------------
        -1 * RANK(C - TSMAX(C, 5)) * RANK(CORR(MEAN(V, 40), L, 5))

    Required panel columns: ``close``, ``volume``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    c = pl.col('close')
    arm1_inner = c - ts_max(c, 5)
    cor = corr(mean(pl.col('volume'), 40), pl.col('low'), 5)
    staged = panel.with_columns(arm1_inner.alias('__g091_a1'), cor.alias('__g091_c'))
    staged = staged.with_columns(rank(pl.col('__g091_a1')).alias('__g091_r1'), rank(pl.col('__g091_c')).alias('__g091_r2'))
    return staged.select((pl.col('__g091_r1') * pl.col('__g091_r2') * -1.0).alias('gtja_091').cast(pl.Float64)).to_series()
