"""gtja_056 — standalone gtja factor.

GTJA Alpha #056 — Cross-sectional rank inequality: open-min vs corr^5.

Guotai Junan Formula
--------------------
    RANK(OPEN - TSMIN(OPEN, 12)) <
    RANK(RANK(CORR(SUM((H+L)/2, 19), SUM(MEAN(V, 40), 19), 13))^5)

Returns a 0/1 indicator (cast to float).

Required panel columns: ``open``, ``high``, ``low``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_056 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #056 — Cross-sectional rank inequality: open-min vs corr^5.

    Guotai Junan Formula
    --------------------
        RANK(OPEN - TSMIN(OPEN, 12)) <
        RANK(RANK(CORR(SUM((H+L)/2, 19), SUM(MEAN(V, 40), 19), 13))^5)

    Returns a 0/1 indicator (cast to float).

    Required panel columns: ``open``, ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    o = pl.col('open')
    arm1_inner = o - ts_min(o, 12)
    sum_mid = sum_((pl.col('high') + pl.col('low')) / 2.0, 19)
    sum_v = sum_(mean(pl.col('volume'), 40), 19)
    cor = corr(sum_mid, sum_v, 13)
    staged = panel.with_columns(arm1_inner.alias('__g056_a1'), cor.alias('__g056_c'))
    staged = staged.with_columns(rank(pl.col('__g056_a1')).alias('__g056_r1'), rank(pl.col('__g056_c')).alias('__g056_rc'))
    staged = staged.with_columns(rank(pl.col('__g056_rc') ** 5.0).alias('__g056_r2'))
    return staged.select((pl.col('__g056_r1') < pl.col('__g056_r2')).cast(pl.Float64).alias('gtja_056')).to_series()
