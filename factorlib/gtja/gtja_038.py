"""gtja_038 — standalone gtja factor.

GTJA Alpha #038 — Conditional negated 2d high delta when high>20d-MA.

Guotai Junan Formula
--------------------
    (MEAN(HIGH, 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0

Required panel columns: ``high``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_038 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #038 — Conditional negated 2d high delta when high>20d-MA.

    Guotai Junan Formula
    --------------------
        (MEAN(HIGH, 20) < HIGH) ? (-1 * DELTA(HIGH, 2)) : 0

    Required panel columns: ``high``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    cond = mean(pl.col('high'), 20) < pl.col('high')
    expr = pl.when(cond).then(-1.0 * delta(pl.col('high'), 2)).otherwise(0.0)
    return panel.select(expr.alias('gtja_038').cast(pl.Float64)).to_series()
