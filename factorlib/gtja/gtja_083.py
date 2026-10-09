"""gtja_083 — standalone gtja factor.

GTJA Alpha #083 — Negated rank of 5d covariance of rank(H) vs rank(V).

Guotai Junan Formula
--------------------
    -1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5))

Required panel columns: ``high``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_083 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #083 — Negated rank of 5d covariance of rank(H) vs rank(V).

    Guotai Junan Formula
    --------------------
        -1 * RANK(COVARIANCE(RANK(HIGH), RANK(VOLUME), 5))

    Required panel columns: ``high``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('high')).alias('__g083_rh'), rank(pl.col('volume')).alias('__g083_rv'))
    staged = staged.with_columns(covariance(pl.col('__g083_rh'), pl.col('__g083_rv'), 5).alias('__g083_c'))
    return staged.select((-1.0 * rank(pl.col('__g083_c'))).alias('gtja_083').cast(pl.Float64)).to_series()
