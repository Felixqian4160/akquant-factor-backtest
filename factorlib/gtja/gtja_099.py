"""gtja_099 — standalone gtja factor.

GTJA Alpha #099 — Negated rank of 5d covariance of rank(C) vs rank(V).

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_099 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #099 — Negated rank of 5d covariance of rank(C) vs rank(V).

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('close')).alias('__g099_rc'), rank(pl.col('volume')).alias('__g099_rv'))
    staged = staged.with_columns(covariance(pl.col('__g099_rc'), pl.col('__g099_rv'), 5).alias('__g099_co'))
    return staged.select((-1.0 * rank(pl.col('__g099_co'))).alias('gtja_099').cast(pl.Float64)).to_series()
