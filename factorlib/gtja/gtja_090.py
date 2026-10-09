"""gtja_090 — standalone gtja factor.

GTJA Alpha #090 — Negated rank of 5d corr(rank-VWAP, rank-V).

Required panel columns: ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_090 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #090 — Negated rank of 5d corr(rank-VWAP, rank-V).

    Required panel columns: ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('vwap')).alias('__g090_rw'), rank(pl.col('volume')).alias('__g090_rv'))
    staged = staged.with_columns(corr(pl.col('__g090_rw'), pl.col('__g090_rv'), 5).alias('__g090_c'))
    return staged.select((-1.0 * rank(pl.col('__g090_c'))).alias('gtja_090').cast(pl.Float64)).to_series()
