"""gtja_077 — standalone gtja factor.

GTJA Alpha #077 — MIN of two rank(DECAYLINEAR(...)) arms.

Guotai Junan Formula
--------------------
    MIN(
      RANK(DECAYLINEAR(((H+L)/2 + H) - (VWAP + H), 20)),
      RANK(DECAYLINEAR(CORR((H+L)/2, MEAN(V, 40), 3), 6))
    )

Required panel columns: ``high``, ``low``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_077 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #077 — MIN of two rank(DECAYLINEAR(...)) arms.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(((H+L)/2 + H) - (VWAP + H), 20)),
          RANK(DECAYLINEAR(CORR((H+L)/2, MEAN(V, 40), 3), 6))
        )

    Required panel columns: ``high``, ``low``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    inner1 = mid + pl.col('high') - (pl.col('vwap') + pl.col('high'))
    arm1_inner = decay_linear(inner1, 20)
    cor = corr(mid, mean(pl.col('volume'), 40), 3)
    arm2_inner = decay_linear(cor, 6)
    staged = panel.with_columns(arm1_inner.alias('__g077_a1'), arm2_inner.alias('__g077_a2'))
    staged = staged.with_columns(rank(pl.col('__g077_a1')).alias('__g077_r1'), rank(pl.col('__g077_a2')).alias('__g077_r2'))
    return staged.select(pl.min_horizontal(pl.col('__g077_r1'), pl.col('__g077_r2')).alias('gtja_077').cast(pl.Float64)).to_series()
