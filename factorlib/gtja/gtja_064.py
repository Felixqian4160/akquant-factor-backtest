"""gtja_064 — standalone gtja factor.

GTJA Alpha #064 — Max of two rank(decay-corr) arms.

Guotai Junan Formula
--------------------
    MAX(
      RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
      RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(V, 60)), 4), 13), 14))
    ) * -1

Daic115 omits the `* -1`; we follow.

Required panel columns: ``vwap``, ``volume``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_064 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #064 — Max of two rank(decay-corr) arms.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 4), 4)),
          RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(V, 60)), 4), 13), 14))
        ) * -1

    Daic115 omits the `* -1`; we follow.

    Required panel columns: ``vwap``, ``volume``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(rank(pl.col('vwap')).alias('__g064_rw'), rank(pl.col('volume')).alias('__g064_rv'), rank(pl.col('close')).alias('__g064_rc'), rank(mean(pl.col('volume'), 60)).alias('__g064_rmv'))
    staged = staged.with_columns(corr(pl.col('__g064_rw'), pl.col('__g064_rv'), 4).alias('__g064_c1'), corr(pl.col('__g064_rc'), pl.col('__g064_rmv'), 4).alias('__g064_c2'))
    staged = staged.with_columns(decay_linear(pl.col('__g064_c1'), 4).alias('__g064_a1_inner'), ts_max(pl.col('__g064_c2'), 13).alias('__g064_a2_max'))
    staged = staged.with_columns(decay_linear(pl.col('__g064_a2_max'), 14).alias('__g064_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g064_a1_inner')).alias('__g064_r1'), rank(pl.col('__g064_a2_inner')).alias('__g064_r2'))
    return staged.select(pl.max_horizontal(pl.col('__g064_r1'), pl.col('__g064_r2')).alias('gtja_064').cast(pl.Float64)).to_series()
