"""gtja_092 — standalone gtja factor.

GTJA Alpha #092 — MAX of rank-decay-delta and TS-rank-decay-abs-corr, negated.

Guotai Junan Formula
--------------------
    MAX(
      RANK(DECAYLINEAR(DELTA(C*0.35 + VWAP*0.65, 2), 3)),
      TSRANK(DECAYLINEAR(|CORR(MEAN(V, 180), C, 13)|, 5), 15)
    ) * -1

Required panel columns: ``close``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_092 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #092 — MAX of rank-decay-delta and TS-rank-decay-abs-corr, negated.

    Guotai Junan Formula
    --------------------
        MAX(
          RANK(DECAYLINEAR(DELTA(C*0.35 + VWAP*0.65, 2), 3)),
          TSRANK(DECAYLINEAR(|CORR(MEAN(V, 180), C, 13)|, 5), 15)
        ) * -1

    Required panel columns: ``close``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('close') * 0.35 + pl.col('vwap') * 0.65
    arm1_inner = decay_linear(delta(weighted, 2), 3)
    cor = corr(mean(pl.col('volume'), 180), pl.col('close'), 13)
    arm2_inner = decay_linear(abs_(cor), 5)
    arm2 = ts_rank(arm2_inner, 15)
    staged = panel.with_columns(arm1_inner.alias('__g092_a1'))
    staged = staged.with_columns(rank(pl.col('__g092_a1')).alias('__g092_r1'))
    return staged.select((pl.max_horizontal(pl.col('__g092_r1'), arm2) * -1.0).alias('gtja_092').cast(pl.Float64)).to_series()
