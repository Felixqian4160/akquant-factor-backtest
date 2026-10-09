"""gtja_039 — standalone gtja factor.

GTJA Alpha #039 — Difference of two rank(decay-linear) arms, negated.

Guotai Junan Formula
--------------------
    (RANK(DECAYLINEAR(DELTA(CLOSE, 2), 8)) -
     RANK(DECAYLINEAR(CORR(VWAP*0.3 + OPEN*0.7,
                          SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1

Required panel columns: ``close``, ``vwap``, ``open``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_039 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #039 — Difference of two rank(decay-linear) arms, negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(CLOSE, 2), 8)) -
         RANK(DECAYLINEAR(CORR(VWAP*0.3 + OPEN*0.7,
                              SUM(MEAN(VOLUME, 180), 37), 14), 12))) * -1

    Required panel columns: ``close``, ``vwap``, ``open``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    p1_inner = decay_linear(delta(pl.col('close'), 2), 8)
    weighted = pl.col('vwap') * 0.3 + pl.col('open') * 0.7
    vol_long = sum_(mean(pl.col('volume'), 180), 37)
    cor = corr(weighted, vol_long, 14)
    p2_inner = decay_linear(cor, 12)
    staged = panel.with_columns(p1_inner.alias('__g039_p1'), p2_inner.alias('__g039_p2'))
    staged = staged.with_columns(rank(pl.col('__g039_p1')).alias('__g039_r1'), rank(pl.col('__g039_p2')).alias('__g039_r2'))
    expr = (pl.col('__g039_r1') - pl.col('__g039_r2')) * -1.0
    return staged.select(expr.alias('gtja_039').cast(pl.Float64)).to_series()
