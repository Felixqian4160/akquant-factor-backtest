"""gtja_035 — standalone gtja factor.

GTJA Alpha #035 — Min of two decay-linear/EWMA-rank arms, negated.

Guotai Junan Formula
--------------------
    MIN(
      RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
      RANK(DECAYLINEAR(CORR(VOLUME, OPEN*0.65 + CLOSE*0.35, 17), 7))
    ) * -1

Daic115 substitutes DECAYLINEAR with EWMA(alpha=1/n). We follow.

Required panel columns: ``open``, ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_035 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #035 — Min of two decay-linear/EWMA-rank arms, negated.

    Guotai Junan Formula
    --------------------
        MIN(
          RANK(DECAYLINEAR(DELTA(OPEN, 1), 15)),
          RANK(DECAYLINEAR(CORR(VOLUME, OPEN*0.65 + CLOSE*0.35, 17), 7))
        ) * -1

    Daic115 substitutes DECAYLINEAR with EWMA(alpha=1/n). We follow.

    Required panel columns: ``open``, ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    open_d = delta(pl.col('open'), 1)
    part1_inner = sma(open_d, 15, 1)
    weighted = pl.col('open') * 0.65 + pl.col('close') * 0.35
    cor_inner = corr(pl.col('volume'), weighted, 17)
    part2_inner = sma(cor_inner, 7, 1)
    staged = panel.with_columns(part1_inner.alias('__g035_p1'), part2_inner.alias('__g035_p2'))
    staged = staged.with_columns(rank(pl.col('__g035_p1')).alias('__g035_r1'), rank(pl.col('__g035_p2')).alias('__g035_r2'))
    expr = pl.min_horizontal(pl.col('__g035_r1'), pl.col('__g035_r2')) * -1.0
    return staged.select(expr.alias('gtja_035').cast(pl.Float64)).to_series()
