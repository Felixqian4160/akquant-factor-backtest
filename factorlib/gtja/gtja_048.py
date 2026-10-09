"""gtja_048 — standalone gtja factor.

GTJA Alpha #048 — Rank(sign-sum) × SUM(V,5)/SUM(V,20).

Guotai Junan Formula
--------------------
    -1 * RANK(SIGN(C - DELAY(C,1)) + SIGN(DELAY(C,1) - DELAY(C,2)) +
              SIGN(DELAY(C,2) - DELAY(C,3))) * SUM(V,5) / SUM(V,20)

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_048 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #048 — Rank(sign-sum) × SUM(V,5)/SUM(V,20).

    Guotai Junan Formula
    --------------------
        -1 * RANK(SIGN(C - DELAY(C,1)) + SIGN(DELAY(C,1) - DELAY(C,2)) +
                  SIGN(DELAY(C,2) - DELAY(C,3))) * SUM(V,5) / SUM(V,20)

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    s1 = sign_(delta(c, 1))
    s2 = sign_(delta(delay(c, 1), 1))
    s3 = sign_(delta(delay(c, 2), 1))
    sgn_sum = s1 + s2 + s3
    staged = panel.with_columns(sgn_sum.alias('__g048_s'))
    staged = staged.with_columns(rank(pl.col('__g048_s')).alias('__g048_r'))
    expr = pl.col('__g048_r') * sum_(pl.col('volume'), 5) / sum_(pl.col('volume'), 20)
    return staged.select(expr.alias('gtja_048').cast(pl.Float64)).to_series()
