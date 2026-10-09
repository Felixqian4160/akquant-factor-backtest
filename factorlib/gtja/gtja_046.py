"""gtja_046 — standalone gtja factor.

GTJA Alpha #046 — Multi-window MA average / close.

Guotai Junan Formula
--------------------
    (MEAN(C,3) + MEAN(C,6) + MEAN(C,12) + MEAN(C,24)) / (4 * C)

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_046 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #046 — Multi-window MA average / close.

    Guotai Junan Formula
    --------------------
        (MEAN(C,3) + MEAN(C,6) + MEAN(C,12) + MEAN(C,24)) / (4 * C)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    c = pl.col('close')
    expr = (mean(c, 3) + mean(c, 6) + mean(c, 12) + mean(c, 24)) / (4.0 * c)
    return panel.select(expr.alias('gtja_046').cast(pl.Float64)).to_series()
