"""gtja_098 — standalone gtja factor.

GTJA Alpha #098 — Long-trend ternary: 100d MA acceleration regime.

Guotai Junan Formula
--------------------
    cond = DELTA(SUM(C, 100)/100, 100) / DELAY(C, 100)
    cond <= 0.05 ? -1*(C - TSMIN(C, 100)) : -1 * DELTA(C, 3)

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_098 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #098 — Long-trend ternary: 100d MA acceleration regime.

    Guotai Junan Formula
    --------------------
        cond = DELTA(SUM(C, 100)/100, 100) / DELAY(C, 100)
        cond <= 0.05 ? -1*(C - TSMIN(C, 100)) : -1 * DELTA(C, 3)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    cond_val = delta(sum_(c, 100) / 100.0, 100) / delay(c, 100)
    branch_low = -1.0 * (c - ts_min(c, 100))
    branch_hi = -1.0 * delta(c, 3)
    expr = pl.when(cond_val <= 0.05).then(branch_low).otherwise(branch_hi)
    return panel.select(expr.alias('gtja_098').cast(pl.Float64)).to_series()
