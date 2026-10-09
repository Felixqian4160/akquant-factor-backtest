"""gtja_086 — standalone gtja factor.

GTJA Alpha #086 — 20/10/0 close-acceleration regime ternary.

Guotai Junan Formula
--------------------
    part1 = (DELAY(C, 20) - DELAY(C, 10)) / 10
    part2 = (DELAY(C, 10) - C) / 10
    if (0.25 < (part1 - part2)) -1
    elif ((part1 - part2) < 0) 1
    else -1 * (C - DELAY(C, 1))

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_086 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #086 — 20/10/0 close-acceleration regime ternary.

    Guotai Junan Formula
    --------------------
        part1 = (DELAY(C, 20) - DELAY(C, 10)) / 10
        part2 = (DELAY(C, 10) - C) / 10
        if (0.25 < (part1 - part2)) -1
        elif ((part1 - part2) < 0) 1
        else -1 * (C - DELAY(C, 1))

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    c = pl.col('close')
    part1 = (delay(c, 20) - delay(c, 10)) / 10.0
    part2 = (delay(c, 10) - c) / 10.0
    diff = part1 - part2
    base = -1.0 * (c - delay(c, 1))
    expr = pl.when(diff > 0.25).then(-1.0).otherwise(pl.when(diff < 0.0).then(1.0).otherwise(base))
    return panel.select(expr.alias('gtja_086').cast(pl.Float64)).to_series()
