"""gtja_055 — standalone gtja factor.

GTJA Alpha #055 — 20d sum of TR-normalised acceleration × max(|H-C-1|, |L-C-1|).

Guotai Junan Formula
--------------------
    SUM(16 * (C - DELAY(C,1) + (C-O)/2 + DELAY(C,1) - DELAY(O,1)) /
        (asymmetric TR normaliser per spec) *
        MAX(|H - DELAY(C,1)|, |L - DELAY(C,1)|), 20)

Required panel columns: ``open``, ``high``, ``low``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_055 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #055 — 20d sum of TR-normalised acceleration × max(|H-C-1|, |L-C-1|).

    Guotai Junan Formula
    --------------------
        SUM(16 * (C - DELAY(C,1) + (C-O)/2 + DELAY(C,1) - DELAY(O,1)) /
            (asymmetric TR normaliser per spec) *
            MAX(|H - DELAY(C,1)|, |L - DELAY(C,1)|), 20)

    Required panel columns: ``open``, ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    o = pl.col('open')
    h = pl.col('high')
    lw = pl.col('low')
    p1 = abs_(h - delay(c, 1))
    p2 = abs_(lw - delay(c, 1))
    p3 = abs_(h - delay(lw, 1))
    p4 = abs_(delay(c, 1) - delay(o, 1))
    var1 = p1 + p2 / 2.0 + p4 / 4.0
    var2 = p2 + p1 / 2.0 + p4 / 4.0
    var3 = p3 + p4 / 4.0
    cond_a = (p1 > p2) & (p1 > p3)
    cond_b = (p2 > p3) & (p2 > p1)
    denom = pl.when(cond_a).then(var1).otherwise(pl.when(cond_b).then(var2).otherwise(var3))
    accel = c - delay(c, 1) + (c - o) / 2.0 + delay(c, 1) - delay(o, 1)
    inner = 16.0 * accel / denom * pl.max_horizontal(p1, p2)
    return panel.select(sum_(inner, 20).alias('gtja_055').cast(pl.Float64)).to_series()
