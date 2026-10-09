"""gtja_037 — standalone gtja factor.

GTJA Alpha #037 — Negated rank of 10d acceleration of (Sum(open,5) × Sum(ret,5)).

Guotai Junan Formula
--------------------
    -1 * RANK((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY(SUM(OPEN, 5) * SUM(RET, 5), 10))

Daic115 uses ``ret = vwap/delay(vwap)-1`` not `returns` column —
these differ by adj_factor + vwap vs close. We use the panel
``returns`` column for consistency.

Required panel columns: ``open``, ``returns``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_037 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #037 — Negated rank of 10d acceleration of (Sum(open,5) × Sum(ret,5)).

    Guotai Junan Formula
    --------------------
        -1 * RANK((SUM(OPEN, 5) * SUM(RET, 5)) - DELAY(SUM(OPEN, 5) * SUM(RET, 5), 10))

    Daic115 uses ``ret = vwap/delay(vwap)-1`` not `returns` column —
    these differ by adj_factor + vwap vs close. We use the panel
    ``returns`` column for consistency.

    Required panel columns: ``open``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    ret = vwap / delay(vwap, 1) - 1.0
    product = sum_(pl.col('open'), 5) * sum_(ret, 5)
    accel = product - delay(product, 10)
    staged = panel.with_columns(accel.alias('__g037_a'))
    return staged.select((-1.0 * rank(pl.col('__g037_a'))).alias('gtja_037').cast(pl.Float64)).to_series()
