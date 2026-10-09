"""gtja_015 — standalone gtja factor.

GTJA Alpha #015 — Overnight gap return (open / prior close - 1).

Guotai Junan Formula
--------------------
    OPEN / DELAY(CLOSE, 1) - 1

Required panel columns: ``open``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_015 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #015 — Overnight gap return (open / prior close - 1).

    Guotai Junan Formula
    --------------------
        OPEN / DELAY(CLOSE, 1) - 1

    Required panel columns: ``open``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col('open') / delay(pl.col('close'), 1) - 1.0
    return panel.select(expr.alias('gtja_015').cast(pl.Float64)).to_series()
