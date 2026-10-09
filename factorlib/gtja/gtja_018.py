"""gtja_018 — standalone gtja factor.

GTJA Alpha #018 — 5d price ratio (close / delay(close, 5)).

Guotai Junan Formula
--------------------
    CLOSE / DELAY(CLOSE, 5)

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_018 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #018 — 5d price ratio (close / delay(close, 5)).

    Guotai Junan Formula
    --------------------
        CLOSE / DELAY(CLOSE, 5)

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    expr = pl.col('close') / delay(pl.col('close'), 5)
    return panel.select(expr.alias('gtja_018').cast(pl.Float64)).to_series()
