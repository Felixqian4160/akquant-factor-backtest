"""gtja_013 — standalone gtja factor.

GTJA Alpha #013 — Geometric mean of (H, L) minus VWAP.

Guotai Junan Formula
--------------------
    (HIGH * LOW)^0.5 - VWAP

Required panel columns: ``high``, ``low``, ``vwap``.

Direction: ``normal``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_013 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #013 — Geometric mean of (H, L) minus VWAP.

    Guotai Junan Formula
    --------------------
        (HIGH * LOW)^0.5 - VWAP

    Required panel columns: ``high``, ``low``, ``vwap``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = (pl.col('high') * pl.col('low')) ** 0.5 - pl.col('vwap')
    return panel.select(expr.alias('gtja_013').cast(pl.Float64)).to_series()
