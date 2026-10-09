"""gtja_070 — standalone gtja factor.

GTJA Alpha #070 — 6-day std of amount.

Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_070 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #070 — 6-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('amount'), 6).alias('gtja_070').cast(pl.Float64)).to_series()
