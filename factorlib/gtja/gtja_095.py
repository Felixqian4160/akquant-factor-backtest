"""gtja_095 — standalone gtja factor.

GTJA Alpha #095 — 20-day std of amount.

Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_095 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #095 — 20-day std of amount.

    Required panel columns: ``amount``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    return panel.select(std_(pl.col('amount'), 20).alias('gtja_095').cast(pl.Float64)).to_series()
