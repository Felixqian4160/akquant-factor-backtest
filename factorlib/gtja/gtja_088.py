"""gtja_088 — standalone gtja factor.

GTJA Alpha #088 — 20-day % change × 100.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_088 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #088 — 20-day % change × 100.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    c = pl.col('close')
    return panel.select(((c - delay(c, 20)) / delay(c, 20) * 100.0).alias('gtja_088').cast(pl.Float64)).to_series()
