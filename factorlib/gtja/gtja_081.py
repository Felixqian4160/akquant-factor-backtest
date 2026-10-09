"""gtja_081 — standalone gtja factor.

GTJA Alpha #081 — EWMA(21, 2) of volume.

Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_081 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #081 — EWMA(21, 2) of volume.

    Required panel columns: ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    return panel.select(sma(pl.col('volume'), 21, 2).alias('gtja_081').cast(pl.Float64)).to_series()
