"""gtja_096 — standalone gtja factor.

GTJA Alpha #096 — SMA(SMA(stochastic-%K, 3, 1), 3, 1).

Guotai Junan Formula
--------------------
    SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1)

Required panel columns: ``close``, ``low``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_096 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #096 — SMA(SMA(stochastic-%K, 3, 1), 3, 1).

    Guotai Junan Formula
    --------------------
        SMA(SMA((C - TSMIN(L, 9)) / (TSMAX(H, 9) - TSMIN(L, 9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    raw = (pl.col('close') - ts_min(pl.col('low'), 9)) / (ts_max(pl.col('high'), 9) - ts_min(pl.col('low'), 9)) * 100.0
    return panel.select(sma(sma(raw, 3, 1), 3, 1).alias('gtja_096').cast(pl.Float64)).to_series()
