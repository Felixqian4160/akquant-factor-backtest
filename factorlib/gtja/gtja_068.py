"""gtja_068 — standalone gtja factor.

GTJA Alpha #068 — EWMA(15,2) of mid-price acceleration × (H-L)/V.

Guotai Junan Formula
--------------------
    SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/V, 15, 2)

Required panel columns: ``high``, ``low``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_068 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #068 — EWMA(15,2) of mid-price acceleration × (H-L)/V.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/V, 15, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    mid_lag = (delay(pl.col('high'), 1) + delay(pl.col('low'), 1)) / 2.0
    inner = (mid - mid_lag) * (pl.col('high') - pl.col('low')) / pl.col('volume')
    return panel.select(sma(inner, 15, 2).alias('gtja_068').cast(pl.Float64)).to_series()
