"""gtja_009 — standalone gtja factor.

GTJA Alpha #009 — EWMA(7,2) of mid-price acceleration weighted by HL/Volume.

Guotai Junan Formula
--------------------
    SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/VOLUME, 7, 2)

Required panel columns: ``high``, ``low``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_009 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #009 — EWMA(7,2) of mid-price acceleration weighted by HL/Volume.

    Guotai Junan Formula
    --------------------
        SMA(((H+L)/2 - (DELAY(H,1)+DELAY(L,1))/2) * (H-L)/VOLUME, 7, 2)

    Required panel columns: ``high``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    mid_prev = (delay(pl.col('high'), 1) + delay(pl.col('low'), 1)) / 2.0
    range_per_vol = (pl.col('high') - pl.col('low')) / pl.col('volume')
    inner = (mid - mid_prev) * range_per_vol
    return panel.select(sma(inner, 7, 2).alias('gtja_009').cast(pl.Float64)).to_series()
