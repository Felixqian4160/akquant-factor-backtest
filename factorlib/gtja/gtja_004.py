"""gtja_004 — standalone gtja factor.

GTJA Alpha #004 — Trend regime conditional with volume gate.

Guotai Junan Formula
--------------------
    if((MEAN(CLOSE,8)+STD(CLOSE,8))<MEAN(CLOSE,2)) -1
    elif(MEAN(CLOSE,2)<(MEAN(CLOSE,8)-STD(CLOSE,8))) 1
    elif(VOLUME/MEAN(VOLUME,20) >= 1) 1
    else -1

Required panel columns: ``close``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_004 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #004 — Trend regime conditional with volume gate.

    Guotai Junan Formula
    --------------------
        if((MEAN(CLOSE,8)+STD(CLOSE,8))<MEAN(CLOSE,2)) -1
        elif(MEAN(CLOSE,2)<(MEAN(CLOSE,8)-STD(CLOSE,8))) 1
        elif(VOLUME/MEAN(VOLUME,20) >= 1) 1
        else -1

    Required panel columns: ``close``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    m8 = mean(pl.col('close'), 8)
    s8 = std_(pl.col('close'), 8)
    m2 = mean(pl.col('close'), 2)
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    expr = pl.when(m8 + s8 < m2).then(-1.0).otherwise(pl.when(m2 < m8 - s8).then(1.0).otherwise(pl.when(vol_ratio >= 1.0).then(1.0).otherwise(-1.0)))
    return panel.select(expr.alias('gtja_004').cast(pl.Float64)).to_series()
