"""gtja_023 — standalone gtja factor.

GTJA Alpha #023 — Up-day std share over 20d, smoothed by SMA(20,1).

Guotai Junan Formula
--------------------
    SMA(cond ? STD(C,20) : 0, 20, 1) /
    (SMA(cond ? STD(C,20) : 0, 20, 1) + SMA(!cond ? STD(C,20) : 0, 20, 1)) * 100
    where cond = C > DELAY(C, 1)

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_023 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #023 — Up-day std share over 20d, smoothed by SMA(20,1).

    Guotai Junan Formula
    --------------------
        SMA(cond ? STD(C,20) : 0, 20, 1) /
        (SMA(cond ? STD(C,20) : 0, 20, 1) + SMA(!cond ? STD(C,20) : 0, 20, 1)) * 100
        where cond = C > DELAY(C, 1)

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    vwap = pl.col('vwap')
    cond = vwap > delay(vwap, 1)
    s = std_(vwap, 20)
    up = ifelse(cond, s, 0.0)
    dn = ifelse(~cond, s, 0.0)
    sma_up = sma(up, 20, 1)
    sma_dn = sma(dn, 20, 1)
    expr = sma_up / (sma_up + sma_dn) * 100.0
    return panel.select(expr.alias('gtja_023').cast(pl.Float64)).to_series()
