"""gtja_028 — standalone gtja factor.

GTJA Alpha #028 — KDJ-style 9d stochastic with two SMA layers.

Guotai Junan Formula
--------------------
    3 * SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1) -
    2 * SMA(SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1), 3, 1)

Required panel columns: ``close``, ``low``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_028 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #028 — KDJ-style 9d stochastic with two SMA layers.

    Guotai Junan Formula
    --------------------
        3 * SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1) -
        2 * SMA(SMA((C - TSMIN(L,9)) / (TSMAX(H,9) - TSMIN(L,9)) * 100, 3, 1), 3, 1)

    Required panel columns: ``close``, ``low``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    low_min = ts_min(pl.col('low'), 9)
    high_max = ts_max(pl.col('high'), 9)
    raw = (pl.col('close') - low_min) / (high_max - low_min) * 100.0
    sma1 = sma(raw, 3, 1)
    sma2 = sma(sma1, 3, 1)
    return panel.select((3.0 * sma1 - 2.0 * sma2).alias('gtja_028').cast(pl.Float64)).to_series()
