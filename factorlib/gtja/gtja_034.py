"""gtja_034 — standalone gtja factor.

GTJA Alpha #034 — 12d MA over current price ratio.

Guotai Junan Formula
--------------------
    MEAN(CLOSE, 12) / CLOSE

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_034 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #034 — 12d MA over current price ratio.

    Guotai Junan Formula
    --------------------
        MEAN(CLOSE, 12) / CLOSE

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    return panel.select((mean(vwap, 12) / vwap).alias('gtja_034').cast(pl.Float64)).to_series()
