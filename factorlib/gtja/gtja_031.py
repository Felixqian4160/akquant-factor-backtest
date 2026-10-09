"""gtja_031 — standalone gtja factor.

GTJA Alpha #031 — 12d mean-distance ratio × 100 (vwap-anchored).

Guotai Junan Formula
--------------------
    (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_031 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #031 — 12d mean-distance ratio × 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (CLOSE - MEAN(CLOSE, 12)) / MEAN(CLOSE, 12) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    m12 = mean(vwap, 12)
    return panel.select(((vwap - m12) / m12 * 100.0).alias('gtja_031').cast(pl.Float64)).to_series()
