"""gtja_026 — standalone gtja factor.

GTJA Alpha #026 — Long-window VWAP/close-lag correlation + mean reversion.

Guotai Junan Formula
--------------------
    (MEAN(CLOSE, 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)

Daic115 default: ma_period=12, corr_period=200.

Required panel columns: ``close``, ``vwap``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_026 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #026 — Long-window VWAP/close-lag correlation + mean reversion.

    Guotai Junan Formula
    --------------------
        (MEAN(CLOSE, 7) - CLOSE) + CORR(VWAP, DELAY(CLOSE, 5), 230)

    Daic115 default: ma_period=12, corr_period=200.

    Required panel columns: ``close``, ``vwap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    expr = mean(pl.col('close'), 12) - pl.col('close') + corr(pl.col('vwap'), delay(pl.col('close'), 5), 200)
    return panel.select(expr.alias('gtja_026').cast(pl.Float64)).to_series()
