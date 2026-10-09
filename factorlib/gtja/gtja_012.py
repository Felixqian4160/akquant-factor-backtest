"""gtja_012 — standalone gtja factor.

GTJA Alpha #012 — Rank(O - MA(VWAP,10)) * -1 * Rank(|C - VWAP|).

Guotai Junan Formula
--------------------
    RANK(OPEN - SUM(VWAP,10)/10) * (-1 * RANK(ABS(CLOSE - VWAP)))

Required panel columns: ``open``, ``vwap``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_012 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #012 — Rank(O - MA(VWAP,10)) * -1 * Rank(|C - VWAP|).

    Guotai Junan Formula
    --------------------
        RANK(OPEN - SUM(VWAP,10)/10) * (-1 * RANK(ABS(CLOSE - VWAP)))

    Required panel columns: ``open``, ``vwap``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    staged = panel.with_columns((pl.col('open') - mean(pl.col('vwap'), 10)).alias('__g012_o'), abs_(pl.col('close') - pl.col('vwap')).alias('__g012_d'))
    return staged.select((rank(pl.col('__g012_o')) * (-1.0 * rank(pl.col('__g012_d')))).alias('gtja_012').cast(pl.Float64)).to_series()
