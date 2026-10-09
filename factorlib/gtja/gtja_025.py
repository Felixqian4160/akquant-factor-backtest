"""gtja_025 — standalone gtja factor.

GTJA Alpha #025 — Composite of close-delta rank, volume EWMA rank, return-sum rank.

Guotai Junan Formula
--------------------
    (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(VOLUME / MEAN(VOLUME, 20), 9))))) *
    (1 + RANK(SUM(RET, 250)))

Daic115 uses ``ewm(alpha=1/9)`` for the decay step (not the linear-
weighted DECAYLINEAR), and `period=150` instead of 250 by default.
We follow Daic115 (period=150, ewma instead of decay_linear).

Required panel columns: ``vwap``, ``volume``, ``returns``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_025 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #025 — Composite of close-delta rank, volume EWMA rank, return-sum rank.

    Guotai Junan Formula
    --------------------
        (-1 * RANK(DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(VOLUME / MEAN(VOLUME, 20), 9))))) *
        (1 + RANK(SUM(RET, 250)))

    Daic115 uses ``ewm(alpha=1/9)`` for the decay step (not the linear-
    weighted DECAYLINEAR), and `period=150` instead of 250 by default.
    We follow Daic115 (period=150, ewma instead of decay_linear).

    Required panel columns: ``vwap``, ``volume``, ``returns``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    ret_sum = sum_(pl.col('returns'), 150)
    vol_ratio = pl.col('volume') / mean(pl.col('volume'), 20)
    vol_ewma = sma(vol_ratio, 9, 1)
    staged = panel.with_columns(delta(vwap, 7).alias('__g025_d7'), vol_ewma.alias('__g025_ve'), ret_sum.alias('__g025_rs'))
    staged = staged.with_columns(rank(pl.col('__g025_d7')).alias('__g025_rd'), rank(pl.col('__g025_ve')).alias('__g025_rv'), rank(pl.col('__g025_rs')).alias('__g025_rr'))
    expr = -1.0 * pl.col('__g025_rd') * pl.col('__g025_rv') * (1.0 + pl.col('__g025_rr'))
    return staged.select(expr.alias('gtja_025').cast(pl.Float64)).to_series()
