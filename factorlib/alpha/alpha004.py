"""alpha004 — standalone alpha factor.

Alpha #004 — short-window time-series rank of cross-section rank of low.

WorldQuant Formula
------------------
    -1 * Ts_Rank(rank(low), 9)

Legacy AQML Expression
----------------------
    -1 * Ts_Rank(Rank(low), 9)

Polars Implementation Notes
---------------------------
1. ``Rank(low)`` is a per-day cross-section pct rank.
2. ``Ts_Rank(..., 9)`` is the 9-day rolling pct rank (last value within
   window) of that ranked column, per stock.
3. Final ``-1 *`` flips the sign so high values indicate "low has been
   persistently low" — a contrarian / mean-revert signal.

Required panel columns: ``low``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha004 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #004 — short-window time-series rank of cross-section rank of low.

    WorldQuant Formula
    ------------------
        -1 * Ts_Rank(rank(low), 9)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Rank(Rank(low), 9)

    Polars Implementation Notes
    ---------------------------
    1. ``Rank(low)`` is a per-day cross-section pct rank.
    2. ``Ts_Rank(..., 9)`` is the 9-day rolling pct rank (last value within
       window) of that ranked column, per stock.
    3. Final ``-1 *`` flips the sign so high values indicate "low has been
       persistently low" — a contrarian / mean-revert signal.

    Required panel columns: ``low``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    staged = panel.with_columns(cs_rank(pl.col('low')).alias('__a004_lr'))
    return staged.select((-1.0 * ts_rank_int(pl.col('__a004_lr'), 9)).alias('alpha004')).to_series()
