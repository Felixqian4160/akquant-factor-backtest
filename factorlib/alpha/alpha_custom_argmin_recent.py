"""alpha_custom_argmin_recent — standalone alpha factor.

AurumQ custom — cross-section rank of 20-day argmin of close.

Legacy AQML Expression
----------------------
    Rank(Ts_ArgMin(close, 20))

Polars Implementation Notes
---------------------------
``ts_argmin(close, 20)`` returns the index of the recent 20-day low.
A larger value (= more recent low) ranks higher; the resulting CS rank
flags stocks where the recent bottom is fresh — a setup for mean-revert
bounce trades.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha_custom_argmin_recent import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — cross-section rank of 20-day argmin of close.

    Legacy AQML Expression
    ----------------------
        Rank(Ts_ArgMin(close, 20))

    Polars Implementation Notes
    ---------------------------
    ``ts_argmin(close, 20)`` returns the index of the recent 20-day low.
    A larger value (= more recent low) ranks higher; the resulting CS rank
    flags stocks where the recent bottom is fresh — a setup for mean-revert
    bounce trades.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    arg = ts_argmin_last(pl.col('close'), 20)
    staged = panel.with_columns(arg.alias('__a_argmin_recent'))
    return staged.select(cs_rank(pl.col('__a_argmin_recent')).alias('alpha_custom_argmin_recent')).to_series()
