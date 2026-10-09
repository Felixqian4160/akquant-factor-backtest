"""alpha_custom_zscore_5d — standalone alpha factor.

AurumQ custom — 5-day rolling z-score of close (sign-flipped).

Legacy AQML Expression
----------------------
    -1 * Ts_Zscore(close, 5)

Polars Implementation Notes
---------------------------
A simple short-horizon mean-reversion factor: positive values indicate
close is well below its 5-day mean (after sign flip), suggesting a
contrarian buy.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha_custom_zscore_5d import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — 5-day rolling z-score of close (sign-flipped).

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Zscore(close, 5)

    Polars Implementation Notes
    ---------------------------
    A simple short-horizon mean-reversion factor: positive values indicate
    close is well below its 5-day mean (after sign flip), suggesting a
    contrarian buy.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select((-1.0 * ts_zscore(pl.col('close'), 5)).alias('alpha_custom_zscore_5d')).to_series()
