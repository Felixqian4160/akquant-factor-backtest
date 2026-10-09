"""alpha032 — standalone alpha factor.

Alpha #032 — short MA divergence + long-horizon vwap × delayed-close corr.

WorldQuant Formula
------------------
    scale((sum(close, 7) / 7 - close)) +
        20 * scale(correlation(vwap, delay(close, 5), 230))

Legacy AQML Expression
----------------------
    Scale(Ts_Sum(close, 7) / 7 - close)
        + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))

Polars Implementation Notes
---------------------------
1. Two cross-section scaled terms summed; both ``scale`` calls normalise
   ``sum(|x|) == 1`` per trade_date, matching STHSF.
2. The 230-day correlation is heavy — most synthetic panel rows will be
   NaN. That's fine for parity (reference is also all-NaN here).

Required panel columns: ``close``, ``vwap``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha032 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #032 — short MA divergence + long-horizon vwap × delayed-close corr.

    WorldQuant Formula
    ------------------
        scale((sum(close, 7) / 7 - close)) +
            20 * scale(correlation(vwap, delay(close, 5), 230))

    Legacy AQML Expression
    ----------------------
        Scale(Ts_Sum(close, 7) / 7 - close)
            + 20 * Scale(Ts_Corr(vwap, Delay(close, 5), 230))

    Polars Implementation Notes
    ---------------------------
    1. Two cross-section scaled terms summed; both ``scale`` calls normalise
       ``sum(|x|) == 1`` per trade_date, matching STHSF.
    2. The 230-day correlation is heavy — most synthetic panel rows will be
       NaN. That's fine for parity (reference is also all-NaN here).

    Required panel columns: ``close``, ``vwap``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col('close')
    vwap = pl.col('vwap')
    short_ma_dev = ts_sum(close, 7) / 7.0 - close
    corr = ts_corr_safe(vwap, delay(close, 5), 230)
    staged = panel.with_columns(short_ma_dev.alias('__a032_ma'), corr.alias('__a032_corr'))
    return staged.select((cs_scale(pl.col('__a032_ma')) + 20.0 * cs_scale(pl.col('__a032_corr'))).alias('alpha032')).to_series()
