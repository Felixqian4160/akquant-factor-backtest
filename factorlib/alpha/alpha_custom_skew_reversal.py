"""alpha_custom_skew_reversal — standalone alpha factor.

AurumQ custom — negative CS rank of 20-day rolling skew of returns.

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Skew(returns, 20))

Polars Implementation Notes
---------------------------
Captures distributional asymmetry: stocks with strongly positive return
skew (rare large up moves) rank higher in raw form, and the sign flip
bets on reversal.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha_custom_skew_reversal import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — negative CS rank of 20-day rolling skew of returns.

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Skew(returns, 20))

    Polars Implementation Notes
    ---------------------------
    Captures distributional asymmetry: stocks with strongly positive return
    skew (rare large up moves) rank higher in raw form, and the sign flip
    bets on reversal.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    skew = ts_skew(pl.col('returns'), 20)
    staged = panel.with_columns(skew.alias('__a_skew_rev'))
    return staged.select((-1.0 * cs_rank(pl.col('__a_skew_rev'))).alias('alpha_custom_skew_reversal')).to_series()
