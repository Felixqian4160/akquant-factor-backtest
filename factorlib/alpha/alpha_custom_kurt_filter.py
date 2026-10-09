"""alpha_custom_kurt_filter — standalone alpha factor.

AurumQ custom — negative CS rank of 20-day rolling kurtosis of returns.

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Kurt(returns, 20))

Polars Implementation Notes
---------------------------
Excess kurtosis flags fat-tailed return regimes. The sign flip filters
out names whose recent distribution is most leptokurtic — a contrarian
risk-off tilt.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha_custom_kurt_filter import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """AurumQ custom — negative CS rank of 20-day rolling kurtosis of returns.

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Kurt(returns, 20))

    Polars Implementation Notes
    ---------------------------
    Excess kurtosis flags fat-tailed return regimes. The sign flip filters
    out names whose recent distribution is most leptokurtic — a contrarian
    risk-off tilt.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    kurt = ts_kurt(pl.col('returns'), 20)
    staged = panel.with_columns(kurt.alias('__a_kurt_filter'))
    return staged.select((-1.0 * cs_rank(pl.col('__a_kurt_filter'))).alias('alpha_custom_kurt_filter')).to_series()
