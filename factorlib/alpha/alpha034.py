"""alpha034 — standalone alpha factor.

Alpha #034 — short/long return-vol ratio plus 1-day close delta.

WorldQuant Formula
------------------
    rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) +
          (1 - rank(delta(close, 1)))))

Legacy AQML Expression
----------------------
    Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) +
         (1 - Rank(Delta(close, 1))))

Polars Implementation Notes
---------------------------
STHSF rewrites the inner expression as ``2 - rank(ratio) - rank(delta)``
and replaces inf/NaN in the volatility ratio with 1 to avoid
constant-window pollution. We mirror both behaviours.

Required panel columns: ``returns``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volatility.py

Usage:
    from factorlib.alpha.alpha034 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, signed_power, ts_argmax, ts_corr_safe, ts_kurt, ts_skew, ts_std

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #034 — short/long return-vol ratio plus 1-day close delta.

    WorldQuant Formula
    ------------------
        rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) +
              (1 - rank(delta(close, 1)))))

    Legacy AQML Expression
    ----------------------
        Rank((1 - Rank(Ts_Std(returns, 2) / Ts_Std(returns, 5))) +
             (1 - Rank(Delta(close, 1))))

    Polars Implementation Notes
    ---------------------------
    STHSF rewrites the inner expression as ``2 - rank(ratio) - rank(delta)``
    and replaces inf/NaN in the volatility ratio with 1 to avoid
    constant-window pollution. We mirror both behaviours.

    Required panel columns: ``returns``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volatility``
    """
    ratio = ts_std(pl.col('returns'), 2) / ts_std(pl.col('returns'), 5)
    delta_close = delta(pl.col('close'), 1)
    staged = panel.with_columns(ratio.alias('__a034_ratio'), delta_close.alias('__a034_d'))
    staged = staged.with_columns(pl.when(pl.col('__a034_ratio').is_finite()).then(pl.col('__a034_ratio')).otherwise(1.0).alias('__a034_ratio'))
    inner = 2.0 - cs_rank(pl.col('__a034_ratio')) - cs_rank(pl.col('__a034_d'))
    staged = staged.with_columns(inner.alias('__a034_inner'))
    return staged.select(cs_rank(pl.col('__a034_inner')).alias('alpha034')).to_series()
