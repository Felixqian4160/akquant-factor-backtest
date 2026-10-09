"""alpha045 — standalone alpha factor.

Alpha #045 — Lagged-MA rank × short-corr × long/short-MA-corr rank, negated.

WorldQuant Formula (Kakushadze 2015, eq. 45)
--------------------------------------------
    (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
           rank(correlation(sum(close, 5), sum(close, 20), 2))))

Legacy AQML Expression
----------------------
    -1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) *
          Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))

Polars Implementation Notes
---------------------------
1. ``Ts_Sum(Delay(close, 5), 20) / 20`` -> per-stock lagged 20d MA.
2. Original WorldQuant formula uses ``Ts_Corr(.,.,2)`` (window=2) for both
   inner correlations. With only 2 observations, when either series has
   zero variance (limit-up day, suspended bar) polars ``rolling_corr``
   returned inf rather than nan, causing ~27k inf cells / year on real
   A-share data. **We widen window=2 → 5** in this implementation: this
   changes the numerical output but keeps the economic intent (recent
   price-volume covariation; recent short-vs-long MA covariation). The
   legacy expression is preserved above for parity reference.
3. Two CS ranks; materialise the lagged-MA and the long/short-MA
   correlation before ranking.

Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha045 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #045 — Lagged-MA rank × short-corr × long/short-MA-corr rank, negated.

    WorldQuant Formula (Kakushadze 2015, eq. 45)
    --------------------------------------------
        (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *
               rank(correlation(sum(close, 5), sum(close, 20), 2))))

    Legacy AQML Expression
    ----------------------
        -1 * (Rank(Ts_Sum(Delay(close, 5), 20) / 20) * Ts_Corr(close, volume, 2)) *
              Rank(Ts_Corr(Ts_Sum(close, 5), Ts_Sum(close, 20), 2))

    Polars Implementation Notes
    ---------------------------
    1. ``Ts_Sum(Delay(close, 5), 20) / 20`` -> per-stock lagged 20d MA.
    2. Original WorldQuant formula uses ``Ts_Corr(.,.,2)`` (window=2) for both
       inner correlations. With only 2 observations, when either series has
       zero variance (limit-up day, suspended bar) polars ``rolling_corr``
       returned inf rather than nan, causing ~27k inf cells / year on real
       A-share data. **We widen window=2 → 5** in this implementation: this
       changes the numerical output but keeps the economic intent (recent
       price-volume covariation; recent short-vs-long MA covariation). The
       legacy expression is preserved above for parity reference.
    3. Two CS ranks; materialise the lagged-MA and the long/short-MA
       correlation before ranking.

    Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    lagged_ma = ts_sum(delay(pl.col('close'), 5), 20) / 20.0
    short_corr = ts_corr(pl.col('close'), pl.col('volume'), 5)
    long_short_corr = ts_corr(ts_sum(pl.col('close'), 5), ts_sum(pl.col('close'), 20), 5)
    staged = panel.with_columns(lagged_ma.alias('__a045_ma'), short_corr.alias('__a045_sc'), long_short_corr.alias('__a045_lsc'))
    expr = -1.0 * (cs_rank(pl.col('__a045_ma')) * pl.col('__a045_sc')) * cs_rank(pl.col('__a045_lsc'))
    return staged.select(expr.alias('alpha045').cast(pl.Float64)).to_series()
