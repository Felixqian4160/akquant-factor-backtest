"""alpha009 — standalone alpha factor.

Alpha #009 — Trend-confirmed price-change momentum.

WorldQuant Formula (Kakushadze 2015, eq. 9)
-------------------------------------------
    ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
     ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

Legacy AQML Expression
----------------------
    If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1),
       If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1),
          -1 * Delta(close, 1)))

Polars Implementation Notes
---------------------------
1. If the past 5-day minimum daily change is positive (consistent up
   trend), pass-through the daily delta.
2. Else if the past 5-day maximum daily change is negative (consistent
   down trend), still pass-through the daily delta.
3. Otherwise (mixed regime) flip sign of the daily delta — i.e. mean
   reversion within choppy markets.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha009 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #009 — Trend-confirmed price-change momentum.

    WorldQuant Formula (Kakushadze 2015, eq. 9)
    -------------------------------------------
        ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) :
         ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))

    Legacy AQML Expression
    ----------------------
        If(Ts_Min(Delta(close, 1), 5) > 0, Delta(close, 1),
           If(Ts_Max(Delta(close, 1), 5) < 0, Delta(close, 1),
              -1 * Delta(close, 1)))

    Polars Implementation Notes
    ---------------------------
    1. If the past 5-day minimum daily change is positive (consistent up
       trend), pass-through the daily delta.
    2. Else if the past 5-day maximum daily change is negative (consistent
       down trend), still pass-through the daily delta.
    3. Otherwise (mixed regime) flip sign of the daily delta — i.e. mean
       reversion within choppy markets.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d1 = delta(pl.col('close'), 1)
    inner = if_then_else(ts_max(d1, 5) < 0.0, d1, -1.0 * d1)
    expr = if_then_else(ts_min(d1, 5) > 0.0, d1, inner)
    return panel.select(expr.alias('alpha009').cast(pl.Float64)).to_series()
