"""alpha046 — standalone alpha factor.

Alpha #046 — Trend-shape conditional one-day reversal.

WorldQuant Formula (Kakushadze 2015, eq. 46)
--------------------------------------------
    ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
              ((delay(close, 10) - close) / 10))) ? -1 :
     ((((delay(close, 20) - delay(close, 10)) / 10) -
       ((delay(close, 10) - close) / 10)) < 0) ? 1 :
     (-1 * (close - delay(close, 1))))

Legacy AQML Expression
----------------------
    If((Delay(close, 20) - Delay(close, 10)) / 10 -
       (Delay(close, 10) - close) / 10 > 0.25, -1,
       If((Delay(close, 20) - Delay(close, 10)) / 10 -
          (Delay(close, 10) - close) / 10 < 0, 1,
          -1 * (close - Delay(close, 1))))

Polars Implementation Notes
---------------------------
1. Compute the ``trend curvature`` once and reuse the materialised
   column twice in the nested ``if``.
2. Three branches: strong upward curvature -> -1; downward curvature
   -> 1; otherwise reverse the daily change.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha046 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #046 — Trend-shape conditional one-day reversal.

    WorldQuant Formula (Kakushadze 2015, eq. 46)
    --------------------------------------------
        ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) -
                  ((delay(close, 10) - close) / 10))) ? -1 :
         ((((delay(close, 20) - delay(close, 10)) / 10) -
           ((delay(close, 10) - close) / 10)) < 0) ? 1 :
         (-1 * (close - delay(close, 1))))

    Legacy AQML Expression
    ----------------------
        If((Delay(close, 20) - Delay(close, 10)) / 10 -
           (Delay(close, 10) - close) / 10 > 0.25, -1,
           If((Delay(close, 20) - Delay(close, 10)) / 10 -
              (Delay(close, 10) - close) / 10 < 0, 1,
              -1 * (close - Delay(close, 1))))

    Polars Implementation Notes
    ---------------------------
    1. Compute the ``trend curvature`` once and reuse the materialised
       column twice in the nested ``if``.
    2. Three branches: strong upward curvature -> -1; downward curvature
       -> 1; otherwise reverse the daily change.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    seg_a = (delay(pl.col('close'), 20) - delay(pl.col('close'), 10)) / 10.0
    seg_b = (delay(pl.col('close'), 10) - pl.col('close')) / 10.0
    curvature = seg_a - seg_b
    inner = if_then_else(curvature < 0.0, pl.lit(1.0), -1.0 * (pl.col('close') - delay(pl.col('close'), 1)))
    expr = if_then_else(curvature > 0.25, pl.lit(-1.0), inner)
    return panel.select(expr.alias('alpha046').cast(pl.Float64)).to_series()
