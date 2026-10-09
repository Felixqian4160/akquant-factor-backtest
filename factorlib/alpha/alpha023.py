"""alpha023 — standalone alpha factor.

Alpha #023 — High-breakout-conditional negative 2d high change.

WorldQuant Formula (Kakushadze 2015, eq. 23)
--------------------------------------------
    (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

Legacy AQML Expression
----------------------
    If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)

Polars Implementation Notes
---------------------------
1. The condition flags any day whose ``high`` exceeds the 20-day
   average — a breakout candidate.
2. On such days the alpha equals the **negative** 2-day change of
   ``high``: rising momentum -> negative signal, falling -> positive.
   On non-breakout days the alpha is zero (no opinion).

Required panel columns: ``high``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``breakout``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/breakout.py

Usage:
    from factorlib.alpha.alpha023 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, if_then_else, ts_corr, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #023 — High-breakout-conditional negative 2d high change.

    WorldQuant Formula (Kakushadze 2015, eq. 23)
    --------------------------------------------
        (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)

    Legacy AQML Expression
    ----------------------
        If(Ts_Sum(high, 20) / 20 < high, -1 * Delta(high, 2), 0)

    Polars Implementation Notes
    ---------------------------
    1. The condition flags any day whose ``high`` exceeds the 20-day
       average — a breakout candidate.
    2. On such days the alpha equals the **negative** 2-day change of
       ``high``: rising momentum -> negative signal, falling -> positive.
       On non-breakout days the alpha is zero (no opinion).

    Required panel columns: ``high``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    avg20_high = ts_sum(pl.col('high'), 20) / 20.0
    delta_high2 = delta(pl.col('high'), 2)
    expr = if_then_else(avg20_high < pl.col('high'), -1.0 * delta_high2, pl.lit(0.0))
    return panel.select(expr.alias('alpha023').cast(pl.Float64)).to_series()
