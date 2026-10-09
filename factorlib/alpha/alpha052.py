"""alpha052 — standalone alpha factor.

Alpha #052 — Low-shift × medium-term excess return rank × volume rank.

WorldQuant Formula (Kakushadze 2015, eq. 52)
--------------------------------------------
    ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
      rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))

Legacy AQML Expression
----------------------
    (-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) *
     Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) *
     Ts_Rank(volume, 5)

Polars Implementation Notes
---------------------------
1. ``-Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)`` measures the change
   in the rolling-min low over the last 5 days vs 5 days ago.
2. The medium-term excess return uses 240 - 20 = 220 day window of
   carry, scaled by 1/220.
3. Materialise the rank-input, then ``cs_rank``, then multiply with
   the TS components.

Required panel columns: ``low``, ``returns``, ``volume``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha052 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #052 — Low-shift × medium-term excess return rank × volume rank.

    WorldQuant Formula (Kakushadze 2015, eq. 52)
    --------------------------------------------
        ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) *
          rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))

    Legacy AQML Expression
    ----------------------
        (-1 * Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)) *
         Rank((Ts_Sum(returns, 240) - Ts_Sum(returns, 20)) / 220) *
         Ts_Rank(volume, 5)

    Polars Implementation Notes
    ---------------------------
    1. ``-Ts_Min(low, 5) + Delay(Ts_Min(low, 5), 5)`` measures the change
       in the rolling-min low over the last 5 days vs 5 days ago.
    2. The medium-term excess return uses 240 - 20 = 220 day window of
       carry, scaled by 1/220.
    3. Materialise the rank-input, then ``cs_rank``, then multiply with
       the TS components.

    Required panel columns: ``low``, ``returns``, ``volume``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    low_min5 = ts_min(pl.col('low'), 5)
    low_term = -1.0 * low_min5 + delay(low_min5, 5)
    excess = (ts_sum(pl.col('returns'), 240) - ts_sum(pl.col('returns'), 20)) / 220.0
    vol_tsr = ts_rank(pl.col('volume'), 5)
    staged = panel.with_columns(low_term.alias('__a052_low'), excess.alias('__a052_ex'), vol_tsr.alias('__a052_vt'))
    expr = pl.col('__a052_low') * cs_rank(pl.col('__a052_ex')) * pl.col('__a052_vt')
    return staged.select(expr.alias('alpha052').cast(pl.Float64)).to_series()
