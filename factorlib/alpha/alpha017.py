"""alpha017 — standalone alpha factor.

Alpha #017 — Momentum exhaustion: rank-mom × second-derivative × volume-surge.

WorldQuant Formula (Kakushadze 2015, eq. 17)
--------------------------------------------
    (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
     rank(ts_rank((volume / adv20), 5)))

Legacy AQML Expression
----------------------
    (-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) *
    Rank(Ts_Rank(volume / adv20, 5))

Polars Implementation Notes
---------------------------
1. Three components, each a CS rank of a TS quantity. We materialise
   the three TS columns first, then cross-section rank, then multiply.
2. ``delta(delta(close, 1), 1)`` is the discrete second derivative —
   acceleration of price.

Required panel columns: ``close``, ``volume``, ``adv20``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha017 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #017 — Momentum exhaustion: rank-mom × second-derivative × volume-surge.

    WorldQuant Formula (Kakushadze 2015, eq. 17)
    --------------------------------------------
        (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *
         rank(ts_rank((volume / adv20), 5)))

    Legacy AQML Expression
    ----------------------
        (-1 * Rank(Ts_Rank(close, 10))) * Rank(Delta(Delta(close, 1), 1)) *
        Rank(Ts_Rank(volume / adv20, 5))

    Polars Implementation Notes
    ---------------------------
    1. Three components, each a CS rank of a TS quantity. We materialise
       the three TS columns first, then cross-section rank, then multiply.
    2. ``delta(delta(close, 1), 1)`` is the discrete second derivative —
       acceleration of price.

    Required panel columns: ``close``, ``volume``, ``adv20``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    tsr_close = ts_rank(pl.col('close'), 10)
    accel = delta(delta(pl.col('close'), 1), 1)
    vol_surge = ts_rank(pl.col('volume') / pl.col('adv20'), 5)
    staged = panel.with_columns(tsr_close.alias('__a017_a'), accel.alias('__a017_b'), vol_surge.alias('__a017_c'))
    expr = -1.0 * cs_rank(pl.col('__a017_a')) * cs_rank(pl.col('__a017_b')) * cs_rank(pl.col('__a017_c'))
    return staged.select(expr.alias('alpha017').cast(pl.Float64)).to_series()
