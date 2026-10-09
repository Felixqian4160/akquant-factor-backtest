"""alpha038 — standalone alpha factor.

Alpha #038 — Rank of 10d close rolling rank times close-over-open rank, negated.

WorldQuant Formula (Kakushadze 2015, eq. 38)
--------------------------------------------
    ((-1 * rank(ts_rank(close, 10))) * rank((close / open)))

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)

Polars Implementation Notes
---------------------------
1. Two CS ranks multiplied; each consumes a TS-column input. Materialise
   intermediates so the two ``cs_rank`` partitions don't collide.

Required panel columns: ``close``, ``open``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha038 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #038 — Rank of 10d close rolling rank times close-over-open rank, negated.

    WorldQuant Formula (Kakushadze 2015, eq. 38)
    --------------------------------------------
        ((-1 * rank(ts_rank(close, 10))) * rank((close / open)))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Rank(close, 10)) * Rank(close / open)

    Polars Implementation Notes
    ---------------------------
    1. Two CS ranks multiplied; each consumes a TS-column input. Materialise
       intermediates so the two ``cs_rank`` partitions don't collide.

    Required panel columns: ``close``, ``open``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    tsr_close = ts_rank(pl.col('close'), 10)
    staged = panel.with_columns(tsr_close.alias('__a038_tsr'))
    expr = -1.0 * cs_rank(pl.col('__a038_tsr')) * cs_rank(pl.col('close') / pl.col('open'))
    return staged.select(expr.alias('alpha038').cast(pl.Float64)).to_series()
