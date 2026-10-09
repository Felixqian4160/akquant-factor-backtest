"""alpha007 — standalone alpha factor.

Alpha #007 — Volume-conditional 7-day signed momentum rank.

WorldQuant Formula (Kakushadze 2015, eq. 7)
-------------------------------------------
    ((adv20 < volume) ? -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1)

Legacy AQML Expression
----------------------
    If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * Sign(Delta(close, 7)), -1)

Polars Implementation Notes
---------------------------
1. ``delta(close, 7)`` is a per-stock 7-day price change.
2. ``ts_rank(abs(...), 60)`` is a per-stock 60-window rank in [0, 1].
3. The condition selects between the rank-based momentum reversal and
   a constant ``-1``. When volume is below ``adv20``, the alpha collapses
   to a flat ``-1`` (i.e. the day carries no signal apart from the
   constant).

Required panel columns: ``adv20``, ``volume``, ``close``, ``stock_code``,
``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha007 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #007 — Volume-conditional 7-day signed momentum rank.

    WorldQuant Formula (Kakushadze 2015, eq. 7)
    -------------------------------------------
        ((adv20 < volume) ? -1 * ts_rank(abs(delta(close, 7)), 60) * sign(delta(close, 7)) : -1)

    Legacy AQML Expression
    ----------------------
        If(adv20 < volume, -1 * Ts_Rank(Abs(Delta(close, 7)), 60) * Sign(Delta(close, 7)), -1)

    Polars Implementation Notes
    ---------------------------
    1. ``delta(close, 7)`` is a per-stock 7-day price change.
    2. ``ts_rank(abs(...), 60)`` is a per-stock 60-window rank in [0, 1].
    3. The condition selects between the rank-based momentum reversal and
       a constant ``-1``. When volume is below ``adv20``, the alpha collapses
       to a flat ``-1`` (i.e. the day carries no signal apart from the
       constant).

    Required panel columns: ``adv20``, ``volume``, ``close``, ``stock_code``,
    ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d7 = delta(pl.col('close'), 7)
    rank60 = ts_rank(d7.abs(), 60)
    branch = -1.0 * rank60 * sign_(d7)
    expr = if_then_else(pl.col('adv20') < pl.col('volume'), branch, pl.lit(-1.0))
    return panel.select(expr.alias('alpha007').cast(pl.Float64)).to_series()
