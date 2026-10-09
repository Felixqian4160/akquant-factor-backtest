"""alpha_custom_argmax_recent — standalone alpha factor.

Alpha custom — Inverse rank of days-since-20d-max (recency-of-peak).

Project-internal custom factor (NOT in WorldQuant 101 paper).

Legacy AQML Expression
----------------------
    1 - Rank(Ts_ArgMax(close, 20))

Polars Implementation Notes
---------------------------
1. ``ts_argmax(close, 20)`` -> position 0..19 of the highest close in
   last 20 days; high values ⇒ peak was recent (today the peak).
2. ``1 - cs_rank(...)`` flips the rank so that recent peaks score high.
   Materialise the argmax column before ranking.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha_custom_argmax_recent import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha custom — Inverse rank of days-since-20d-max (recency-of-peak).

    Project-internal custom factor (NOT in WorldQuant 101 paper).

    Legacy AQML Expression
    ----------------------
        1 - Rank(Ts_ArgMax(close, 20))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_argmax(close, 20)`` -> position 0..19 of the highest close in
       last 20 days; high values ⇒ peak was recent (today the peak).
    2. ``1 - cs_rank(...)`` flips the rank so that recent peaks score high.
       Materialise the argmax column before ranking.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    arg = ts_argmax(pl.col('close'), 20)
    staged = panel.with_columns(arg.alias('__a_custom_arg_recent'))
    return staged.select((1.0 - cs_rank(pl.col('__a_custom_arg_recent'))).alias('alpha_custom_argmax_recent').cast(pl.Float64)).to_series()
