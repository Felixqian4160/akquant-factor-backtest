"""alpha_custom_decaylinear_mom — standalone alpha factor.

Alpha custom — Decay-linear-weighted 10d momentum rank.

Project-internal custom factor (NOT in WorldQuant 101 paper).

Legacy AQML Expression
----------------------
    Rank(Ts_DecayLinear(returns, 10))

Polars Implementation Notes
---------------------------
1. ``ts_decay_linear(returns, 10)`` -> per-stock weighted MA of returns
   with linearly decaying weights ``[10, 9, ..., 1] / 55``.
2. Cross-section rank (``cs_rank``) the materialised column. Two
   partitions -> stage first.

Required panel columns: ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha_custom_decaylinear_mom import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha custom — Decay-linear-weighted 10d momentum rank.

    Project-internal custom factor (NOT in WorldQuant 101 paper).

    Legacy AQML Expression
    ----------------------
        Rank(Ts_DecayLinear(returns, 10))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_decay_linear(returns, 10)`` -> per-stock weighted MA of returns
       with linearly decaying weights ``[10, 9, ..., 1] / 55``.
    2. Cross-section rank (``cs_rank``) the materialised column. Two
       partitions -> stage first.

    Required panel columns: ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    decay = ts_decay_linear(pl.col('returns'), 10)
    staged = panel.with_columns(decay.alias('__a_custom_dl_mom'))
    return staged.select(cs_rank(pl.col('__a_custom_dl_mom')).alias('alpha_custom_decaylinear_mom').cast(pl.Float64)).to_series()
