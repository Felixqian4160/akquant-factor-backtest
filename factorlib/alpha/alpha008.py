"""alpha008 — standalone alpha factor.

Alpha #008 — Acceleration of (open · returns) sum compared to 10 days ago.

WorldQuant Formula (Kakushadze 2015, eq. 8)
-------------------------------------------
    -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))

Legacy AQML Expression
----------------------
    -1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))

Polars Implementation Notes
---------------------------
1. Build the per-stock 5-day rolling sum of ``open`` and ``returns``.
2. Multiply pointwise -> momentum proxy.
3. Subtract the 10-day-ago version -> acceleration.
4. Materialise before ``cs_rank`` (CS partition differs from TS).

Required panel columns: ``open``, ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha008 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #008 — Acceleration of (open · returns) sum compared to 10 days ago.

    WorldQuant Formula (Kakushadze 2015, eq. 8)
    -------------------------------------------
        -1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10)))

    Legacy AQML Expression
    ----------------------
        -1 * Rank((Ts_Sum(open, 5) * Ts_Sum(returns, 5)) - Delay(Ts_Sum(open, 5) * Ts_Sum(returns, 5), 10))

    Polars Implementation Notes
    ---------------------------
    1. Build the per-stock 5-day rolling sum of ``open`` and ``returns``.
    2. Multiply pointwise -> momentum proxy.
    3. Subtract the 10-day-ago version -> acceleration.
    4. Materialise before ``cs_rank`` (CS partition differs from TS).

    Required panel columns: ``open``, ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    sum_open = ts_sum(pl.col('open'), 5)
    sum_ret = ts_sum(pl.col('returns'), 5)
    product = sum_open * sum_ret
    accel = product - delay(product, 10)
    staged = panel.with_columns(accel.alias('__a008_accel'))
    return staged.select((-1.0 * cs_rank(pl.col('__a008_accel'))).alias('alpha008').cast(pl.Float64)).to_series()
