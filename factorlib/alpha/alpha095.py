"""alpha095 — standalone alpha factor.

Alpha #095 — Open-trough rank vs medium-term correlation rank-power.

WorldQuant Formula (Kakushadze 2015, eq. 95)
--------------------------------------------
    (rank((open - ts_min(open, 12.4105))) <
     Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351),
                                sum(adv40, 19.1351), 12.8742))^5), 11.7584))

Legacy AQML Expression (windows rounded to integers)
---------------------------------------------------
    If(Rank(open - Ts_Min(open, 12)) <
       Ts_Rank(Power(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19),
                                  Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)

Polars Implementation Notes
---------------------------
1. Left side: cross-section rank of ``open - ts_min(open, 12)``
   (how high open is above its 12d trough).
2. Right side: rolling rank of (rank-of-corr ** 5) — a heavy-tailed
   indicator of how unusual the medium-term correlation between
   ``hl2`` sums and ``adv40`` sums has been over the last 12 days.
3. Two cross-section ranks need staging; the final boolean cast
   returns the WorldQuant 0/1 flag.

Required panel columns: ``open``, ``high``, ``low``, ``adv40``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``breakout``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/breakout.py

Usage:
    from factorlib.alpha.alpha095 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delta, if_then_else, ts_corr, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #095 — Open-trough rank vs medium-term correlation rank-power.

    WorldQuant Formula (Kakushadze 2015, eq. 95)
    --------------------------------------------
        (rank((open - ts_min(open, 12.4105))) <
         Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351),
                                    sum(adv40, 19.1351), 12.8742))^5), 11.7584))

    Legacy AQML Expression (windows rounded to integers)
    ---------------------------------------------------
        If(Rank(open - Ts_Min(open, 12)) <
           Ts_Rank(Power(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19),
                                      Ts_Sum(adv40, 19), 13)), 5), 12), 1, 0)

    Polars Implementation Notes
    ---------------------------
    1. Left side: cross-section rank of ``open - ts_min(open, 12)``
       (how high open is above its 12d trough).
    2. Right side: rolling rank of (rank-of-corr ** 5) — a heavy-tailed
       indicator of how unusual the medium-term correlation between
       ``hl2`` sums and ``adv40`` sums has been over the last 12 days.
    3. Two cross-section ranks need staging; the final boolean cast
       returns the WorldQuant 0/1 flag.

    Required panel columns: ``open``, ``high``, ``low``, ``adv40``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``breakout``
    """
    left_input = pl.col('open') - ts_min(pl.col('open'), 12)
    hl2 = (pl.col('high') + pl.col('low')) / 2.0
    sum_hl2_19 = ts_sum(hl2, 19)
    sum_adv40_19 = ts_sum(pl.col('adv40'), 19)
    corr = ts_corr(sum_hl2_19, sum_adv40_19, 13)
    staged = panel.with_columns(left_input.alias('__a095_left'), corr.alias('__a095_corr'))
    staged = staged.with_columns(cs_rank(pl.col('__a095_left')).alias('__a095_left_rank'), cs_rank(pl.col('__a095_corr')).pow(5).alias('__a095_corr_rank5'))
    right = ts_rank(pl.col('__a095_corr_rank5'), 12)
    staged = staged.with_columns(right.alias('__a095_right'))
    expr = if_then_else(pl.col('__a095_left_rank') < pl.col('__a095_right'), pl.lit(1.0), pl.lit(0.0))
    return staged.select(expr.alias('alpha095').cast(pl.Float64)).to_series()
