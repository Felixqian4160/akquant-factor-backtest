"""alpha099 — standalone alpha factor.

Alpha #099 — Mid-price-adv60 corr vs low-volume corr.

WorldQuant Formula
------------------
    if rank(correlation(sum((high+low)/2, 19), sum(adv60, 19), 8))
          < rank(correlation(low, volume, 6))
    then -1 else 0

Legacy AQML Expression
----------------------
    If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8))
            < Rank(Ts_Corr(low, volume, 6)), -1, 0)

Polars Implementation Notes
---------------------------
AQML ``Ts_Sum`` vs STHSF ``sma`` again — STHSF parity may diverge for
constant-factor reasons. False branch is 0.

Required panel columns: ``high``, ``low``, ``adv60``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha099 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #099 — Mid-price-adv60 corr vs low-volume corr.

    WorldQuant Formula
    ------------------
        if rank(correlation(sum((high+low)/2, 19), sum(adv60, 19), 8))
              < rank(correlation(low, volume, 6))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(Ts_Sum((high + low) / 2, 19), Ts_Sum(adv60, 19), 8))
                < Rank(Ts_Corr(low, volume, 6)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    AQML ``Ts_Sum`` vs STHSF ``sma`` again — STHSF parity may diverge for
    constant-factor reasons. False branch is 0.

    Required panel columns: ``high``, ``low``, ``adv60``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    mid = (pl.col('high') + pl.col('low')) / 2.0
    sum_mid = ts_sum(mid, 19)
    sum_adv60 = ts_sum(pl.col('adv60'), 19)
    staged = panel.with_columns(ts_corr(sum_mid, sum_adv60, 8).alias('__a099_corr1'), ts_corr(pl.col('low'), pl.col('volume'), 6).alias('__a099_corr2'))
    staged2 = staged.with_columns(cs_rank(pl.col('__a099_corr1')).alias('__a099_r1'), cs_rank(pl.col('__a099_corr2')).alias('__a099_r2'))
    return staged2.select(if_then_else(pl.col('__a099_r1') < pl.col('__a099_r2'), -1.0, 0.0).cast(pl.Float64).alias('alpha099')).to_series()
