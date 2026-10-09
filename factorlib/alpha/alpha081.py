"""alpha081 — standalone alpha factor.

Alpha #081 — Log-product of double-rank corr vs vwap-volume corr.

WorldQuant Formula
------------------
    if rank(log(product(rank(rank(correlation(vwap, sum(adv10, 50), 8)^4)), 15)))
          < rank(correlation(rank(vwap), rank(volume), 5))
    then -1 else 0

Legacy AQML Expression
----------------------
    If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, Ts_Sum(adv10, 50), 8), 4))), 15)))
            < Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)

Polars Implementation Notes
---------------------------
Heavy nested expression. Stage step-by-step.

Required panel columns: ``vwap``, ``adv10``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha081 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #081 — Log-product of double-rank corr vs vwap-volume corr.

    WorldQuant Formula
    ------------------
        if rank(log(product(rank(rank(correlation(vwap, sum(adv10, 50), 8)^4)), 15)))
              < rank(correlation(rank(vwap), rank(volume), 5))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Log(Ts_Product(Rank(Rank(Power(Ts_Corr(vwap, Ts_Sum(adv10, 50), 8), 4))), 15)))
                < Rank(Ts_Corr(Rank(vwap), Rank(volume), 5)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    Heavy nested expression. Stage step-by-step.

    Required panel columns: ``vwap``, ``adv10``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    sum_adv10 = ts_sum(pl.col('adv10'), 50)
    staged = panel.with_columns(ts_corr(pl.col('vwap'), sum_adv10, 8).alias('__a081_corr1'), cs_rank(pl.col('vwap')).alias('__a081_rv'), cs_rank(pl.col('volume')).alias('__a081_rvol'))
    staged2 = staged.with_columns(pl.col('__a081_corr1').pow(4).alias('__a081_corr1_p4'), ts_corr(pl.col('__a081_rv'), pl.col('__a081_rvol'), 5).alias('__a081_corr2'))
    staged3 = staged2.with_columns(cs_rank(cs_rank(pl.col('__a081_corr1_p4'))).alias('__a081_rr'))
    staged4 = staged3.with_columns(ts_product(pl.col('__a081_rr'), 15).alias('__a081_prod'))
    staged5 = staged4.with_columns(cs_rank(log_(pl.col('__a081_prod'))).alias('__a081_lhs'), cs_rank(pl.col('__a081_corr2')).alias('__a081_rhs'))
    return staged5.select(if_then_else(pl.col('__a081_lhs') < pl.col('__a081_rhs'), -1.0, 0.0).cast(pl.Float64).alias('alpha081')).to_series()
