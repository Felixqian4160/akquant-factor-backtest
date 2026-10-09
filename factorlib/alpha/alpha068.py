"""alpha068 — standalone alpha factor.

Alpha #068 — Composite price/adv15 rank vs weighted-price delta.

WorldQuant Formula
------------------
    if ts_rank(correlation(rank(high), rank(adv15), 9), 14)
          < rank(delta(close*0.518 + low*0.482, 1))
    then -1 else 1

Legacy AQML Expression
----------------------
    If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14)
            < Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)

Polars Implementation Notes
---------------------------
Stage CS rank of high and adv15 → TS corr → TS rank → compare with
CS rank of TS delta of weighted price.

Required panel columns: ``high``, ``adv15``, ``close``, ``low``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha068 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #068 — Composite price/adv15 rank vs weighted-price delta.

    WorldQuant Formula
    ------------------
        if ts_rank(correlation(rank(high), rank(adv15), 9), 14)
              < rank(delta(close*0.518 + low*0.482, 1))
        then -1 else 1

    Legacy AQML Expression
    ----------------------
        If(Ts_Rank(Ts_Corr(Rank(high), Rank(adv15), 9), 14)
                < Rank(Delta(close * 0.518 + low * 0.482, 1)), -1, 1)

    Polars Implementation Notes
    ---------------------------
    Stage CS rank of high and adv15 → TS corr → TS rank → compare with
    CS rank of TS delta of weighted price.

    Required panel columns: ``high``, ``adv15``, ``close``, ``low``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(cs_rank(pl.col('high')).alias('__a068_rh'), cs_rank(pl.col('adv15')).alias('__a068_ra'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a068_rh'), pl.col('__a068_ra'), 9).alias('__a068_corr'), delta(pl.col('close') * 0.518 + pl.col('low') * 0.482, 1).alias('__a068_dpx'))
    staged3 = staged2.with_columns(ts_rank(pl.col('__a068_corr'), 14).alias('__a068_trc'), cs_rank(pl.col('__a068_dpx')).alias('__a068_rdpx'))
    return staged3.select(if_then_else(pl.col('__a068_trc') < pl.col('__a068_rdpx'), -1.0, 1.0).cast(pl.Float64).alias('alpha068')).to_series()
