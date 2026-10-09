"""alpha078 — standalone alpha factor.

Alpha #078 — Power composition of two correlation ranks.

WorldQuant Formula
------------------
    rank(correlation(sum(low*0.352 + vwap*0.648, 20), sum(adv40, 20), 7))
    ^ rank(correlation(rank(vwap), rank(volume), 6))

Legacy AQML Expression
----------------------
    Power(
      Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), Ts_Sum(adv40, 20), 7)),
      Rank(Ts_Corr(Rank(vwap), Rank(volume), 6))
    )

Polars Implementation Notes
---------------------------
Two CS-rank values produce the (base, exponent) pair for ``Power``.

Required panel columns: ``low``, ``vwap``, ``adv40``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha078 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #078 — Power composition of two correlation ranks.

    WorldQuant Formula
    ------------------
        rank(correlation(sum(low*0.352 + vwap*0.648, 20), sum(adv40, 20), 7))
        ^ rank(correlation(rank(vwap), rank(volume), 6))

    Legacy AQML Expression
    ----------------------
        Power(
          Rank(Ts_Corr(Ts_Sum(low * 0.352 + vwap * 0.648, 20), Ts_Sum(adv40, 20), 7)),
          Rank(Ts_Corr(Rank(vwap), Rank(volume), 6))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS-rank values produce the (base, exponent) pair for ``Power``.

    Required panel columns: ``low``, ``vwap``, ``adv40``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('low') * 0.352 + pl.col('vwap') * 0.648
    sum_w = ts_sum(weighted, 20)
    sum_adv40 = ts_sum(pl.col('adv40'), 20)
    staged = panel.with_columns(cs_rank(pl.col('vwap')).alias('__a078_rv'), cs_rank(pl.col('volume')).alias('__a078_rvol'), ts_corr(sum_w, sum_adv40, 7).alias('__a078_corr1'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a078_rv'), pl.col('__a078_rvol'), 6).alias('__a078_corr2'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a078_corr1')).alias('__a078_base'), cs_rank(pl.col('__a078_corr2')).alias('__a078_exp'))
    return staged3.select(power(pl.col('__a078_base'), pl.col('__a078_exp')).alias('alpha078')).to_series()
