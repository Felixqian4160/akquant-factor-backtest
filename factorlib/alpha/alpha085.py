"""alpha085 — standalone alpha factor.

Alpha #085 — Power composition of weighted-price/adv30 and rank-rank corrs.

WorldQuant Formula
------------------
    rank(correlation(high*0.876 + close*0.124, adv30, 10))
    ^ rank(correlation(ts_rank((high+low)/2, 4), ts_rank(volume, 10), 7))

Legacy AQML Expression
----------------------
    Power(
      Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)),
      Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7))
    )

Polars Implementation Notes
---------------------------
Two CS ranks form base/exponent of pow.

Required panel columns: ``high``, ``close``, ``adv30``, ``low``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha085 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #085 — Power composition of weighted-price/adv30 and rank-rank corrs.

    WorldQuant Formula
    ------------------
        rank(correlation(high*0.876 + close*0.124, adv30, 10))
        ^ rank(correlation(ts_rank((high+low)/2, 4), ts_rank(volume, 10), 7))

    Legacy AQML Expression
    ----------------------
        Power(
          Rank(Ts_Corr(high * 0.876 + close * 0.124, adv30, 10)),
          Rank(Ts_Corr(Ts_Rank((high + low) / 2, 4), Ts_Rank(volume, 10), 7))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS ranks form base/exponent of pow.

    Required panel columns: ``high``, ``close``, ``adv30``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('high') * 0.876 + pl.col('close') * 0.124
    mid = (pl.col('high') + pl.col('low')) / 2.0
    staged = panel.with_columns(ts_corr(weighted, pl.col('adv30'), 10).alias('__a085_corr1'), ts_rank(mid, 4).alias('__a085_trmid'), ts_rank(pl.col('volume'), 10).alias('__a085_trv'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a085_trmid'), pl.col('__a085_trv'), 7).alias('__a085_corr2'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a085_corr1')).alias('__a085_base'), cs_rank(pl.col('__a085_corr2')).alias('__a085_exp'))
    return staged3.select(power(pl.col('__a085_base'), pl.col('__a085_exp')).alias('alpha085')).to_series()
