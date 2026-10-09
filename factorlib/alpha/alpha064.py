"""alpha064 — standalone alpha factor.

Alpha #064 — Open-low blend sum-vs-adv120-sum corr inequality with delta-blend rank.

WorldQuant Formula
------------------
    (rank(correlation(sum(open*0.178404 + low*(1-0.178404), 12.7054),
                      sum(adv120, 12.7054), 16.6208)) <
     rank(delta((((high+low)/2)*0.178404 + vwap*(1-0.178404), 3.69741))) * -1

Required panel columns: ``open``, ``low``, ``adv120``, ``high``, ``vwap``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/adv_extended.py

Usage:
    from factorlib.alpha.alpha064 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, ts_corr, ts_mean, ts_min, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #064 — Open-low blend sum-vs-adv120-sum corr inequality with delta-blend rank.

    WorldQuant Formula
    ------------------
        (rank(correlation(sum(open*0.178404 + low*(1-0.178404), 12.7054),
                          sum(adv120, 12.7054), 16.6208)) <
         rank(delta((((high+low)/2)*0.178404 + vwap*(1-0.178404), 3.69741))) * -1

    Required panel columns: ``open``, ``low``, ``adv120``, ``high``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    blend1 = pl.col('open') * 0.178404 + pl.col('low') * (1.0 - 0.178404)
    midpoint = (pl.col('high') + pl.col('low')) / 2.0
    blend2 = midpoint * 0.178404 + pl.col('vwap') * (1.0 - 0.178404)
    staged = panel.with_columns(ts_sum(blend1, 13).alias('__a064_s1'), ts_sum(pl.col('adv120'), 13).alias('__a064_s2'), (blend2 - blend2.shift(4).over(TS_PART)).alias('__a064_db'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a064_s1'), pl.col('__a064_s2'), 17).alias('__a064_corr'))
    return staged2.select(((cs_rank(pl.col('__a064_corr')) < cs_rank(pl.col('__a064_db'))).cast(pl.Float64) * -1.0).alias('alpha064')).to_series()
