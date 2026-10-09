"""alpha075 — standalone alpha factor.

Alpha #075 — vwap-volume corr rank inequality with low-adv50 rank corr.

WorldQuant Formula
------------------
    rank(correlation(vwap, volume, 4.24304)) <
    rank(correlation(rank(low), rank(adv50), 12.4413))

Required panel columns: ``vwap``, ``volume``, ``low``, ``adv50``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/adv_extended.py

Usage:
    from factorlib.alpha.alpha075 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, ts_corr, ts_mean, ts_min, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #075 — vwap-volume corr rank inequality with low-adv50 rank corr.

    WorldQuant Formula
    ------------------
        rank(correlation(vwap, volume, 4.24304)) <
        rank(correlation(rank(low), rank(adv50), 12.4413))

    Required panel columns: ``vwap``, ``volume``, ``low``, ``adv50``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    staged = panel.with_columns(ts_corr(pl.col('vwap'), pl.col('volume'), 4).alias('__a075_c1'), cs_rank(pl.col('low')).alias('__a075_rl'), cs_rank(pl.col('adv50')).alias('__a075_ra'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a075_rl'), pl.col('__a075_ra'), 12).alias('__a075_c2'))
    return staged2.select((cs_rank(pl.col('__a075_c1')) < cs_rank(pl.col('__a075_c2'))).cast(pl.Float64).alias('alpha075')).to_series()
