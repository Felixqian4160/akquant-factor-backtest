"""alpha061 — standalone alpha factor.

Alpha #061 — Vwap-from-min rank inequality with vwap-adv180 corr rank.

WorldQuant Formula
------------------
    rank(vwap - ts_min(vwap, 16.1219)) <
    rank(correlation(vwap, adv180, 17.9282))

Required panel columns: ``vwap``, ``adv180``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/adv_extended.py

Usage:
    from factorlib.alpha.alpha061 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, ts_corr, ts_mean, ts_min, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #061 — Vwap-from-min rank inequality with vwap-adv180 corr rank.

    WorldQuant Formula
    ------------------
        rank(vwap - ts_min(vwap, 16.1219)) <
        rank(correlation(vwap, adv180, 17.9282))

    Required panel columns: ``vwap``, ``adv180``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    staged = panel.with_columns((pl.col('vwap') - ts_min(pl.col('vwap'), 16)).alias('__a061_diff'), ts_corr(pl.col('vwap'), pl.col('adv180'), 18).alias('__a061_corr'))
    return staged.select((cs_rank(pl.col('__a061_diff')) < cs_rank(pl.col('__a061_corr'))).cast(pl.Float64).alias('alpha061')).to_series()
