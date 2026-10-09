"""alpha011 — standalone alpha factor.

Alpha #011 — VWAP-close range bookended by volume change rank.

WorldQuant Formula
------------------
    (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
    rank(delta(volume, 3))

Polars Implementation Notes
---------------------------
Stage the per-stock ts_max/ts_min/delta first, then take three CS
ranks on the materialised columns and combine.

Required panel columns: ``vwap``, ``close``, ``volume``, ``stock_code``,
``trade_date``, ``sub_industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha011 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #011 — VWAP-close range bookended by volume change rank.

    WorldQuant Formula
    ------------------
        (rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *
        rank(delta(volume, 3))

    Polars Implementation Notes
    ---------------------------
    Stage the per-stock ts_max/ts_min/delta first, then take three CS
    ranks on the materialised columns and combine.

    Required panel columns: ``vwap``, ``close``, ``volume``, ``stock_code``,
    ``trade_date``, ``sub_industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    diff = pl.col('vwap') - pl.col('close')
    staged = panel.with_columns(ts_max(diff, 3).alias('__a011_max'), ts_min(diff, 3).alias('__a011_min'), delta(pl.col('volume'), 3).alias('__a011_dv'))
    return staged.select(((cs_rank(pl.col('__a011_max')) + cs_rank(pl.col('__a011_min'))) * cs_rank(pl.col('__a011_dv'))).alias('alpha011')).to_series()
