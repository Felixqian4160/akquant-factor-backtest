"""alpha047 — standalone alpha factor.

Alpha #047 — Inverse-close * volume / adv20 amplification minus VWAP momentum.

WorldQuant Formula
------------------
    ((((rank(1/close) * volume) / adv20) *
      ((high * rank(high - close)) / (sum(high, 5) / 5))) -
     rank(vwap - delay(vwap, 5)))

Required panel columns: ``close``, ``volume``, ``adv20``, ``high``,
``vwap``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha047 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #047 — Inverse-close * volume / adv20 amplification minus VWAP momentum.

    WorldQuant Formula
    ------------------
        ((((rank(1/close) * volume) / adv20) *
          ((high * rank(high - close)) / (sum(high, 5) / 5))) -
         rank(vwap - delay(vwap, 5)))

    Required panel columns: ``close``, ``volume``, ``adv20``, ``high``,
    ``vwap``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    pre = panel.with_columns(ts_mean(pl.col('high'), 5).alias('__a047_h5m'), (pl.col('vwap') - delay(pl.col('vwap'), 5)).alias('__a047_vw_d'))
    staged = pre.with_columns(cs_rank(1.0 / pl.col('close')).alias('__a047_rinv'), cs_rank(pl.col('high') - pl.col('close')).alias('__a047_rhc'), cs_rank(pl.col('__a047_vw_d')).alias('__a047_rvw'))
    return staged.select((pl.col('__a047_rinv') * pl.col('volume') / pl.col('adv20') * (pl.col('high') * pl.col('__a047_rhc') / pl.col('__a047_h5m')) - pl.col('__a047_rvw')).alias('alpha047')).to_series()
