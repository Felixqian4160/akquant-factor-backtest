"""alpha074 — standalone alpha factor.

Alpha #074 — Close vs adv30-sum corr vs weighted-price-volume corr.

WorldQuant Formula
------------------
    if rank(correlation(close, sum(adv30, 37), 15))
          < rank(correlation(rank(high*0.0261 + vwap*0.9739), rank(volume), 11))
    then -1 else 0

Legacy AQML Expression
----------------------
    If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15))
            < Rank(Ts_Corr(Rank(high*0.0261 + vwap*0.9739), Rank(volume), 11)), -1, 0)

Polars Implementation Notes
---------------------------
AQML uses ``Ts_Sum``; STHSF uses ``sma`` (rolling mean) — STHSF parity
may diverge as for #065. False branch is 0 (not 1) per AQML.

Required panel columns: ``close``, ``adv30``, ``high``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha074 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #074 — Close vs adv30-sum corr vs weighted-price-volume corr.

    WorldQuant Formula
    ------------------
        if rank(correlation(close, sum(adv30, 37), 15))
              < rank(correlation(rank(high*0.0261 + vwap*0.9739), rank(volume), 11))
        then -1 else 0

    Legacy AQML Expression
    ----------------------
        If(Rank(Ts_Corr(close, Ts_Sum(adv30, 37), 15))
                < Rank(Ts_Corr(Rank(high*0.0261 + vwap*0.9739), Rank(volume), 11)), -1, 0)

    Polars Implementation Notes
    ---------------------------
    AQML uses ``Ts_Sum``; STHSF uses ``sma`` (rolling mean) — STHSF parity
    may diverge as for #065. False branch is 0 (not 1) per AQML.

    Required panel columns: ``close``, ``adv30``, ``high``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('high') * 0.0261 + pl.col('vwap') * 0.9739
    sum_adv30 = ts_sum(pl.col('adv30'), 37)
    staged = panel.with_columns(cs_rank(weighted).alias('__a074_rw'), cs_rank(pl.col('volume')).alias('__a074_rv'), ts_corr(pl.col('close'), sum_adv30, 15).alias('__a074_corr1'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a074_rw'), pl.col('__a074_rv'), 11).alias('__a074_corr2'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a074_corr1')).alias('__a074_rcorr1'), cs_rank(pl.col('__a074_corr2')).alias('__a074_rcorr2'))
    return staged3.select(if_then_else(pl.col('__a074_rcorr1') < pl.col('__a074_rcorr2'), -1.0, 0.0).cast(pl.Float64).alias('alpha074')).to_series()
