"""alpha002 — standalone alpha factor.

Alpha #002 — Volume change rank vs intraday return rank correlation.

WorldQuant Formula
------------------
    -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)

Legacy AQML Expression
----------------------
    -1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)

Polars Implementation Notes
---------------------------
1. Inner ``Rank`` partitions by ``trade_date`` (CS); outer ``Ts_Corr``
   partitions by ``stock_code`` (TS). Materialise the two per-row
   ranked Series before correlating.

Required panel columns: ``volume``, ``close``, ``open``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha002 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #002 — Volume change rank vs intraday return rank correlation.

    WorldQuant Formula
    ------------------
        -1 * correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(Rank(Delta(Log(volume), 2)), Rank((close - open) / open), 6)

    Polars Implementation Notes
    ---------------------------
    1. Inner ``Rank`` partitions by ``trade_date`` (CS); outer ``Ts_Corr``
       partitions by ``stock_code`` (TS). Materialise the two per-row
       ranked Series before correlating.

    Required panel columns: ``volume``, ``close``, ``open``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged_ts = panel.with_columns(delta(log_(pl.col('volume')), 2).alias('__a002_dlogv'))
    staged = staged_ts.with_columns(cs_rank(pl.col('__a002_dlogv')).alias('__a002_rv'), cs_rank((pl.col('close') - pl.col('open')) / pl.col('open')).alias('__a002_rret'))
    return staged.select((-1.0 * ts_corr(pl.col('__a002_rv'), pl.col('__a002_rret'), 6)).alias('alpha002')).to_series()
