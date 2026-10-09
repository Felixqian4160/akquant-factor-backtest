"""alpha003 — standalone alpha factor.

Alpha #003 — Open price rank vs volume rank correlation.

WorldQuant Formula
------------------
    -1 * correlation(rank(open), rank(volume), 10)

Legacy AQML Expression
----------------------
    -1 * Ts_Corr(Rank(open), Rank(volume), 10)

Polars Implementation Notes
---------------------------
Two CS ranks materialised before TS corr.

Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha003 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #003 — Open price rank vs volume rank correlation.

    WorldQuant Formula
    ------------------
        -1 * correlation(rank(open), rank(volume), 10)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(Rank(open), Rank(volume), 10)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks materialised before TS corr.

    Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(cs_rank(pl.col('open')).alias('__a003_ro'), cs_rank(pl.col('volume')).alias('__a003_rv'))
    return staged.select((-1.0 * ts_corr(pl.col('__a003_ro'), pl.col('__a003_rv'), 10)).alias('alpha003')).to_series()
