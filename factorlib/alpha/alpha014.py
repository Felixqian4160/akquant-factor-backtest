"""alpha014 — standalone alpha factor.

Alpha #014 — Returns-acceleration rank scaled by open-volume correlation.

WorldQuant Formula
------------------
    (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

Legacy AQML Expression
----------------------
    (-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)

Polars Implementation Notes
---------------------------
Inner Delta is TS, then CS rank, then outer multiplied by TS corr.
We stage the delta to make the CS rank pure.

Required panel columns: ``returns``, ``open``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha014 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #014 — Returns-acceleration rank scaled by open-volume correlation.

    WorldQuant Formula
    ------------------
        (-1 * rank(delta(returns, 3))) * correlation(open, volume, 10)

    Legacy AQML Expression
    ----------------------
        (-1 * Rank(Delta(returns, 3))) * Ts_Corr(open, volume, 10)

    Polars Implementation Notes
    ---------------------------
    Inner Delta is TS, then CS rank, then outer multiplied by TS corr.
    We stage the delta to make the CS rank pure.

    Required panel columns: ``returns``, ``open``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(delta(pl.col('returns'), 3).alias('__a014_dret'))
    return staged.select((-1.0 * cs_rank(pl.col('__a014_dret')) * ts_corr(pl.col('open'), pl.col('volume'), 10)).alias('alpha014')).to_series()
