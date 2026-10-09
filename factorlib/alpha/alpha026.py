"""alpha026 — standalone alpha factor.

Alpha #026 — Max recent volume-rank vs high-rank correlation.

WorldQuant Formula
------------------
    -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

Legacy AQML Expression
----------------------
    -1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)

Polars Implementation Notes
---------------------------
Pure TS chain. Stage the two TS ranks before the corr.

Required panel columns: ``volume``, ``high``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha026 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #026 — Max recent volume-rank vs high-rank correlation.

    WorldQuant Formula
    ------------------
        -1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Max(Ts_Corr(Ts_Rank(volume, 5), Ts_Rank(high, 5), 5), 3)

    Polars Implementation Notes
    ---------------------------
    Pure TS chain. Stage the two TS ranks before the corr.

    Required panel columns: ``volume``, ``high``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(ts_rank(pl.col('volume'), 5).alias('__a026_rv'), ts_rank(pl.col('high'), 5).alias('__a026_rh'))
    corr = ts_corr(pl.col('__a026_rv'), pl.col('__a026_rh'), 5)
    return staged.select((-1.0 * ts_max(corr, 3)).alias('alpha026')).to_series()
