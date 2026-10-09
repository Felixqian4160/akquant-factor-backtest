"""alpha035 — standalone alpha factor.

Alpha #035 — Volume rank × inverse range rank × inverse returns rank.

WorldQuant Formula
------------------
    ts_rank(volume, 32)
    * (1 - ts_rank(close + high - low, 16))
    * (1 - ts_rank(returns, 32))

Legacy AQML Expression
----------------------
    Ts_Rank(volume, 32)
    * (1 - Ts_Rank(close + high - low, 16))
    * (1 - Ts_Rank(returns, 32))

Polars Implementation Notes
---------------------------
Pure TS chain. Three rolling ranks combined multiplicatively.

Required panel columns: ``volume``, ``close``, ``high``, ``low``, ``returns``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha035 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #035 — Volume rank × inverse range rank × inverse returns rank.

    WorldQuant Formula
    ------------------
        ts_rank(volume, 32)
        * (1 - ts_rank(close + high - low, 16))
        * (1 - ts_rank(returns, 32))

    Legacy AQML Expression
    ----------------------
        Ts_Rank(volume, 32)
        * (1 - Ts_Rank(close + high - low, 16))
        * (1 - Ts_Rank(returns, 32))

    Polars Implementation Notes
    ---------------------------
    Pure TS chain. Three rolling ranks combined multiplicatively.

    Required panel columns: ``volume``, ``close``, ``high``, ``low``, ``returns``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select((ts_rank(pl.col('volume'), 32) * (1.0 - ts_rank(pl.col('close') + pl.col('high') - pl.col('low'), 16)) * (1.0 - ts_rank(pl.col('returns'), 32))).alias('alpha035')).to_series()
