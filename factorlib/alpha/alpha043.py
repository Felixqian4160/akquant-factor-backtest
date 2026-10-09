"""alpha043 — standalone alpha factor.

Alpha #043 — Volume-surge rank × 7d-decline rank.

WorldQuant Formula
------------------
    ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

Legacy AQML Expression
----------------------
    Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)

Polars Implementation Notes
---------------------------
Pure TS chain.

Required panel columns: ``volume``, ``adv20``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha043 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #043 — Volume-surge rank × 7d-decline rank.

    WorldQuant Formula
    ------------------
        ts_rank(volume / adv20, 20) * ts_rank(-1 * delta(close, 7), 8)

    Legacy AQML Expression
    ----------------------
        Ts_Rank(volume / adv20, 20) * Ts_Rank(-1 * Delta(close, 7), 8)

    Polars Implementation Notes
    ---------------------------
    Pure TS chain.

    Required panel columns: ``volume``, ``adv20``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select((ts_rank(pl.col('volume') / pl.col('adv20'), 20) * ts_rank(-1.0 * delta(pl.col('close'), 7), 8)).alias('alpha043')).to_series()
