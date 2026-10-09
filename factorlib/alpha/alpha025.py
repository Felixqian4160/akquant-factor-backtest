"""alpha025 — standalone alpha factor.

Alpha #025 — Rank of negative-returns × volume-weighted price-tail.

WorldQuant Formula
------------------
    rank(((-1 * returns) * adv20) * vwap * (high - close))

Legacy AQML Expression
----------------------
    Rank((-1 * returns) * adv20 * vwap * (high - close))

Polars Implementation Notes
---------------------------
Build the multiplicative payload first, then CS rank.

Required panel columns: ``returns``, ``adv20``, ``vwap``, ``high``, ``close``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha025 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #025 — Rank of negative-returns × volume-weighted price-tail.

    WorldQuant Formula
    ------------------
        rank(((-1 * returns) * adv20) * vwap * (high - close))

    Legacy AQML Expression
    ----------------------
        Rank((-1 * returns) * adv20 * vwap * (high - close))

    Polars Implementation Notes
    ---------------------------
    Build the multiplicative payload first, then CS rank.

    Required panel columns: ``returns``, ``adv20``, ``vwap``, ``high``, ``close``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner = -1.0 * pl.col('returns') * pl.col('adv20') * pl.col('vwap') * (pl.col('high') - pl.col('close'))
    staged = panel.with_columns(inner.alias('__a025_inner'))
    return staged.select(cs_rank(pl.col('__a025_inner')).alias('alpha025')).to_series()
