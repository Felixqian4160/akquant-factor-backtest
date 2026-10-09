"""alpha028 — standalone alpha factor.

Alpha #028 — Standardised mid-price vs close gap with volume modifier.

WorldQuant Formula
------------------
    scale(correlation(adv20, low, 5) + (high + low) / 2 - close)

Legacy AQML Expression
----------------------
    Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)

Polars Implementation Notes
---------------------------
TS corr then arithmetic, then CS scale (rescale to ``sum(|x|) == 1`` per day).

Required panel columns: ``adv20``, ``low``, ``high``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha028 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #028 — Standardised mid-price vs close gap with volume modifier.

    WorldQuant Formula
    ------------------
        scale(correlation(adv20, low, 5) + (high + low) / 2 - close)

    Legacy AQML Expression
    ----------------------
        Scale((Ts_Corr(adv20, low, 5) + (high + low) / 2) - close)

    Polars Implementation Notes
    ---------------------------
    TS corr then arithmetic, then CS scale (rescale to ``sum(|x|) == 1`` per day).

    Required panel columns: ``adv20``, ``low``, ``high``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner = ts_corr(pl.col('adv20'), pl.col('low'), 5) + (pl.col('high') + pl.col('low')) / 2.0 - pl.col('close')
    staged = panel.with_columns(inner.alias('__a028_inner'))
    return staged.select(cs_scale(pl.col('__a028_inner')).alias('alpha028')).to_series()
