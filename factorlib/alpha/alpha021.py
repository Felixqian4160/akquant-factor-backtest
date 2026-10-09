"""alpha021 — standalone alpha factor.

Alpha #021 — Volatility-vs-momentum regime switch with volume confirmation.

WorldQuant Formula
------------------
    ((sum(close, 8) / 8 + stddev(close, 8)) < (sum(close, 2) / 2))
    ? -1 :
    ((sum(close, 2) / 2) < (sum(close, 8) / 8 - stddev(close, 8)))
    ? 1 :
    ((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : -1

Polars Implementation Notes
---------------------------
The STHSF reference simplifies the cascade: cond1 OR cond2 ⇒ -1
(any volatility-narrowing), else +1. We implement the literal paper
cascade for fidelity.

Required panel columns: ``close``, ``volume``, ``adv20``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``adv_extended``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/adv_extended.py

Usage:
    from factorlib.alpha.alpha021 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, ts_corr, ts_mean, ts_min, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #021 — Volatility-vs-momentum regime switch with volume confirmation.

    WorldQuant Formula
    ------------------
        ((sum(close, 8) / 8 + stddev(close, 8)) < (sum(close, 2) / 2))
        ? -1 :
        ((sum(close, 2) / 2) < (sum(close, 8) / 8 - stddev(close, 8)))
        ? 1 :
        ((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : -1

    Polars Implementation Notes
    ---------------------------
    The STHSF reference simplifies the cascade: cond1 OR cond2 ⇒ -1
    (any volatility-narrowing), else +1. We implement the literal paper
    cascade for fidelity.

    Required panel columns: ``close``, ``volume``, ``adv20``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``adv_extended``
    """
    avg8 = ts_mean(pl.col('close'), 8)
    avg2 = ts_mean(pl.col('close'), 2)
    std8 = ts_std(pl.col('close'), 8)
    cond_a = avg8 + std8 < avg2
    cond_b = avg2 < avg8 - std8
    cond_c = pl.col('volume') / pl.col('adv20') >= 1.0
    return panel.select(pl.when(cond_a).then(-1.0).when(cond_b).then(1.0).when(cond_c).then(1.0).otherwise(-1.0).alias('alpha021')).to_series()
