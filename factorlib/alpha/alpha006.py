"""alpha006 — standalone alpha factor.

Alpha #006 — 10-day correlation between open and volume.

WorldQuant Formula
------------------
    -1 * correlation(open, volume, 10)

Legacy AQML Expression
----------------------
    -1 * Ts_Corr(open, volume, 10)

Polars Implementation Notes
---------------------------
Pure TS — no CS staging needed.

Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha006 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #006 — 10-day correlation between open and volume.

    WorldQuant Formula
    ------------------
        -1 * correlation(open, volume, 10)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Corr(open, volume, 10)

    Polars Implementation Notes
    ---------------------------
    Pure TS — no CS staging needed.

    Required panel columns: ``open``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select((-1.0 * ts_corr(pl.col('open'), pl.col('volume'), 10)).alias('alpha006')).to_series()
