"""alpha012 — standalone alpha factor.

Alpha #012 — Volume direction times negative price change.

WorldQuant Formula
------------------
    sign(delta(volume, 1)) * (-1 * delta(close, 1))

Legacy AQML Expression
----------------------
    Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))

Polars Implementation Notes
---------------------------
Pure TS — sign of one-day volume change times negative one-day
close change. Sign returns 0 on zero change.

Required panel columns: ``volume``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha012 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #012 — Volume direction times negative price change.

    WorldQuant Formula
    ------------------
        sign(delta(volume, 1)) * (-1 * delta(close, 1))

    Legacy AQML Expression
    ----------------------
        Sign(Delta(volume, 1)) * (-1 * Delta(close, 1))

    Polars Implementation Notes
    ---------------------------
    Pure TS — sign of one-day volume change times negative one-day
    close change. Sign returns 0 on zero change.

    Required panel columns: ``volume``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    return panel.select((sign_(delta(pl.col('volume'), 1)) * (-1.0 * delta(pl.col('close'), 1))).alias('alpha012')).to_series()
