"""alpha005 — standalone alpha factor.

Alpha #005 — Open vs 10d VWAP-mean rank, scaled by close-VWAP rank deviation.

WorldQuant Formula
------------------
    rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))

Legacy AQML Expression
----------------------
    Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))

Polars Implementation Notes
---------------------------
Both inner expressions can be evaluated as expressions, then ranked
cross-sectionally. We stage the TS-sum and the inner deviations to
keep the CS rank step pure-CS.

Required panel columns: ``open``, ``vwap``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha005 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #005 — Open vs 10d VWAP-mean rank, scaled by close-VWAP rank deviation.

    WorldQuant Formula
    ------------------
        rank(open - sum(vwap, 10) / 10) * (-1 * abs(rank(close - vwap)))

    Legacy AQML Expression
    ----------------------
        Rank(open - Ts_Sum(vwap, 10) / 10) * (-1 * Abs(Rank(close - vwap)))

    Polars Implementation Notes
    ---------------------------
    Both inner expressions can be evaluated as expressions, then ranked
    cross-sectionally. We stage the TS-sum and the inner deviations to
    keep the CS rank step pure-CS.

    Required panel columns: ``open``, ``vwap``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    inner1 = pl.col('open') - ts_sum(pl.col('vwap'), 10) / 10.0
    inner2 = pl.col('close') - pl.col('vwap')
    staged = panel.with_columns(inner1.alias('__a005_inner1'), inner2.alias('__a005_inner2'))
    return staged.select((cs_rank(pl.col('__a005_inner1')) * (-1.0 * cs_rank(pl.col('__a005_inner2')).abs())).alias('alpha005')).to_series()
