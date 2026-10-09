"""alpha015 — standalone alpha factor.

Alpha #015 — Negative 3d sum of ranked high-volume rank correlation.

WorldQuant Formula
------------------
    -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

Legacy AQML Expression
----------------------
    -1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)

Polars Implementation Notes
---------------------------
Two CS ranks are materialised before the 3-day TS correlation. STHSF
fills NaN/inf correlation rows with zero before the outer rank; we do
the same so the early-window rank/sum semantics match the reference.

Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha015 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #015 — Negative 3d sum of ranked high-volume rank correlation.

    WorldQuant Formula
    ------------------
        -1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Ts_Sum(Rank(Ts_Corr(Rank(high), Rank(volume), 3)), 3)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks are materialised before the 3-day TS correlation. STHSF
    fills NaN/inf correlation rows with zero before the outer rank; we do
    the same so the early-window rank/sum semantics match the reference.

    Required panel columns: ``high``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged1 = panel.with_columns(cs_rank(pl.col('high')).alias('__a015_rh'), cs_rank(pl.col('volume')).alias('__a015_rv'))
    staged2 = staged1.with_columns(ts_corr(pl.col('__a015_rh'), pl.col('__a015_rv'), 3).fill_nan(0.0).fill_null(0.0).alias('__a015_corr'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a015_corr')).alias('__a015_rcorr'))
    return staged3.select((-1.0 * ts_sum(pl.col('__a015_rcorr'), 3)).alias('alpha015').cast(pl.Float64)).to_series()
