"""alpha041 — standalone alpha factor.

Alpha #041 — geometric mean of (high, low) minus vwap.

WorldQuant Formula
------------------
    ((high * low)^0.5) - vwap

Legacy AQML Expression
----------------------
    Power(high * low, 0.5) - vwap

Polars Implementation Notes
---------------------------
Pure scalar arithmetic — no rolling or cross-section ops. Negative values
indicate vwap above the geometric mid-price.

Required panel columns: ``high``, ``low``, ``vwap``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha041 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #041 — geometric mean of (high, low) minus vwap.

    WorldQuant Formula
    ------------------
        ((high * low)^0.5) - vwap

    Legacy AQML Expression
    ----------------------
        Power(high * low, 0.5) - vwap

    Polars Implementation Notes
    ---------------------------
    Pure scalar arithmetic — no rolling or cross-section ops. Negative values
    indicate vwap above the geometric mid-price.

    Required panel columns: ``high``, ``low``, ``vwap``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select(((pl.col('high') * pl.col('low')).sqrt() - pl.col('vwap')).alias('alpha041')).to_series()
