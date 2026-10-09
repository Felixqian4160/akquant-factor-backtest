"""alpha033 — standalone alpha factor.

Alpha #033 — cross-section rank of (-1 + open/close).

WorldQuant Formula
------------------
    rank((-1 * ((1 - (open / close))^1)))

Legacy AQML Expression
----------------------
    Rank(-1 * Power(1 - (open / close), 1))

Polars Implementation Notes
---------------------------
Algebraic simplification used by STHSF: ``-1 * (1 - open/close)``
collapses to ``open/close - 1``. Cross-section pct rank only.

Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha033 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #033 — cross-section rank of (-1 + open/close).

    WorldQuant Formula
    ------------------
        rank((-1 * ((1 - (open / close))^1)))

    Legacy AQML Expression
    ----------------------
        Rank(-1 * Power(1 - (open / close), 1))

    Polars Implementation Notes
    ---------------------------
    Algebraic simplification used by STHSF: ``-1 * (1 - open/close)``
    collapses to ``open/close - 1``. Cross-section pct rank only.

    Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    inner = pl.col('open') / pl.col('close') - 1.0
    return panel.select(cs_rank(inner).alias('alpha033')).to_series()
