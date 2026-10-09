"""alpha053 — standalone alpha factor.

Alpha #053 — 9-day delta of normalised position-within-day.

WorldQuant Formula
------------------
    -1 * delta(((close - low) - (high - close)) / (close - low), 9)

Legacy AQML Expression
----------------------
    -1 * Delta(((close - low) - (high - close)) / (close - low), 9)

Polars Implementation Notes
---------------------------
1. STHSF guards against ``(close - low) == 0`` by replacing with ``1e-4``.
   We mirror that to avoid division-by-zero NaN.
2. The inner ratio sits in [-1, 1] (close at low → -1, close at high → 1).

Required panel columns: ``close``, ``low``, ``high``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha053 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #053 — 9-day delta of normalised position-within-day.

    WorldQuant Formula
    ------------------
        -1 * delta(((close - low) - (high - close)) / (close - low), 9)

    Legacy AQML Expression
    ----------------------
        -1 * Delta(((close - low) - (high - close)) / (close - low), 9)

    Polars Implementation Notes
    ---------------------------
    1. STHSF guards against ``(close - low) == 0`` by replacing with ``1e-4``.
       We mirror that to avoid division-by-zero NaN.
    2. The inner ratio sits in [-1, 1] (close at low → -1, close at high → 1).

    Required panel columns: ``close``, ``low``, ``high``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col('close')
    low = pl.col('low')
    high = pl.col('high')
    cl = close - low
    safe_cl = pl.when(cl == 0.0).then(0.0001).otherwise(cl)
    inner = (cl - (high - close)) / safe_cl
    return panel.select((-1.0 * delta(inner, 9)).alias('alpha053')).to_series()
