"""alpha042 — standalone alpha factor.

Alpha #042 — relative rank of vwap-close versus vwap+close.

WorldQuant Formula
------------------
    rank((vwap - close)) / rank((vwap + close))

Legacy AQML Expression
----------------------
    Rank(vwap - close) / Rank(vwap + close)

Polars Implementation Notes
---------------------------
Two CS ranks; divide. ``rank(vwap + close)`` should never be zero on real
data but the synthetic panel may produce ties — Polars handles that with
average-rank semantics.

Required panel columns: ``vwap``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha042 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #042 — relative rank of vwap-close versus vwap+close.

    WorldQuant Formula
    ------------------
        rank((vwap - close)) / rank((vwap + close))

    Legacy AQML Expression
    ----------------------
        Rank(vwap - close) / Rank(vwap + close)

    Polars Implementation Notes
    ---------------------------
    Two CS ranks; divide. ``rank(vwap + close)`` should never be zero on real
    data but the synthetic panel may produce ties — Polars handles that with
    average-rank semantics.

    Required panel columns: ``vwap``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    vmc = pl.col('vwap') - pl.col('close')
    vpc = pl.col('vwap') + pl.col('close')
    return panel.select((cs_rank(vmc) / cs_rank(vpc)).alias('alpha042')).to_series()
