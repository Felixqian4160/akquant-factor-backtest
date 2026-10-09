"""alpha057 — standalone alpha factor.

Alpha #057 — premium-to-vwap divided by decayed rank-of-30d-argmax.

WorldQuant Formula
------------------
    0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))

Legacy AQML Expression
----------------------
    -1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))

Polars Implementation Notes
---------------------------
1. ``ts_argmax(close, 30)``: position of 30-day max — encodes recency of
   the recent peak.
2. CS rank of that position, then 2-day decay-linear smoothing.
3. Divide ``close - vwap`` (intraday premium) by the smoothed rank.
4. Sign flipped so that "premium plus stale peak" is bearish.

NOTE: Not present in STHSF reference parquet — parity test skipped.

Required panel columns: ``close``, ``vwap``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha057 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #057 — premium-to-vwap divided by decayed rank-of-30d-argmax.

    WorldQuant Formula
    ------------------
        0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2)))

    Legacy AQML Expression
    ----------------------
        -1 * ((close - vwap) / Ts_DecayLinear(Rank(Ts_ArgMax(close, 30)), 2))

    Polars Implementation Notes
    ---------------------------
    1. ``ts_argmax(close, 30)``: position of 30-day max — encodes recency of
       the recent peak.
    2. CS rank of that position, then 2-day decay-linear smoothing.
    3. Divide ``close - vwap`` (intraday premium) by the smoothed rank.
    4. Sign flipped so that "premium plus stale peak" is bearish.

    NOTE: Not present in STHSF reference parquet — parity test skipped.

    Required panel columns: ``close``, ``vwap``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    close = pl.col('close')
    vwap = pl.col('vwap')
    arg = ts_argmax_last(close, 30)
    staged = panel.with_columns(arg.alias('__a057_arg'))
    staged = staged.with_columns(cs_rank(pl.col('__a057_arg')).alias('__a057_rk'))
    staged = staged.with_columns(ts_decay_linear(pl.col('__a057_rk'), 2).alias('__a057_dl'))
    return staged.select((-1.0 * (close - vwap) / pl.col('__a057_dl')).alias('alpha057')).to_series()
