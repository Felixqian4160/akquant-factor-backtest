"""alpha083 — standalone alpha factor.

Alpha #083 — Range/MA delay rank times volume rank-squared, scaled.

WorldQuant Formula
------------------
    (rank(delay((high - low) / (sum(close, 5) / 5), 2)) * rank(rank(volume)))
    / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))

Legacy AQML Expression
----------------------
    (Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * Rank(Rank(volume)))
    / (((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))

Polars Implementation Notes
---------------------------
Stage the range-over-MA series, its delay, and the double-CS-rank of
volume; final division involves the present-day ratio and price gap.

Numerical safety
----------------
On A-share limit-up days (一字板) high == low == close == vwap, so:
  - ``high - low = 0`` → range_ma = 0 → outer division uses 0/0
  - ``vwap - close = 0`` → inner division (range_ma / (vwap-close)) is 0/0
Without protection this previously emitted ~3k inf cells/year.
We now use ``safe_div`` for both denominators (returns null on |den|<1e-9).
The legacy formula above is preserved for parity reference.

Required panel columns: ``high``, ``low``, ``close``, ``volume``, ``vwap``,
``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha083 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #083 — Range/MA delay rank times volume rank-squared, scaled.

    WorldQuant Formula
    ------------------
        (rank(delay((high - low) / (sum(close, 5) / 5), 2)) * rank(rank(volume)))
        / (((high - low) / (sum(close, 5) / 5)) / (vwap - close))

    Legacy AQML Expression
    ----------------------
        (Rank(Delay((high - low) / (Ts_Sum(close, 5) / 5), 2)) * Rank(Rank(volume)))
        / (((high - low) / (Ts_Sum(close, 5) / 5)) / (vwap - close))

    Polars Implementation Notes
    ---------------------------
    Stage the range-over-MA series, its delay, and the double-CS-rank of
    volume; final division involves the present-day ratio and price gap.

    Numerical safety
    ----------------
    On A-share limit-up days (一字板) high == low == close == vwap, so:
      - ``high - low = 0`` → range_ma = 0 → outer division uses 0/0
      - ``vwap - close = 0`` → inner division (range_ma / (vwap-close)) is 0/0
    Without protection this previously emitted ~3k inf cells/year.
    We now use ``safe_div`` for both denominators (returns null on |den|<1e-9).
    The legacy formula above is preserved for parity reference.

    Required panel columns: ``high``, ``low``, ``close``, ``volume``, ``vwap``,
    ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    range_ma = safe_div(pl.col('high') - pl.col('low'), ts_sum(pl.col('close'), 5) / 5.0)
    staged = panel.with_columns(range_ma.alias('__a083_rm'))
    staged2 = staged.with_columns(delay(pl.col('__a083_rm'), 2).alias('__a083_drm'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a083_drm')).alias('__a083_r1'), cs_rank(cs_rank(pl.col('volume'))).alias('__a083_r2'))
    inner = safe_div(pl.col('__a083_rm'), pl.col('vwap') - pl.col('close'))
    return staged3.with_columns(inner.alias('__a083_inner')).select(safe_div(pl.col('__a083_r1') * pl.col('__a083_r2'), pl.col('__a083_inner')).alias('alpha083').cast(pl.Float64)).to_series()
