"""gtja_021 — standalone gtja factor.

GTJA Alpha #021 — Rolling 6d slope of MEAN(close, 6) vs sequence.

Guotai Junan Formula
--------------------
    REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))

Daic115 uses ``qlib.data.ops.rolling_slope`` (cython). We compute
the same number via ``regbeta(y, x, 6)`` where ``x`` is a per-stock
row counter — the rolling slope is invariant to additive shifts of
``x``. Reference parquet does NOT include gtja_021 because the
Daic115 builder skipped qlib-dependent alphas; reference test is
skipped.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_021 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

# --- private helper (from canonical source) ---
def _stock_row_index(panel: pl.DataFrame) -> pl.Expr:
    """Per-stock cumulative row counter (0-based), expressed as Float64.

    ``regbeta(y, _stock_row_index(...), n)`` returns the rolling slope of
    ``y`` against ``[k, k+1, …, k+n-1]``, which (by translation
    invariance of covariance) equals the slope against the natural
    SEQUENCE(1..n) used in the Guotai Junan paper.
    """
    return pl.int_range(pl.len(), dtype=pl.Int64).over('stock_code').cast(pl.Float64)

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #021 — Rolling 6d slope of MEAN(close, 6) vs sequence.

    Guotai Junan Formula
    --------------------
        REGBETA(MEAN(CLOSE, 6), SEQUENCE(6))

    Daic115 uses ``qlib.data.ops.rolling_slope`` (cython). We compute
    the same number via ``regbeta(y, x, 6)`` where ``x`` is a per-stock
    row counter — the rolling slope is invariant to additive shifts of
    ``x``. Reference parquet does NOT include gtja_021 because the
    Daic115 builder skipped qlib-dependent alphas; reference test is
    skipped.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    val_mean = mean(pl.col('vwap'), 6)
    staged = panel.with_columns(val_mean.alias('__g021_y'), _stock_row_index(panel).alias('__g021_x'))
    return staged.select(regbeta(pl.col('__g021_y'), pl.col('__g021_x'), 6).alias('gtja_021').cast(pl.Float64)).to_series()
