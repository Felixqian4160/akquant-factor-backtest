"""gtja_120 — standalone gtja factor.

GTJA #120 — RANK(VWAP-CLOSE) / RANK(VWAP+CLOSE).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_120 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #120 — RANK(VWAP-CLOSE) / RANK(VWAP+CLOSE)."""
    df = panel.with_columns([rank(pl.col('vwap') - pl.col('close')).alias('__a'), rank(pl.col('vwap') + pl.col('close')).alias('__b')])
    return df.select((pl.col('__a') / pl.col('__b')).alias('gtja_120')).to_series()
