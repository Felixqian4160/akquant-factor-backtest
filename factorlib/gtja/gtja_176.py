"""gtja_176 — standalone gtja factor.

GTJA #176 — CORR(RANK(stoch_K), RANK(VOLUME), 6).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_176 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #176 — CORR(RANK(stoch_K), RANK(VOLUME), 6)."""
    stoch = (pl.col('close') - ts_min(pl.col('low'), 12)) / (ts_max(pl.col('high'), 12) - ts_min(pl.col('low'), 12))
    df = panel.with_columns(stoch.alias('__stoch'))
    df = df.with_columns([rank(pl.col('__stoch')).alias('__rs'), rank(pl.col('volume')).alias('__rv')])
    expr = corr(pl.col('__rs'), pl.col('__rv'), 6).alias('gtja_176')
    return df.select(expr).to_series()
