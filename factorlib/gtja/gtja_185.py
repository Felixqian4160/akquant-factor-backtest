"""gtja_185 — standalone gtja factor.

GTJA #185 — RANK(-1 * (1 - O/C)^2). Squared open-close gap, ranked.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_185 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #185 — RANK(-1 * (1 - O/C)^2). Squared open-close gap, ranked."""
    inner = -1.0 * (1.0 - pl.col('open') / pl.col('close')).pow(2)
    df = panel.with_columns(inner.alias('__i'))
    return df.select(rank(pl.col('__i')).alias('gtja_185')).to_series()
