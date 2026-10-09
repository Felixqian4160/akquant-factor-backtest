"""gtja_168 — standalone gtja factor.

GTJA #168 — -V / MEAN(V,20). Inverse-volume-relative.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_168 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #168 — -V / MEAN(V,20). Inverse-volume-relative."""
    expr = (-1.0 * pl.col('volume') / mean(pl.col('volume'), 20)).alias('gtja_168')
    return panel.select(expr).to_series()
