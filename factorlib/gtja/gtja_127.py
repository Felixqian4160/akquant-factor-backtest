"""gtja_127 — standalone gtja factor.

GTJA #127 — RMS of pct-distance from 12-day rolling max.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_127 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #127 — RMS of pct-distance from 12-day rolling max."""
    tmax = ts_max(pl.col('close'), 12)
    inner = (100.0 * (pl.col('close') - tmax) / tmax).pow(2)
    expr = mean(inner, 12).pow(0.5).alias('gtja_127')
    return panel.select(expr).to_series()
