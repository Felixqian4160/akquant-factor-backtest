"""gtja_132 — standalone gtja factor.

GTJA #132 — MEAN(AMOUNT, 20). Average daily turnover.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_132 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #132 — MEAN(AMOUNT, 20). Average daily turnover."""
    expr = mean(pl.col('amount'), 20).alias('gtja_132')
    return panel.select(expr).to_series()
