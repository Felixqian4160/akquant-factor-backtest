"""gtja_134 — standalone gtja factor.

GTJA #134 — (C-prev_C12)/prev_C12 * V — vol-weighted 12d return.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_134 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #134 — (C-prev_C12)/prev_C12 * V — vol-weighted 12d return."""
    pc = delay(pl.col('close'), 12)
    expr = ((pl.col('close') - pc) / pc * pl.col('volume')).alias('gtja_134')
    return panel.select(expr).to_series()
