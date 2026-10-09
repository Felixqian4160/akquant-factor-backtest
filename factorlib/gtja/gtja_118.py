"""gtja_118 — standalone gtja factor.

GTJA #118 — SUM(H-O,20) / SUM(O-L,20) * 100.

Open-relative range-skew over 20 days.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_118 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #118 — SUM(H-O,20) / SUM(O-L,20) * 100.

    Open-relative range-skew over 20 days.
    """
    expr = (sum_(pl.col('high') - pl.col('open'), 20) / sum_(pl.col('open') - pl.col('low'), 20) * 100.0).alias('gtja_118')
    return panel.select(expr).to_series()
