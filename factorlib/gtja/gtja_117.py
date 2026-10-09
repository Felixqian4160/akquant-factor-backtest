"""gtja_117 — standalone gtja factor.

GTJA #117 — TSRANK(VOL,32) * (1-TSRANK(C+H-L,16)) * (1-TSRANK(RET,32)).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_117 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #117 — TSRANK(VOL,32) * (1-TSRANK(C+H-L,16)) * (1-TSRANK(RET,32))."""
    ret = pl.col('close') / delay(pl.col('close'), 1) - 1.0
    chl = pl.col('close') + pl.col('high') - pl.col('low')
    expr = (ts_rank(pl.col('volume'), 32) * (1.0 - ts_rank(chl, 16)) * (1.0 - ts_rank(ret, 32))).alias('gtja_117')
    return panel.select(expr).to_series()
