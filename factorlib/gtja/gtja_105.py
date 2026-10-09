"""gtja_105 — standalone gtja factor.

GTJA #105 — -1 * CORR(RANK(OPEN), RANK(VOLUME), 10).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_105 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #105 — -1 * CORR(RANK(OPEN), RANK(VOLUME), 10)."""
    df = panel.with_columns([rank(pl.col('open')).alias('__ro'), rank(pl.col('volume')).alias('__rv')])
    return df.select((-1.0 * corr(pl.col('__ro'), pl.col('__rv'), 10)).alias('gtja_105')).to_series()
