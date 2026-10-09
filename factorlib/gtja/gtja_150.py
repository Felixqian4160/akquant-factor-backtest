"""gtja_150 — standalone gtja factor.

GTJA #150 — (C+H+L)/3 * LOG(VOLUME). Daic115 uses log(volume).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_150 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #150 — (C+H+L)/3 * LOG(VOLUME). Daic115 uses log(volume)."""
    expr = ((pl.col('close') + pl.col('high') + pl.col('low')) / 3.0 * log_(pl.col('volume'))).alias('gtja_150')
    return panel.select(expr).to_series()
