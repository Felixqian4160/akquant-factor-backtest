"""gtja_165 — standalone gtja factor.

GTJA #165 — Errata (Daic115 expands SUMAC).

Guotai Junan Formula (paper)
----------------------------
    MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)

Daic115 expands SUMAC as ``SUM(diff, 48)`` and computes:
    TS_MAX(SUM(diff,48),48) - TS_MIN(SUM(diff,48),48) / STD(CLOSE,48)

Listed in errata as ``return 0`` in some references. We follow
Daic115's expansion. Quality flag = 1.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_161_180.py

Usage:
    from factorlib.gtja.gtja_165 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, delay, delta, highday, log_, mean, rank, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #165 — Errata (Daic115 expands SUMAC).

    Guotai Junan Formula (paper)
    ----------------------------
        MAX(SUMAC(CLOSE-MEAN(CLOSE,48))) - MIN(SUMAC(CLOSE-MEAN(CLOSE,48))) / STD(CLOSE,48)

    Daic115 expands SUMAC as ``SUM(diff, 48)`` and computes:
        TS_MAX(SUM(diff,48),48) - TS_MIN(SUM(diff,48),48) / STD(CLOSE,48)

    Listed in errata as ``return 0`` in some references. We follow
    Daic115's expansion. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col('close') - mean(pl.col('close'), 48)
    s48 = sum_(diff, 48)
    expr = (ts_max(s48, 48) - ts_min(s48, 48) / std_(pl.col('close'), 48)).alias('gtja_165')
    return panel.select(expr).to_series()
