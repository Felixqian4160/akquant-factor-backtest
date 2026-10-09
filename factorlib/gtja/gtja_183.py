"""gtja_183 — standalone gtja factor.

GTJA #183 — Errata (Daic115 SUMAC expansion).

Guotai Junan Formula
--------------------
    MAX(SUMAC(C-MEAN(C,24))) - MIN(SUMAC(C-MEAN(C,24))) / STD(C,24)

Daic115 expands SUMAC = SUM(diff, 24). Quality flag = 1.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_183 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #183 — Errata (Daic115 SUMAC expansion).

    Guotai Junan Formula
    --------------------
        MAX(SUMAC(C-MEAN(C,24))) - MIN(SUMAC(C-MEAN(C,24))) / STD(C,24)

    Daic115 expands SUMAC = SUM(diff, 24). Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    diff = pl.col('close') - mean(pl.col('close'), 24)
    s = sum_(diff, 24)
    expr = (ts_max(s, 24) - ts_min(s, 24) / std_(pl.col('close'), 24)).alias('gtja_183')
    return panel.select(expr).to_series()
