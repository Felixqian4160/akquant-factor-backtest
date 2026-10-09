"""gtja_191 — standalone gtja factor.

GTJA #191 — Errata (best-effort).

Guotai Junan Formula
--------------------
    CORR(MEAN(VOLUME,20), LOW, 5) + (HIGH+LOW)/2 - CLOSE

Listed in errata. The formula is well-defined; we implement it as
Daic115 does. Quality flag = 1.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_181_191.py

Usage:
    from factorlib.gtja.gtja_191 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, count_, delay, log_, mean, rank, sma, std_, sum_, sumif, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #191 — Errata (best-effort).

    Guotai Junan Formula
    --------------------
        CORR(MEAN(VOLUME,20), LOW, 5) + (HIGH+LOW)/2 - CLOSE

    Listed in errata. The formula is well-defined; we implement it as
    Daic115 does. Quality flag = 1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = (corr(mean(pl.col('volume'), 20), pl.col('low'), 5) + (pl.col('high') + pl.col('low')) / 2.0 - pl.col('close')).alias('gtja_191')
    return panel.select(expr).to_series()
