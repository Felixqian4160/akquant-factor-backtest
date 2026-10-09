"""gtja_151 — standalone gtja factor.

GTJA #151 — Errata factor (best-effort).

Guotai Junan Formula
--------------------
    SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)

Listed in ``wpwp/Alpha-101-GTJA-191`` errata. The formula itself is
well-defined; we implement it as Daic115 does. Quality flag = 1
pending downstream verification.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_141_160.py

Usage:
    from factorlib.gtja.gtja_151 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, count_, decay_linear, delay, delta, log_, mean, rank, regbeta, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #151 — Errata factor (best-effort).

    Guotai Junan Formula
    --------------------
        SMA(CLOSE - DELAY(CLOSE, 20), 20, 1)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata. The formula itself is
    well-defined; we implement it as Daic115 does. Quality flag = 1
    pending downstream verification.

    Direction: ``normal``. Quality flag: ``1``.
    """
    expr = sma(pl.col('close') - delay(pl.col('close'), 20), 20, 1).alias('gtja_151')
    return panel.select(expr).to_series()
