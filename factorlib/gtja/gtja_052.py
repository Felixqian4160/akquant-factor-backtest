"""gtja_052 — standalone gtja factor.

GTJA Alpha #052 — 26d upward / downward typical-price pressure × 100.

Guotai Junan Formula
--------------------
    SUM(MAX(0, H - DELAY((H+L+C)/3, 1)), 26) /
    SUM(MAX(0, DELAY((H+L+C)/3, 1) - L), 26) * 100

Required panel columns: ``high``, ``low``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_052 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #052 — 26d upward / downward typical-price pressure × 100.

    Guotai Junan Formula
    --------------------
        SUM(MAX(0, H - DELAY((H+L+C)/3, 1)), 26) /
        SUM(MAX(0, DELAY((H+L+C)/3, 1) - L), 26) * 100

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    typ = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    typ_lag = delay(typ, 1)
    up = pl.max_horizontal(pl.col('high') - typ_lag, pl.lit(0.0))
    dn = pl.max_horizontal(typ_lag - pl.col('low'), pl.lit(0.0))
    expr = sum_(up, 26) / sum_(dn, 26) * 100.0
    return panel.select(expr.alias('gtja_052').cast(pl.Float64)).to_series()
