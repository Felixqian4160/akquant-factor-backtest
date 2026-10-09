"""gtja_078 — standalone gtja factor.

GTJA Alpha #078 — CCI-style typical price oscillator.

Guotai Junan Formula
--------------------
    ((H+L+C)/3 - MA((H+L+C)/3, 12)) /
    (0.015 * MEAN(|C - MEAN((H+L+C)/3, 12)|, 12))

Required panel columns: ``high``, ``low``, ``close``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_078 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #078 — CCI-style typical price oscillator.

    Guotai Junan Formula
    --------------------
        ((H+L+C)/3 - MA((H+L+C)/3, 12)) /
        (0.015 * MEAN(|C - MEAN((H+L+C)/3, 12)|, 12))

    Required panel columns: ``high``, ``low``, ``close``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    typ = (pl.col('high') + pl.col('low') + pl.col('close')) / 3.0
    typ_ma = mean(typ, 12)
    expr = (typ - typ_ma) / (0.015 * mean(abs_(pl.col('close') - typ_ma), 12))
    return panel.select(expr.alias('gtja_078').cast(pl.Float64)).to_series()
