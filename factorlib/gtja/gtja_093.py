"""gtja_093 — standalone gtja factor.

GTJA Alpha #093 — 20d sum of conditional max(O-L, O-DELAY(O,1)) when O<DELAY(O,1).

Guotai Junan Formula
--------------------
    SUM((O >= DELAY(O, 1) ? 0 : MAX(O - L, O - DELAY(O, 1))), 20)

Required panel columns: ``open``, ``low``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volatility``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_093 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #093 — 20d sum of conditional max(O-L, O-DELAY(O,1)) when O<DELAY(O,1).

    Guotai Junan Formula
    --------------------
        SUM((O >= DELAY(O, 1) ? 0 : MAX(O - L, O - DELAY(O, 1))), 20)

    Required panel columns: ``open``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volatility``
    """
    o = pl.col('open')
    o_lag = delay(o, 1)
    inner = pl.max_horizontal(o - pl.col('low'), o - o_lag)
    expr = pl.when(o >= o_lag).then(0.0).otherwise(inner)
    return panel.select(sum_(expr, 20).alias('gtja_093').cast(pl.Float64)).to_series()
