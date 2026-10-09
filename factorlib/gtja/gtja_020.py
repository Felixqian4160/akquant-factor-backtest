"""gtja_020 — standalone gtja factor.

GTJA Alpha #020 — 6-day % change times 100 (vwap-anchored).

Guotai Junan Formula
--------------------
    (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6) * 100

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_020 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #020 — 6-day % change times 100 (vwap-anchored).

    Guotai Junan Formula
    --------------------
        (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6) * 100

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    vwap = pl.col('vwap')
    vwap_lag = delay(vwap, 6)
    expr = (vwap - vwap_lag) / vwap_lag * 100.0
    return panel.select(expr.alias('gtja_020').cast(pl.Float64)).to_series()
