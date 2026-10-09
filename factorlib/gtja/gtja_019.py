"""gtja_019 — standalone gtja factor.

GTJA Alpha #019 — Asymmetric 6-day price change ratio (vwap-anchored).

Guotai Junan Formula
--------------------
    if (VWAP < DELAY(VWAP, 6)) (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6)
    elif VWAP == DELAY(VWAP, 6) 0
    else (VWAP - DELAY(VWAP, 6)) / VWAP

Daic115 uses VWAP by default (use_vwap=True). We follow that.

Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_019 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #019 — Asymmetric 6-day price change ratio (vwap-anchored).

    Guotai Junan Formula
    --------------------
        if (VWAP < DELAY(VWAP, 6)) (VWAP - DELAY(VWAP, 6)) / DELAY(VWAP, 6)
        elif VWAP == DELAY(VWAP, 6) 0
        else (VWAP - DELAY(VWAP, 6)) / VWAP

    Daic115 uses VWAP by default (use_vwap=True). We follow that.

    Required panel columns: ``vwap``, ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``mean_reversion``
    """
    vwap = pl.col('vwap')
    vwap_lag = delay(vwap, 6)
    diff = vwap - vwap_lag
    branch_down = diff / vwap_lag
    branch_up = diff / vwap
    expr = pl.when(vwap < vwap_lag).then(branch_down).otherwise(branch_up)
    return panel.select(expr.alias('gtja_019').cast(pl.Float64)).to_series()
