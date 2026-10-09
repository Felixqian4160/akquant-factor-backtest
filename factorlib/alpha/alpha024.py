"""alpha024 — standalone alpha factor.

Alpha #024 — Long-horizon mean-acceleration switch.

WorldQuant Formula
------------------
    ((delta(sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05)
    ? -1 * (close - ts_min(close, 100))
    : -1 * delta(close, 3)

Polars Implementation Notes
---------------------------
The 100-day windows make this a long-horizon factor; on the synthetic
60-day panel the output is mostly NaN until day ~100 (which never
arrives on synthetic). Steady-state test is consequently best-effort.

Required panel columns: ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``cap_weighted``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/cap_weighted.py

Usage:
    from factorlib.alpha.alpha024 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, delay, delta, ts_mean, ts_min, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #024 — Long-horizon mean-acceleration switch.

    WorldQuant Formula
    ------------------
        ((delta(sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05)
        ? -1 * (close - ts_min(close, 100))
        : -1 * delta(close, 3)

    Polars Implementation Notes
    ---------------------------
    The 100-day windows make this a long-horizon factor; on the synthetic
    60-day panel the output is mostly NaN until day ~100 (which never
    arrives on synthetic). Steady-state test is consequently best-effort.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``cap_weighted``
    """
    cond_num = delta(ts_mean(pl.col('close'), 100), 100)
    cond_den = delay(pl.col('close'), 100)
    cond = cond_num / cond_den <= 0.05
    branch_true = -1.0 * (pl.col('close') - ts_min(pl.col('close'), 100))
    branch_false = -1.0 * delta(pl.col('close'), 3)
    return panel.select(pl.when(cond).then(branch_true).otherwise(branch_false).alias('alpha024')).to_series()
