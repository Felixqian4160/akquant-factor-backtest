"""alpha101 — standalone alpha factor.

Alpha #101 — intraday body over range.

WorldQuant Formula
------------------
    ((close - open) / ((high - low) + 0.001))

Legacy AQML Expression
----------------------
    (close - open) / ((high - low) + 0.001)

Polars Implementation Notes
---------------------------
The simplest WorldQuant alpha. Despite its name (#101) it is purely
intraday and ships in STHSF's reference. Acts as a candle-strength
contrarian — large positive bodies relative to the day's range are
expected to fade.

Required panel columns: ``close``, ``open``, ``high``, ``low``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha101 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #101 — intraday body over range.

    WorldQuant Formula
    ------------------
        ((close - open) / ((high - low) + 0.001))

    Legacy AQML Expression
    ----------------------
        (close - open) / ((high - low) + 0.001)

    Polars Implementation Notes
    ---------------------------
    The simplest WorldQuant alpha. Despite its name (#101) it is purely
    intraday and ships in STHSF's reference. Acts as a candle-strength
    contrarian — large positive bodies relative to the day's range are
    expected to fade.

    Required panel columns: ``close``, ``open``, ``high``, ``low``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    return panel.select(((pl.col('close') - pl.col('open')) / (pl.col('high') - pl.col('low') + 0.001)).alias('alpha101')).to_series()
