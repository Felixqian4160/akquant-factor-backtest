"""alpha037 — standalone alpha factor.

Alpha #037 — long-horizon (open-close) vs close correlation + open-close rank.

WorldQuant Formula
------------------
    rank(correlation(delay((open - close), 1), close, 200)) +
        rank((open - close))

Legacy AQML Expression
----------------------
    Rank(Ts_Corr(Delay(open - close, 1), close, 200))
        + Rank(open - close)

Polars Implementation Notes
---------------------------
The 200-day window will yield NaN on the synthetic panel (60 days). The
second term still produces values immediately. Both ranks are CS pct.

Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``mean_reversion``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/mean_reversion.py

Usage:
    from factorlib.alpha.alpha037 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, cs_scale, delay, delta, ts_argmax_last, ts_argmin_last, ts_corr_safe, ts_decay_linear, ts_rank_int, ts_sum, ts_zscore

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #037 — long-horizon (open-close) vs close correlation + open-close rank.

    WorldQuant Formula
    ------------------
        rank(correlation(delay((open - close), 1), close, 200)) +
            rank((open - close))

    Legacy AQML Expression
    ----------------------
        Rank(Ts_Corr(Delay(open - close, 1), close, 200))
            + Rank(open - close)

    Polars Implementation Notes
    ---------------------------
    The 200-day window will yield NaN on the synthetic panel (60 days). The
    second term still produces values immediately. Both ranks are CS pct.

    Required panel columns: ``open``, ``close``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``mean_reversion``
    """
    oc = pl.col('open') - pl.col('close')
    corr = ts_corr_safe(delay(oc, 1), pl.col('close'), 200)
    staged = panel.with_columns(corr.alias('__a037_corr'), oc.alias('__a037_oc'))
    return staged.select((cs_rank(pl.col('__a037_corr')) + cs_rank(pl.col('__a037_oc'))).alias('alpha037')).to_series()
