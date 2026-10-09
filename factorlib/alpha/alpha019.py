"""alpha019 — standalone alpha factor.

Alpha #019 — 7d-return sign times annual rank-mom multiplier.

WorldQuant Formula (Kakushadze 2015, eq. 19)
--------------------------------------------
    ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
     (1 + rank((1 + sum(returns, 250)))))

Legacy AQML Expression
----------------------
    (-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) *
    (1 + Rank(1 + Ts_Sum(returns, 250)))

Polars Implementation Notes
---------------------------
1. ``close - delay(close, 7)`` and ``delta(close, 7)`` are
   algebraically identical; the WorldQuant paper writes both for
   robustness. We follow the formula verbatim — adding the same
   quantity twice doubles the magnitude but keeps the sign.
2. The annual ``ts_sum(returns, 250)`` rank-multiplier requires a
   250-day window that is longer than the typical synthetic panel
   (60 days), so the output is null for the first ~250 rows of each
   stock. STHSF reference is computed on the same panel and shows
   the same null pattern.

Required panel columns: ``close``, ``returns``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha019 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #019 — 7d-return sign times annual rank-mom multiplier.

    WorldQuant Formula (Kakushadze 2015, eq. 19)
    --------------------------------------------
        ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) *
         (1 + rank((1 + sum(returns, 250)))))

    Legacy AQML Expression
    ----------------------
        (-1 * Sign((close - Delay(close, 7)) + Delta(close, 7))) *
        (1 + Rank(1 + Ts_Sum(returns, 250)))

    Polars Implementation Notes
    ---------------------------
    1. ``close - delay(close, 7)`` and ``delta(close, 7)`` are
       algebraically identical; the WorldQuant paper writes both for
       robustness. We follow the formula verbatim — adding the same
       quantity twice doubles the magnitude but keeps the sign.
    2. The annual ``ts_sum(returns, 250)`` rank-multiplier requires a
       250-day window that is longer than the typical synthetic panel
       (60 days), so the output is null for the first ~250 rows of each
       stock. STHSF reference is computed on the same panel and shows
       the same null pattern.

    Required panel columns: ``close``, ``returns``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    sign_part = -1.0 * sign_(pl.col('close') - delay(pl.col('close'), 7) + delta(pl.col('close'), 7))
    annual = ts_sum(pl.col('returns'), 250)
    staged = panel.with_columns(sign_part.alias('__a019_sign'), annual.alias('__a019_ann'))
    expr = pl.col('__a019_sign') * (1.0 + cs_rank(1.0 + pl.col('__a019_ann')))
    return staged.select(expr.alias('alpha019').cast(pl.Float64)).to_series()
