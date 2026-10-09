"""alpha010 — standalone alpha factor.

Alpha #010 — Cross-sectional rank of trend-confirmed price change.

WorldQuant Formula (Kakushadze 2015, eq. 10)
--------------------------------------------
    rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
          ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))

Legacy AQML Expression
----------------------
    Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1),
           If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1),
              -1 * Delta(close, 1))))

Polars Implementation Notes
---------------------------
Same logic as :func:`alpha009` but with a 4-day lookback for the
trend confirmation, then cross-section ranked. Materialise before
ranking so that the TS lookback finishes before the CS partition.

Required panel columns: ``close``, ``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/momentum.py

Usage:
    from factorlib.alpha.alpha010 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import cs_rank, delay, delta, if_then_else, sign_, signed_power, ts_argmax, ts_corr, ts_decay_linear, ts_max, ts_min, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #010 — Cross-sectional rank of trend-confirmed price change.

    WorldQuant Formula (Kakushadze 2015, eq. 10)
    --------------------------------------------
        rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) :
              ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))

    Legacy AQML Expression
    ----------------------
        Rank(If(Ts_Min(Delta(close, 1), 4) > 0, Delta(close, 1),
               If(Ts_Max(Delta(close, 1), 4) < 0, Delta(close, 1),
                  -1 * Delta(close, 1))))

    Polars Implementation Notes
    ---------------------------
    Same logic as :func:`alpha009` but with a 4-day lookback for the
    trend confirmation, then cross-section ranked. Materialise before
    ranking so that the TS lookback finishes before the CS partition.

    Required panel columns: ``close``, ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``momentum``
    """
    d1 = delta(pl.col('close'), 1)
    inner = if_then_else(ts_max(d1, 4) < 0.0, d1, -1.0 * d1)
    inner_branch = if_then_else(ts_min(d1, 4) > 0.0, d1, inner)
    staged = panel.with_columns(inner_branch.alias('__a010_branch'))
    return staged.select(cs_rank(pl.col('__a010_branch')).alias('alpha010').cast(pl.Float64)).to_series()
