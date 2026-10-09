"""alpha056 — standalone alpha factor.

Alpha #056 — Inverse rank of returns ratio scaled by cap-weighted return rank.

WorldQuant Formula
------------------
    0 - 1 * (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) *
             rank(returns * cap))

Polars Implementation Notes
---------------------------
The denominator ``sum(sum(returns,2),3)`` is the 3-day rolling sum of
the 2-day rolling sum, i.e. cumulative returns over an effective
4-day window. ``returns * cap`` is the dollar return; CS rank
captures cross-section preference for high-dollar-return stocks.

Stocks with ``cap IS NULL`` produce a NaN factor row (because
``returns * NULL == NULL``, which propagates through CS rank).

Required panel columns: ``returns``, ``cap``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``cap_weighted``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/cap_weighted.py

Usage:
    from factorlib.alpha.alpha056 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, delay, delta, ts_mean, ts_min, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #056 — Inverse rank of returns ratio scaled by cap-weighted return rank.

    WorldQuant Formula
    ------------------
        0 - 1 * (rank(sum(returns, 10) / sum(sum(returns, 2), 3)) *
                 rank(returns * cap))

    Polars Implementation Notes
    ---------------------------
    The denominator ``sum(sum(returns,2),3)`` is the 3-day rolling sum of
    the 2-day rolling sum, i.e. cumulative returns over an effective
    4-day window. ``returns * cap`` is the dollar return; CS rank
    captures cross-section preference for high-dollar-return stocks.

    Stocks with ``cap IS NULL`` produce a NaN factor row (because
    ``returns * NULL == NULL``, which propagates through CS rank).

    Required panel columns: ``returns``, ``cap``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``cap_weighted``
    """
    inner_ratio = ts_sum(pl.col('returns'), 10) / ts_sum(ts_sum(pl.col('returns'), 2), 3)
    staged = panel.with_columns(inner_ratio.alias('__a056_ratio'), (pl.col('returns') * pl.col('cap')).alias('__a056_dollar'))
    return staged.select((-1.0 * cs_rank(pl.col('__a056_ratio')) * cs_rank(pl.col('__a056_dollar'))).alias('alpha056')).to_series()
