"""gtja_033 — standalone gtja factor.

GTJA Alpha #033 — Long/medium return spread × low-min change × turnover rank.

Guotai Junan Formula
--------------------
    (((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) *
     RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * TSRANK(VOLUME, 5)

Daic115 references ``data["turn"]`` (turnover_rate) which we don't
have. Use ``amount / cap`` as proxy. Reference parquet does NOT
include gtja_033; reference test is skipped.

Required panel columns: ``low``, ``returns``, ``amount``, ``cap``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_021_040.py

Usage:
    from factorlib.gtja.gtja_033 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, ifelse, mean, rank, regbeta, sma, std_, sum_, ts_max, ts_min

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #033 — Long/medium return spread × low-min change × turnover rank.

    Guotai Junan Formula
    --------------------
        (((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) *
         RANK((SUM(RET, 240) - SUM(RET, 20)) / 220)) * TSRANK(VOLUME, 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have. Use ``amount / cap`` as proxy. Reference parquet does NOT
    include gtja_033; reference test is skipped.

    Required panel columns: ``low``, ``returns``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``volume_price``
    """
    low_min = ts_min(pl.col('low'), 5)
    ret = pl.col('returns')
    ret_sum = sum_(ret, 240) - sum_(ret, 20)
    turn_proxy = pl.col('amount') / pl.col('cap')
    staged = panel.with_columns(ret_sum.alias('__g033_rs'), turn_proxy.alias('__g033_tp'))
    staged = staged.with_columns(rank(pl.col('__g033_rs')).alias('__g033_rr'), rank(pl.col('__g033_tp')).alias('__g033_rt'))
    expr = (-1.0 * low_min + delay(low_min, 5)) * pl.col('__g033_rr') * delay(pl.col('__g033_rt'), 5)
    return staged.select(expr.alias('gtja_033').cast(pl.Float64)).to_series()
