"""gtja_062 — standalone gtja factor.

GTJA Alpha #062 — -CORR(HIGH, RANK(TURN), 5) — turnover proxied by amount/cap.

Guotai Junan Formula
--------------------
    -CORR(HIGH, RANK(TURN), 5)

Daic115 references ``data["turn"]`` (turnover_rate) which we don't
have in the synthetic panel. Use ``amount / cap`` as turnover proxy.
Reference parquet does NOT include gtja_062; reference test is
skipped.

Required panel columns: ``high``, ``amount``, ``cap``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_062 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #062 — -CORR(HIGH, RANK(TURN), 5) — turnover proxied by amount/cap.

    Guotai Junan Formula
    --------------------
        -CORR(HIGH, RANK(TURN), 5)

    Daic115 references ``data["turn"]`` (turnover_rate) which we don't
    have in the synthetic panel. Use ``amount / cap`` as turnover proxy.
    Reference parquet does NOT include gtja_062; reference test is
    skipped.

    Required panel columns: ``high``, ``amount``, ``cap``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    turn_proxy = pl.col('amount') / pl.col('cap')
    staged = panel.with_columns(rank(turn_proxy).alias('__g062_rt'))
    return staged.select((-1.0 * corr(pl.col('high'), pl.col('__g062_rt'), 5)).alias('gtja_062').cast(pl.Float64)).to_series()
