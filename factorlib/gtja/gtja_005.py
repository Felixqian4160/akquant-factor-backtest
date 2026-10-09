"""gtja_005 — standalone gtja factor.

GTJA Alpha #005 — Negated 3d rolling-max of 5d ts-rank corr.

Guotai Junan Formula
--------------------
    (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))

Reference parquet does not include gtja_005 because Daic115 used
pandas ``rolling.rank`` whose semantics changed across versions.
Our implementation uses our own ``ts_rank`` (rank of last value in
the window, polars rolling_rank); reference test will be skipped.

Required panel columns: ``volume``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_001_020.py

Usage:
    from factorlib.gtja.gtja_005 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, delay, delta, ifelse, log_, mean, rank, safe_pow_clip, sign_, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #005 — Negated 3d rolling-max of 5d ts-rank corr.

    Guotai Junan Formula
    --------------------
        (-1 * TSMAX(CORR(TSRANK(VOLUME, 5), TSRANK(HIGH, 5), 5), 3))

    Reference parquet does not include gtja_005 because Daic115 used
    pandas ``rolling.rank`` whose semantics changed across versions.
    Our implementation uses our own ``ts_rank`` (rank of last value in
    the window, polars rolling_rank); reference test will be skipped.

    Required panel columns: ``volume``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged = panel.with_columns(ts_rank(pl.col('volume'), 5).alias('__g005_tv'), ts_rank(pl.col('high'), 5).alias('__g005_th'))
    staged = staged.with_columns(corr(pl.col('__g005_tv'), pl.col('__g005_th'), 5).alias('__g005_c'))
    return staged.select((-1.0 * ts_max(pl.col('__g005_c'), 3)).alias('gtja_005').cast(pl.Float64)).to_series()
