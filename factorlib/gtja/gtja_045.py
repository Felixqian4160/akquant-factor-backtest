"""gtja_045 — standalone gtja factor.

GTJA Alpha #045 — Rank(C0.6+O0.4 delta) × Rank(corr(VWAP, MEAN(V,150), 15)).

Guotai Junan Formula
--------------------
    RANK(DELTA(C*0.6 + O*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))

Daic115 multiplies by raw corr (not rank); we follow.

Required panel columns: ``close``, ``open``, ``vwap``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_041_060.py

Usage:
    from factorlib.gtja.gtja_045 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, ifelse, mean, rank, sign_, sma, std_, sum_, sumif, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #045 — Rank(C0.6+O0.4 delta) × Rank(corr(VWAP, MEAN(V,150), 15)).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(C*0.6 + O*0.4, 1)) * RANK(CORR(VWAP, MEAN(VOLUME, 150), 15))

    Daic115 multiplies by raw corr (not rank); we follow.

    Required panel columns: ``close``, ``open``, ``vwap``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    weighted = pl.col('close') * 0.6 + pl.col('open') * 0.4
    inner_d = delta(weighted, 1)
    corr_v = corr(pl.col('vwap'), mean(pl.col('volume'), 150), 15)
    staged = panel.with_columns(inner_d.alias('__g045_d'))
    staged = staged.with_columns(rank(pl.col('__g045_d')).alias('__g045_r'))
    return staged.select((pl.col('__g045_r') * corr_v).alias('gtja_045').cast(pl.Float64)).to_series()
