"""gtja_061 — standalone gtja factor.

GTJA Alpha #061 — Max of rank(decay-VWAP-delta), rank(decay-rank-corr).

Guotai Junan Formula
--------------------
    MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
        RANK(DECAYLINEAR(RANK(CORR(LOW, MEAN(V, 80), 8)), 17))) * -1

Daic115 omits the `* -1`. We follow Daic115 (no negation).

Required panel columns: ``vwap``, ``low``, ``volume``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_061 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #061 — Max of rank(decay-VWAP-delta), rank(decay-rank-corr).

    Guotai Junan Formula
    --------------------
        MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)),
            RANK(DECAYLINEAR(RANK(CORR(LOW, MEAN(V, 80), 8)), 17))) * -1

    Daic115 omits the `* -1`. We follow Daic115 (no negation).

    Required panel columns: ``vwap``, ``low``, ``volume``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col('vwap'), 1), 12)
    cor = corr(pl.col('low'), mean(pl.col('volume'), 80), 8)
    staged = panel.with_columns(arm1_inner.alias('__g061_a1'), cor.alias('__g061_c'))
    staged = staged.with_columns(rank(pl.col('__g061_a1')).alias('__g061_r1'), rank(pl.col('__g061_c')).alias('__g061_rc'))
    staged = staged.with_columns(decay_linear(pl.col('__g061_rc'), 17).alias('__g061_a2_inner'))
    staged = staged.with_columns(rank(pl.col('__g061_a2_inner')).alias('__g061_r2'))
    return staged.select(pl.max_horizontal(pl.col('__g061_r1'), pl.col('__g061_r2')).alias('gtja_061').cast(pl.Float64)).to_series()
