"""gtja_087 — standalone gtja factor.

GTJA Alpha #087 — Rank-decay(vwap delta) + TS-rank-decay(asymmetric spread), negated.

Guotai Junan Formula
--------------------
    (RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
     TSRANK(DECAYLINEAR(((L*0.9 + L*0.1) - VWAP) / (O - (H+L)/2), 11), 7)) * -1

Required panel columns: ``vwap``, ``low``, ``open``, ``high``,
``stock_code``, ``trade_date``.

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_081_100.py

Usage:
    from factorlib.gtja.gtja_087 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, covariance, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #087 — Rank-decay(vwap delta) + TS-rank-decay(asymmetric spread), negated.

    Guotai Junan Formula
    --------------------
        (RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) +
         TSRANK(DECAYLINEAR(((L*0.9 + L*0.1) - VWAP) / (O - (H+L)/2), 11), 7)) * -1

    Required panel columns: ``vwap``, ``low``, ``open``, ``high``,
    ``stock_code``, ``trade_date``.

    Direction: ``reverse``
    Category: ``volume_price``
    """
    arm1_inner = decay_linear(delta(pl.col('vwap'), 4), 7)
    spread_num = pl.col('low') * 0.9 + pl.col('low') * 0.1 - pl.col('vwap')
    spread_den = pl.col('open') - (pl.col('high') + pl.col('low')) / 2.0 + 1e-07
    arm2_inner = decay_linear(spread_num / spread_den, 11)
    arm2 = ts_rank(arm2_inner, 7)
    staged = panel.with_columns(arm1_inner.alias('__g087_a1_inner'))
    staged = staged.with_columns(rank(pl.col('__g087_a1_inner')).alias('__g087_r1'))
    return staged.select(((pl.col('__g087_r1') + arm2) * -1.0).alias('gtja_087').cast(pl.Float64)).to_series()
