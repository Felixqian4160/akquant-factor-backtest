"""alpha029 — standalone alpha factor.

Alpha #029 — Deeply nested rank-scale-log composite plus delayed return ts_rank.

WorldQuant Formula
------------------
    min(product(rank(rank(scale(log(sum(ts_min(
        rank(rank(-1 * rank(delta((close - 1), 5)))), 2), 1))))), 1), 5) +
    ts_rank(delay(-1 * returns, 6), 5)

Polars Implementation Notes
---------------------------
Outer ``product(... , 1)`` is identity (1-window product). Inner
cascade: delta → CS rank → CS rank → CS rank → TS min(2) → TS sum(1)
is identity → log → scale → CS rank → CS rank, then TS min(5).

Required panel columns: ``close``, ``returns``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha029 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #029 — Deeply nested rank-scale-log composite plus delayed return ts_rank.

    WorldQuant Formula
    ------------------
        min(product(rank(rank(scale(log(sum(ts_min(
            rank(rank(-1 * rank(delta((close - 1), 5)))), 2), 1))))), 1), 5) +
        ts_rank(delay(-1 * returns, 6), 5)

    Polars Implementation Notes
    ---------------------------
    Outer ``product(... , 1)`` is identity (1-window product). Inner
    cascade: delta → CS rank → CS rank → CS rank → TS min(2) → TS sum(1)
    is identity → log → scale → CS rank → CS rank, then TS min(5).

    Required panel columns: ``close``, ``returns``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    staged = panel.with_columns(delta(pl.col('close') - 1.0, 5).alias('__a029_d'))
    staged = staged.with_columns(cs_rank(pl.col('__a029_d')).alias('__a029_r1'))
    staged = staged.with_columns((-1.0 * pl.col('__a029_r1')).alias('__a029_inner'))
    staged = staged.with_columns(cs_rank(pl.col('__a029_inner')).alias('__a029_r2'))
    staged = staged.with_columns(cs_rank(pl.col('__a029_r2')).alias('__a029_rr'))
    staged = staged.with_columns(ts_min(pl.col('__a029_rr'), 2).alias('__a029_min2'))
    staged = staged.with_columns(ts_sum(pl.col('__a029_min2'), 1).alias('__a029_sum1'))
    eps = 1e-12
    staged = staged.with_columns(log_(pl.col('__a029_sum1').clip(lower_bound=eps)).alias('__a029_log'))
    staged = staged.with_columns(cs_scale(pl.col('__a029_log')).alias('__a029_scale'))
    staged = staged.with_columns(cs_rank(pl.col('__a029_scale')).alias('__a029_rs1'))
    staged = staged.with_columns(cs_rank(pl.col('__a029_rs1')).alias('__a029_rrs'))
    staged = staged.with_columns(ts_min(pl.col('__a029_rrs'), 5).alias('__a029_part1'), ts_rank(delay(-1.0 * pl.col('returns'), 6), 5).alias('__a029_part2'))
    return staged.select((pl.col('__a029_part1') + pl.col('__a029_part2')).alias('alpha029')).to_series()
