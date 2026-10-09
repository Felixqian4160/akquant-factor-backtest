"""alpha013 — standalone alpha factor.

Alpha #013 — Negative rank of close-rank vs volume-rank covariance.

WorldQuant Formula
------------------
    -1 * rank(covariance(rank(close), rank(volume), 5))

Legacy AQML Expression
----------------------
    -1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))

Polars Implementation Notes
---------------------------
Two CS ranks materialised, then TS covariance, then outer CS rank.
Two staging passes (CS → TS) followed by a final select.

Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha013 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #013 — Negative rank of close-rank vs volume-rank covariance.

    WorldQuant Formula
    ------------------
        -1 * rank(covariance(rank(close), rank(volume), 5))

    Legacy AQML Expression
    ----------------------
        -1 * Rank(Ts_Cov(Rank(close), Rank(volume), 5))

    Polars Implementation Notes
    ---------------------------
    Two CS ranks materialised, then TS covariance, then outer CS rank.
    Two staging passes (CS → TS) followed by a final select.

    Required panel columns: ``close``, ``volume``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    staged1 = panel.with_columns(cs_rank(pl.col('close')).alias('__a013_rc'), cs_rank(pl.col('volume')).alias('__a013_rv'))
    staged2 = staged1.with_columns(ts_cov(pl.col('__a013_rc'), pl.col('__a013_rv'), 5).alias('__a013_cov'))
    return staged2.select((-1.0 * cs_rank(pl.col('__a013_cov'))).alias('alpha013')).to_series()
