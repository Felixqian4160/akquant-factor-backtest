"""alpha060 — standalone alpha factor.

Alpha #060 — Williams %R volume rank minus argmax rank, scaled.

WorldQuant Formula
------------------
    -1 * (
        2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
        - scale(rank(ts_argmax(close, 10)))
    )

Legacy AQML Expression
----------------------
    -1 * (
        2 * Scale(Rank(((close - low) - (high - close)) / (high - low) * volume))
        - Scale(Rank(Ts_ArgMax(close, 10)))
    )

Polars Implementation Notes
---------------------------
Two CS-scaled rank chains, one based on the Williams-%R-style payload,
the other on TS argmax of close.

Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha060 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #060 — Williams %R volume rank minus argmax rank, scaled.

    WorldQuant Formula
    ------------------
        -1 * (
            2 * scale(rank(((close - low) - (high - close)) / (high - low) * volume))
            - scale(rank(ts_argmax(close, 10)))
        )

    Legacy AQML Expression
    ----------------------
        -1 * (
            2 * Scale(Rank(((close - low) - (high - close)) / (high - low) * volume))
            - Scale(Rank(Ts_ArgMax(close, 10)))
        )

    Polars Implementation Notes
    ---------------------------
    Two CS-scaled rank chains, one based on the Williams-%R-style payload,
    the other on TS argmax of close.

    Required panel columns: ``close``, ``low``, ``high``, ``volume``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    williams = (pl.col('close') - pl.col('low') - (pl.col('high') - pl.col('close'))) / (pl.col('high') - pl.col('low')) * pl.col('volume')
    staged = panel.with_columns(williams.alias('__a060_will'), ts_argmax(pl.col('close'), 10).alias('__a060_arg'))
    staged2 = staged.with_columns(cs_rank(pl.col('__a060_will')).alias('__a060_rwill'), cs_rank(pl.col('__a060_arg')).alias('__a060_rarg'))
    return staged2.select((-1.0 * (2.0 * cs_scale(pl.col('__a060_rwill')) - cs_scale(pl.col('__a060_rarg')))).alias('alpha060')).to_series()
