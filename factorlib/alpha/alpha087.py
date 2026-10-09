"""alpha087 — standalone alpha factor.

Alpha #087 — Negative max of close-vwap-blend delta-decay rank and abs-corr ts_rank.

WorldQuant Formula
------------------
    max(rank(decay_linear(delta(close * 0.369701 + vwap * (1 - 0.369701),
                                1.91233), 2.65461)),
        Ts_Rank(decay_linear(abs(correlation(
            IndNeutralize(adv81, IndClass.industry), close, 13.4132
        )), 4.89768), 14.4535)) * -1

Required panel columns: ``close``, ``vwap``, ``adv81``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha087 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

# --- private helper (from canonical source) ---
def _abs(col: pl.Expr) -> pl.Expr:
    return col.abs()

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #087 — Negative max of close-vwap-blend delta-decay rank and abs-corr ts_rank.

    WorldQuant Formula
    ------------------
        max(rank(decay_linear(delta(close * 0.369701 + vwap * (1 - 0.369701),
                                    1.91233), 2.65461)),
            Ts_Rank(decay_linear(abs(correlation(
                IndNeutralize(adv81, IndClass.industry), close, 13.4132
            )), 4.89768), 14.4535)) * -1

    Required panel columns: ``close``, ``vwap``, ``adv81``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend = pl.col('close') * 0.369701 + pl.col('vwap') * (1.0 - 0.369701)
    staged = panel.with_columns(ind_neutralize(pl.col('adv81'), 'industry').alias('__a087_ia'), delta(blend, 2).alias('__a087_db'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a087_ia'), pl.col('close'), 13).alias('__a087_corr'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a087_db'), 3).alias('__a087_d1'), ts_decay_linear(_abs(pl.col('__a087_corr')), 5).alias('__a087_d2'))
    staged4 = staged3.with_columns(cs_rank(pl.col('__a087_d1')).alias('__a087_p1'), ts_rank(pl.col('__a087_d2'), 14).alias('__a087_p2'))
    return staged4.select((pl.max_horizontal(pl.col('__a087_p1'), pl.col('__a087_p2')) * -1.0).alias('alpha087')).to_series()
