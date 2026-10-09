"""alpha094 — standalone alpha factor.

Alpha #094 — Negative power of VWAP-trough rank with correlation exponent.

WorldQuant Formula
------------------
    -1 * rank(vwap - ts_min(vwap, 12))
          ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)

Legacy AQML Expression
----------------------
    -1 * Power(
      Rank(vwap - Ts_Min(vwap, 12)),
      Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)
    )

Polars Implementation Notes
---------------------------
Stage TS ranks before TS corr; CS rank base; TS rank exponent.

Required panel columns: ``vwap``, ``adv60``, ``stock_code``, ``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha094 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #094 — Negative power of VWAP-trough rank with correlation exponent.

    WorldQuant Formula
    ------------------
        -1 * rank(vwap - ts_min(vwap, 12))
              ^ ts_rank(correlation(ts_rank(vwap, 20), ts_rank(adv60, 4), 18), 3)

    Legacy AQML Expression
    ----------------------
        -1 * Power(
          Rank(vwap - Ts_Min(vwap, 12)),
          Ts_Rank(Ts_Corr(Ts_Rank(vwap, 20), Ts_Rank(adv60, 4), 18), 3)
        )

    Polars Implementation Notes
    ---------------------------
    Stage TS ranks before TS corr; CS rank base; TS rank exponent.

    Required panel columns: ``vwap``, ``adv60``, ``stock_code``, ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    base_inner = pl.col('vwap') - ts_min(pl.col('vwap'), 12)
    staged = panel.with_columns(ts_rank(pl.col('vwap'), 20).alias('__a094_trv'), ts_rank(pl.col('adv60'), 4).alias('__a094_tra'), base_inner.alias('__a094_base_inner'))
    staged2 = staged.with_columns(ts_corr(pl.col('__a094_trv'), pl.col('__a094_tra'), 18).alias('__a094_corr'))
    staged3 = staged2.with_columns(cs_rank(pl.col('__a094_base_inner')).alias('__a094_base'), ts_rank(pl.col('__a094_corr'), 3).alias('__a094_exp'))
    return staged3.select((-1.0 * power(pl.col('__a094_base'), pl.col('__a094_exp'))).alias('alpha094')).to_series()
