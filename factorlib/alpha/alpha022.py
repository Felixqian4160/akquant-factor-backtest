"""alpha022 — standalone alpha factor.

Alpha #022 — Change in high-volume correlation scaled by 20d stdev rank.

WorldQuant Formula
------------------
    -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

Legacy AQML Expression
----------------------
    -1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))

Polars Implementation Notes
---------------------------
Inner TS corr → TS delta → multiplied by CS rank of TS std. Stage the
20-day std before the CS rank.

Required panel columns: ``high``, ``volume``, ``close``, ``stock_code``,
``trade_date``

Direction: ``reverse``
Category: ``volume_price``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/volume_price.py

Usage:
    from factorlib.alpha.alpha022 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import TS_PART, cs_rank, cs_scale, delay, delta, if_then_else, log_, pmax, pmin, power, safe_div, sign_, ts_argmax, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_min, ts_product, ts_rank, ts_std, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #022 — Change in high-volume correlation scaled by 20d stdev rank.

    WorldQuant Formula
    ------------------
        -1 * delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))

    Legacy AQML Expression
    ----------------------
        -1 * Delta(Ts_Corr(high, volume, 5), 5) * Rank(Ts_Std(close, 20))

    Polars Implementation Notes
    ---------------------------
    Inner TS corr → TS delta → multiplied by CS rank of TS std. Stage the
    20-day std before the CS rank.

    Required panel columns: ``high``, ``volume``, ``close``, ``stock_code``,
    ``trade_date``

    Direction: ``reverse``
    Category: ``volume_price``
    """
    corr5 = ts_corr(pl.col('high'), pl.col('volume'), 5)
    delta_corr = (corr5 - corr5.shift(5)).over(TS_PART)
    staged = panel.with_columns(delta_corr.alias('__a022_dcorr'), ts_std(pl.col('close'), 20).alias('__a022_std'))
    return staged.select((-1.0 * pl.col('__a022_dcorr') * cs_rank(pl.col('__a022_std'))).alias('alpha022')).to_series()
