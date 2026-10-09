"""alpha036 — standalone alpha factor.

Alpha #036 — Multi-component composite (5 weighted ranks).

WorldQuant Formula
------------------
    2.21 * rank(correlation((close - open), delay(volume, 1), 15)) +
    0.7  * rank(open - close) +
    0.73 * rank(Ts_Rank(delay(-1 * returns, 6), 5)) +
    rank(abs(correlation(vwap, adv20, 6))) +
    0.6  * rank((sum(close, 200) / 200 - open) * (close - open))

Required panel columns: ``close``, ``open``, ``volume``, ``returns``,
``vwap``, ``adv20``, ``stock_code``, ``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha036 import compute
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
    """Alpha #036 — Multi-component composite (5 weighted ranks).

    WorldQuant Formula
    ------------------
        2.21 * rank(correlation((close - open), delay(volume, 1), 15)) +
        0.7  * rank(open - close) +
        0.73 * rank(Ts_Rank(delay(-1 * returns, 6), 5)) +
        rank(abs(correlation(vwap, adv20, 6))) +
        0.6  * rank((sum(close, 200) / 200 - open) * (close - open))

    Required panel columns: ``close``, ``open``, ``volume``, ``returns``,
    ``vwap``, ``adv20``, ``stock_code``, ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    co = pl.col('close') - pl.col('open')
    staged = panel.with_columns(ts_corr(co, delay(pl.col('volume'), 1), 15).alias('__a036_c1'), (pl.col('open') - pl.col('close')).alias('__a036_oc'), ts_rank(delay(-1.0 * pl.col('returns'), 6), 5).alias('__a036_tr'), ts_corr(pl.col('vwap'), pl.col('adv20'), 6).alias('__a036_c2'), ((ts_mean(pl.col('close'), 200) - pl.col('open')) * co).alias('__a036_p5'))
    return staged.select((2.21 * cs_rank(pl.col('__a036_c1')) + 0.7 * cs_rank(pl.col('__a036_oc')) + 0.73 * cs_rank(pl.col('__a036_tr')) + cs_rank(_abs(pl.col('__a036_c2'))) + 0.6 * cs_rank(pl.col('__a036_p5'))).alias('alpha036')).to_series()
