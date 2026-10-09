"""alpha089 — standalone alpha factor.

Alpha #089 — Diff of two ts_rank(decay) composites with industry-neutralised vwap.

WorldQuant Formula
------------------
    Ts_Rank(decay_linear(correlation(low * 0.967285 + low * (1 - 0.967285),
                                      adv10, 6.94279), 5.51607), 3.79744) -
    Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry),
                                3.48158), 10.1466), 15.3012)

Polars Implementation Notes
---------------------------
Synthetic panel uses ``adv15`` as substitute for ``adv10``.

Required panel columns: ``low``, ``vwap``, ``adv10``, ``stock_code``,
``trade_date``, ``industry``

Direction: ``reverse``
Category: ``industry_neutral``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/alpha101/industry_neutral.py

Usage:
    from factorlib.alpha.alpha089 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.alpha101_ops import CS_PART, TS_PART, cs_rank, cs_scale, delay, delta, ind_neutralize, log_, sign_, ts_argmax, ts_argmin, ts_corr, ts_cov, ts_decay_linear, ts_max, ts_mean, ts_min, ts_product, ts_rank, ts_sum

def compute(panel: pl.DataFrame) -> pl.Series:
    """Alpha #089 — Diff of two ts_rank(decay) composites with industry-neutralised vwap.

    WorldQuant Formula
    ------------------
        Ts_Rank(decay_linear(correlation(low * 0.967285 + low * (1 - 0.967285),
                                          adv10, 6.94279), 5.51607), 3.79744) -
        Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry),
                                    3.48158), 10.1466), 15.3012)

    Polars Implementation Notes
    ---------------------------
    Synthetic panel uses ``adv15`` as substitute for ``adv10``.

    Required panel columns: ``low``, ``vwap``, ``adv10``, ``stock_code``,
    ``trade_date``, ``industry``

    Direction: ``reverse``
    Category: ``industry_neutral``
    """
    blend_low = pl.col('low') * 0.967285 + pl.col('low') * (1.0 - 0.967285)
    staged = panel.with_columns(ind_neutralize(pl.col('vwap'), 'industry').alias('__a089_iv'), ts_corr(blend_low, pl.col('adv15'), 7).alias('__a089_corr'))
    staged2 = staged.with_columns(delta(pl.col('__a089_iv'), 3).alias('__a089_div'), ts_decay_linear(pl.col('__a089_corr'), 6).alias('__a089_dec1'))
    staged3 = staged2.with_columns(ts_decay_linear(pl.col('__a089_div'), 10).alias('__a089_dec2'))
    return staged3.select((ts_rank(pl.col('__a089_dec1'), 4) - ts_rank(pl.col('__a089_dec2'), 15)).alias('alpha089')).to_series()
