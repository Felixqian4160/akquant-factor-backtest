"""gtja_137 — standalone gtja factor.

GTJA #137 — Complex true-range-normalised price change.

Daic115 reference implementation followed verbatim (with conditional
decomposed using IFELSE chain).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_137 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #137 — Complex true-range-normalised price change.

    Daic115 reference implementation followed verbatim (with conditional
    decomposed using IFELSE chain).
    """
    pc = delay(pl.col('close'), 1)
    pl_ = delay(pl.col('low'), 1)
    po = delay(pl.col('open'), 1)
    abshc = (pl.col('high') - pc).abs()
    abslc = (pl.col('low') - pc).abs()
    absco = (pc - po).abs()
    abshl = (pl.col('high') - pl_).abs()
    num = 16.0 * (pl.col('close') - pc + (pl.col('close') - pl.col('open')) / 2.0 + pc - po)
    case1 = abshc + abslc / 2.0 + absco / 4.0
    case2 = abslc + abshc / 2.0 + absco / 4.0
    case3 = abshl + absco / 4.0
    cond1 = (abshc > abslc) & (abshc > abshl)
    cond2 = (abslc > abshl) & (abslc > abshc)
    den = pl.when(cond1).then(case1).when(cond2).then(case2).otherwise(case3) + 1e-07
    max_h = pl.when(abshc > abslc).then(abshc).otherwise(abslc)
    expr = (num / den * max_h).alias('gtja_137')
    return panel.select(expr).to_series()
