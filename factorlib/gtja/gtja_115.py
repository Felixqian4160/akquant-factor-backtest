"""gtja_115 — standalone gtja factor.

GTJA #115 — Pow of two corrs: (HIGH*0.9+CLOSE*0.1)~MA(V,30) ^ HL2 mid-rank ~ vol-rank.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_115 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #115 — Pow of two corrs: (HIGH*0.9+CLOSE*0.1)~MA(V,30) ^ HL2 mid-rank ~ vol-rank."""
    df = panel.with_columns([corr(pl.col('high') * 0.9 + pl.col('close') * 0.1, mean(pl.col('volume'), 30), 10).alias('__c1')])
    df = df.with_columns([ts_rank((pl.col('high') + pl.col('low')) / 2.0, 4).alias('__t1'), ts_rank(pl.col('volume'), 10).alias('__t2')])
    df = df.with_columns([corr(pl.col('__t1'), pl.col('__t2'), 7).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    expr = pl.col('__r1').pow(pl.col('__r2')).alias('gtja_115')
    return df.select(expr).to_series()
