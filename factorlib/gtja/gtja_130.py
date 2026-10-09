"""gtja_130 — standalone gtja factor.

GTJA #130 — Decay-linear corr ratio: HL2~MA(V,40) over rank(VWAP)~rank(VOL).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_130 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #130 — Decay-linear corr ratio: HL2~MA(V,40) over rank(VWAP)~rank(VOL)."""
    df = panel.with_columns([corr((pl.col('high') + pl.col('low')) / 2.0, mean(pl.col('volume'), 40), 9).alias('__c1'), rank(pl.col('vwap')).alias('__rv'), rank(pl.col('volume')).alias('__rvo')])
    df = df.with_columns([corr(pl.col('__rv'), pl.col('__rvo'), 7).alias('__c2')])
    df = df.with_columns([decay_linear(pl.col('__c1'), 10).alias('__dl1'), decay_linear(pl.col('__c2'), 3).alias('__dl2')])
    df = df.with_columns([rank(pl.col('__dl1')).alias('__r1'), rank(pl.col('__dl2')).alias('__r2')])
    return df.select((pl.col('__r1') / pl.col('__r2')).alias('gtja_130')).to_series()
