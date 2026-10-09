"""gtja_125 — standalone gtja factor.

GTJA #125 — Decay-linear ratios on corr(VWAP, MA(V,80)) and DELTA(0.5C+0.5VWAP).

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_125 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #125 — Decay-linear ratios on corr(VWAP, MA(V,80)) and DELTA(0.5C+0.5VWAP)."""
    df = panel.with_columns([corr(pl.col('vwap'), mean(pl.col('volume'), 80), 17).alias('__c'), delta((pl.col('close') + pl.col('vwap')) / 2.0, 3).alias('__d')])
    df = df.with_columns([decay_linear(pl.col('__c'), 20).alias('__dlc'), decay_linear(pl.col('__d'), 16).alias('__dld')])
    df = df.with_columns([rank(pl.col('__dlc')).alias('__r1'), rank(pl.col('__dld')).alias('__r2')])
    return df.select((pl.col('__r1') / pl.col('__r2')).alias('gtja_125')).to_series()
