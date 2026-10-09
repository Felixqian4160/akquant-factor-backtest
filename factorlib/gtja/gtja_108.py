"""gtja_108 — standalone gtja factor.

GTJA #108 — RANK(HIGH - MIN(HIGH,2)) ^ RANK(CORR(VWAP, MA(V,120),6))) * -1.

The XOR-looking caret in the paper is exponentiation in Daic115's
pandas reference (``a ** b``); we follow that.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_108 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #108 — RANK(HIGH - MIN(HIGH,2)) ^ RANK(CORR(VWAP, MA(V,120),6))) * -1.

    The XOR-looking caret in the paper is exponentiation in Daic115's
    pandas reference (``a ** b``); we follow that.
    """
    df = panel.with_columns([(pl.col('high') - ts_min(pl.col('high'), 2)).alias('__d'), corr(pl.col('vwap'), mean(pl.col('volume'), 120), 6).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__rd'), rank(pl.col('__c')).alias('__rc')])
    expr = (pl.col('__rd').pow(pl.col('__rc')) * -1.0).alias('gtja_108')
    return df.select(expr).to_series()
