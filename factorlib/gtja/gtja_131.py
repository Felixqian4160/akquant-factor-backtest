"""gtja_131 — standalone gtja factor.

GTJA #131 — Errata factor (best-effort implementation).

Guotai Junan Formula
--------------------
    RANK(DELTA(VWAP, 1)) ^ TSRANK(CORR(CLOSE, MEAN(VOLUME, 50), 18), 18)

Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
Daic115 reference implements the paper literally. We follow Daic115
(best-effort) and tag with quality_flag=1.

Direction: ``normal``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_131 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #131 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        RANK(DELTA(VWAP, 1)) ^ TSRANK(CORR(CLOSE, MEAN(VOLUME, 50), 18), 18)

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper literally. We follow Daic115
    (best-effort) and tag with quality_flag=1.

    Direction: ``normal``. Quality flag: ``1``.
    """
    df = panel.with_columns([delta(pl.col('vwap'), 1).alias('__d'), corr(pl.col('close'), mean(pl.col('volume'), 50), 18).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__r'), ts_rank(pl.col('__c'), 18).alias('__tr')])
    expr = pl.col('__r').pow(pl.col('__tr')).alias('gtja_131')
    return df.select(expr).to_series()
