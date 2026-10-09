"""gtja_121 — standalone gtja factor.

GTJA #121 — Errata factor (best-effort implementation).

Guotai Junan Formula
--------------------
    (RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20),
        TSRANK(MEAN(VOLUME, 60), 2), 18), 3)) * -1

Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
Daic115 reference implements the paper formula literally. We follow
Daic115 here (best-effort) and tag with quality_flag=1 so downstream
code can mask it out if desired.

Direction: ``reverse``. Quality flag: ``1``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_121_140.py

Usage:
    from factorlib.gtja.gtja_121 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import corr, decay_linear, delay, delta, highday, log_, lowday, mean, rank, sma, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #121 — Errata factor (best-effort implementation).

    Guotai Junan Formula
    --------------------
        (RANK((VWAP - MIN(VWAP, 12))) ^ TSRANK(CORR(TSRANK(VWAP, 20),
            TSRANK(MEAN(VOLUME, 60), 2), 18), 3)) * -1

    Listed in ``wpwp/Alpha-101-GTJA-191`` errata as ``return 0``. The
    Daic115 reference implements the paper formula literally. We follow
    Daic115 here (best-effort) and tag with quality_flag=1 so downstream
    code can mask it out if desired.

    Direction: ``reverse``. Quality flag: ``1``.
    """
    df = panel.with_columns([(pl.col('vwap') - ts_min(pl.col('vwap'), 12)).alias('__d'), ts_rank(pl.col('vwap'), 20).alias('__t1'), ts_rank(mean(pl.col('volume'), 60), 2).alias('__t2')])
    df = df.with_columns([corr(pl.col('__t1'), pl.col('__t2'), 18).alias('__c')])
    df = df.with_columns([rank(pl.col('__d')).alias('__r'), ts_rank(pl.col('__c'), 3).alias('__tr')])
    expr = (pl.col('__r').pow(pl.col('__tr')) * -1.0).alias('gtja_121')
    return df.select(expr).to_series()
