"""gtja_101 — standalone gtja factor.

GTJA #101 — VWAP-volume corr ranked < volume-mean corr ranked.

Guotai Junan Formula
--------------------
    ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME, 30), 37), 15)) <
      RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
                RANK(VOLUME), 11))) * -1)

Polars Implementation Notes
---------------------------
Two-stage ``with_columns`` to materialise the per-stock CORRs before
cross-section ranking, since polars cannot mix CS+TS partitions in a
single expression.

Direction: ``reverse`` (binary -1/0 — multiply by -1 in spec).
Category: ``correlation``.

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_101_120.py

Usage:
    from factorlib.gtja.gtja_101 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import TS_PART, corr, decay_linear, delay, delta, mean, rank, regbeta, safe_div, sma, std_, sum_, ts_min, ts_rank

# --- private helper (from canonical source) ---
def _bool_lt_signed(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    """``(a < b) * -1`` — Daic115 returns -1/0 with null where either is null."""
    return pl.when(a.is_null() | b.is_null()).then(None).otherwise((a < b).cast(pl.Float64) * -1.0).cast(pl.Float64)

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA #101 — VWAP-volume corr ranked < volume-mean corr ranked.

    Guotai Junan Formula
    --------------------
        ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME, 30), 37), 15)) <
          RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))),
                    RANK(VOLUME), 11))) * -1)

    Polars Implementation Notes
    ---------------------------
    Two-stage ``with_columns`` to materialise the per-stock CORRs before
    cross-section ranking, since polars cannot mix CS+TS partitions in a
    single expression.

    Direction: ``reverse`` (binary -1/0 — multiply by -1 in spec).
    Category: ``correlation``.
    """
    df = panel.with_columns([mean(pl.col('volume'), 30).alias('__mv30'), rank(pl.col('high') * 0.1 + pl.col('vwap') * 0.9).alias('__rp'), rank(pl.col('volume')).alias('__rv')])
    df = df.with_columns([sum_(pl.col('__mv30'), 37).alias('__smv30_37')])
    df = df.with_columns([corr(pl.col('close'), pl.col('__smv30_37'), 15).alias('__c1'), corr(pl.col('__rp'), pl.col('__rv'), 11).alias('__c2')])
    df = df.with_columns([rank(pl.col('__c1')).alias('__r1'), rank(pl.col('__c2')).alias('__r2')])
    return df.select(_bool_lt_signed(pl.col('__r1'), pl.col('__r2')).alias('gtja_101')).to_series()
