"""gtja_069 — standalone gtja factor.

GTJA Alpha #069 — DTM/DBM 20-day asymmetric momentum ratio.

Guotai Junan Formula
--------------------
    DTM = (O <= DELAY(O,1)) ? 0 : MAX((H-O), (O-DELAY(O,1)))
    DBM = (O >= DELAY(O,1)) ? 0 : MAX((O-L), (O-DELAY(O,1)))
    S_DTM = SUM(DTM, 20); S_DBM = SUM(DBM, 20)
    S_DTM > S_DBM ? (S_DTM - S_DBM)/S_DTM
                   : S_DTM == S_DBM ? 0 : (S_DTM - S_DBM)/S_DBM

Required panel columns: ``open``, ``high``, ``low``,
``stock_code``, ``trade_date``.

Direction: ``normal``
Category: ``momentum``

Canonical source: /media/felix/f/quant/aurumq-rl/quant_workflow/src/aurumq_rl/factors/gtja191/batch_061_080.py

Usage:
    from factorlib.gtja.gtja_069 import compute
    values = compute(panel)

The panel must be sorted by (stock_code, trade_date). The returned
polars Series is aligned to the input rows.
"""
from __future__ import annotations

import polars as pl
from factorlib._ops.gtja191_ops import abs_, corr, decay_linear, delay, delta, mean, rank, sma, std_, sum_, ts_max, ts_min, ts_rank

def compute(panel: pl.DataFrame) -> pl.Series:
    """GTJA Alpha #069 — DTM/DBM 20-day asymmetric momentum ratio.

    Guotai Junan Formula
    --------------------
        DTM = (O <= DELAY(O,1)) ? 0 : MAX((H-O), (O-DELAY(O,1)))
        DBM = (O >= DELAY(O,1)) ? 0 : MAX((O-L), (O-DELAY(O,1)))
        S_DTM = SUM(DTM, 20); S_DBM = SUM(DBM, 20)
        S_DTM > S_DBM ? (S_DTM - S_DBM)/S_DTM
                       : S_DTM == S_DBM ? 0 : (S_DTM - S_DBM)/S_DBM

    Required panel columns: ``open``, ``high``, ``low``,
    ``stock_code``, ``trade_date``.

    Direction: ``normal``
    Category: ``momentum``
    """
    o = pl.col('open')
    o_lag = delay(o, 1)
    dtm_inner = pl.max_horizontal(pl.col('high') - o, o - o_lag)
    dbm_inner = pl.max_horizontal(o - pl.col('low'), o - o_lag)
    dtm = pl.when(o <= o_lag).then(0.0).otherwise(dtm_inner)
    dbm = pl.when(o >= o_lag).then(0.0).otherwise(dbm_inner)
    s_dtm = sum_(dtm, 20)
    s_dbm = sum_(dbm, 20)
    expr = pl.when(s_dtm > s_dbm).then((s_dtm - s_dbm) / s_dtm).otherwise(pl.when(s_dtm == s_dbm).then(0.0).otherwise((s_dtm - s_dbm) / s_dbm))
    return panel.select(expr.alias('gtja_069').cast(pl.Float64)).to_series()
