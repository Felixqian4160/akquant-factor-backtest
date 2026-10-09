"""Final units check before building v4: cap/circ_cap scale + hfq convention."""
from __future__ import annotations

from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"

v34 = pl.read_parquet(
    V34, columns=["ts_code", "trade_date", "close", "cap", "circ_cap", "adj_factor"]
).filter(pl.col("ts_code") == "000001.SZ")
fsdb = pl.read_parquet(
    FSDB,
    columns=["ts_code", "trade_date", "close", "total_mv", "float_mv",
             "adj_factor_hfq", "close_hfq", "close_qfq"],
).filter(pl.col("ts_code") == "000001.SZ")

print("=== v34 (Tushare) 000001.SZ ===")
print(v34.head(3))
print("...")
print(v34.filter(pl.col("trade_date").cast(pl.Utf8).str.contains("2020-01-02")))

print()
print("=== fsdb (stockdb) 000001.SZ ===")
print(fsdb.head(3))
print("...")
print(fsdb.filter(pl.col("trade_date") == "20200102"))
print()
print("fsdb tail:")
print(fsdb.tail(2))

# ratio checks on a common date: 20200102
print()
print("=== scale factors on 20200102 ===")
v = v34.filter(pl.col("trade_date").cast(pl.Utf8).str.contains("2020-01-02"))
f = fsdb.filter(pl.col("trade_date") == "20200102")
if v.height and f.height:
    vrow, frow = v.row(0, named=True), f.row(0, named=True)
    print(f"close:   v34={vrow['close']} fsdb_raw={frow['close']} fsdb_hfq={frow['close_hfq']} fsdb_qfq={frow['close_qfq']}")
    print(f"cap:     v34={vrow['cap']} fsdb_total_mv={frow['total_mv']}")
    print(f"circ_cap:v34={vrow['circ_cap']} fsdb_float_mv={frow['float_mv']}")
    print(f"adj_f:   v34={vrow['adj_factor']} fsdb_hfq={frow['adj_factor_hfq']}")
    if vrow["cap"] and frow["total_mv"]:
        print(f"cap ratio v34/total_mv = {vrow['cap'] / frow['total_mv']:.6f}")
    if vrow["circ_cap"] and frow["float_mv"]:
        print(f"circ_cap ratio v34/float_mv = {vrow['circ_cap'] / frow['float_mv']:.6f}")
    if vrow["close"]:
        print(f"close_hfq/raw = {frow['close_hfq'] / frow['close']:.4f}  (adj_factor_hfq={frow['adj_factor_hfq']:.6f})")
        print(f"close_qfq/raw = {frow['close_qfq'] / frow['close']:.4f}")
