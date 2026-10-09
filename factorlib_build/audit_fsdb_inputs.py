"""Audit: map factorlib input requirements onto the fsdb v3 panel.

Outputs:
1. fsdb v3 full column list + dtypes
2. External input requirements per factor group (AST scan of pl.col)
3. Availability diff under the proposed field mapping
4. Units check: fsdb vol/amount vs v34 vol/amount on a sample stock
"""
from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import polars as pl

BASE = Path("/media/felix/f/quant/akquant-factor-backtest")
LIB = BASE / "factorlib"
FSDB = BASE / "data" / "wavehunter_hs300_fsdb_v3_20261009_001500.parquet"
V34 = BASE / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"

print("=== 1. fsdb v3 schema ===")
schema = pl.scan_parquet(FSDB).collect_schema()
for name, dtype in schema.items():
    print(f"  {name:<28} {dtype}")

print()
print("=== 2. factorlib external input requirements ===")
req = defaultdict(set)
for group in ("alpha", "gtja", "talib", "academic", "github"):
    for path in (LIB / group).glob("*.py"):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pl"
                and node.func.attr == "col"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                name = node.args[0].value
                if not name.startswith("_"):
                    req[group].add(name)

all_req = set()
for group, names in req.items():
    all_req |= names
    print(f"  {group:<9} ({len(names):>2}): {sorted(names)}")

print()
print(f"  UNION ({len(all_req)}): {sorted(all_req)}")

print()
print("=== 3. availability under mapping ===")
MAPPING = {
    "stock_code": "ts_code",
    "open": "open_hfq",
    "high": "high_hfq",
    "low": "low_hfq",
    "close": "close_hfq",
    "volume": "volume",
    "amount": "amount",
    "cap": "total_mv",
    "circ_cap": "float_mv",
    "pe": "pe_ttm",
}
fsdb_cols = set(schema.names())
missing = []
for name in sorted(all_req):
    if name in MAPPING:
        src = MAPPING[name]
        ok = src in fsdb_cols
        print(f"  {name:<24} <- {src:<14} {'OK' if ok else 'MISSING'}")
        if not ok:
            missing.append((name, src))
    elif name in fsdb_cols:
        print(f"  {name:<24}    (direct)")
    else:
        print(f"  {name:<24}    ## NOT AVAILABLE ##")
        missing.append((name, None))
print(f"  missing: {missing}")

print()
print("=== 4. units check: fsdb vs v34 (vol, amount, close) ===")
sample = pl.read_parquet(
    V34, columns=["ts_code", "trade_date", "close", "vol", "amount"]
).filter(pl.col("ts_code") == "000001.SZ").head(5)
fs = pl.read_parquet(
    FSDB, columns=["ts_code", "trade_date", "close", "volume", "amount"]
).filter(pl.col("ts_code") == "000001.SZ").head(5)
print("v34 sample:")
print(sample)
print("fsdb sample:")
print(fs)
