"""Exact per-module input dependency scan for the fsdb recompute.

For each of the 414 factor modules, list:
- pl.col("X") constants used (excluding internal __ aliases)
- string constants passed to ind_neutralize (industry group names)
- the docstring "Required panel columns" line

Then compare against the fsdb v3 available set under mapping and report:
- factors needing bps / grossprofit_margin / netprofit_yoy / industry / sub_industry
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

LIB = Path("/media/felix/f/quant/akquant-factor-backtest/factorlib")

AVAILABLE = {
    # direct
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "volume", "amount", "turnover", "pct_chg", "amplitude", "vol_ratio",
    "total_share", "float_share", "total_mv", "float_mv", "pe_ttm", "pb",
    "is_st", "name", "cum_t", "cum_latest", "adj_factor_hfq", "qfq_factor",
    "open_hfq", "high_hfq", "low_hfq", "close_hfq",
    "open_qfq", "high_qfq", "low_qfq", "close_qfq",
    "sw_l1_code", "sw_l1_name", "sw_l2_code", "sw_l2_name", "sw_l3_code", "sw_l3_name",
    # derived in the input contract
    "stock_code", "returns", "vwap", "cap", "circ_cap", "pe", "vol",
    *(f"adv{w}" for w in (5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180)),
    "industry", "sub_industry",
}
MISSING_SOURCES = {"bps", "grossprofit_margin", "netprofit_yoy"}

need_missing = defaultdict(list)
need_industry = []
for group in ("alpha", "gtja", "talib", "academic", "github"):
    for path in sorted((LIB / group).glob("*.py")):
        if path.name == "__init__.py":
            continue
        text = path.read_text()
        tree = ast.parse(text)
        cols: set[str] = set()
        industry_args: set[str] = set()
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
                    cols.add(name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ind_neutralize"
            ):
                for arg in node.args[1:]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        industry_args.add(arg.value)
                    if (
                        isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Attribute)
                        and getattr(arg.func, "attr", None) == "col"
                        and arg.args
                        and isinstance(arg.args[0], ast.Constant)
                    ):
                        industry_args.add(arg.args[0].value)
        missing = sorted(cols & MISSING_SOURCES)
        if missing:
            need_missing[path.stem] = missing
        if industry_args:
            need_industry.append((path.stem, sorted(industry_args)))

print("=== modules needing missing fundamental columns ===")
for stem, cols in sorted(need_missing.items()):
    print(f"  {stem:<28} {cols}")

print()
print(f"=== modules calling ind_neutralize ({len(need_industry)}) ===")
for stem, args in need_industry[:40]:
    print(f"  {stem:<28} {args}")

# cross-check union of all external cols vs available
all_cols: set[str] = set()
for group in ("alpha", "gtja", "talib", "academic", "github"):
    for path in sorted((LIB / group).glob("*.py")):
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
                    all_cols.add(name)
# talib modules use compute_talib which needs ohlcv; already covered
unknown = sorted(c for c in all_cols if c not in AVAILABLE)
print()
print("=== all external cols NOT satisfiable by contract ===")
print(f"  {unknown}")
print()
print(f"total external cols scanned: {len(all_cols)}")
