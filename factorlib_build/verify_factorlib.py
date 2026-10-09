"""Verify every factorlib module end to end.

1. Coverage: every expected module exists, compiles, imports, exposes compute.
2. Runtime: EVERY module of all five groups actually executes on a synthetic
   OHLCV panel and returns an aligned polars Series. No factor is skipped.

Exit code 0 with "VERIFY_PASS" only when all checks pass.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl

LIB = Path(__file__).resolve().parent.parent / "factorlib"
sys.path.insert(0, str(LIB.parent))

GROUPS = {"alpha": 107, "gtja": 177, "talib": 77, "academic": 22, "github": 15}


def build_synthetic_panel() -> tuple[pl.DataFrame, int]:
    np.random.seed(7)
    dates = pl.date_range(pl.date(2018, 1, 1), pl.date(2021, 6, 30), eager=True).to_list()[:600]
    stocks = [f"{i:06d}.SZ" for i in range(1, 11)]
    keys = pl.DataFrame(
        list(product(stocks, dates)), schema=["stock_code", "trade_date"], orient="row"
    )
    n = keys.height
    close = np.random.uniform(10, 20, n)
    industries = ["bank", "tech", "energy", "health", "consumer"]
    panel = (
        keys.with_columns([
            pl.Series("open", close * (1 + np.random.normal(0, 0.01, n))),
            pl.Series("high", close * 1.02),
            pl.Series("low", close * 0.98),
            pl.Series("close", close),
            pl.Series("volume", np.random.uniform(1e5, 1e6, n)),
            pl.Series("amount", np.random.uniform(1e6, 1e8, n)),
            pl.Series("cap", np.random.uniform(1e9, 1e11, n)),
            pl.Series("circ_cap", np.random.uniform(1e9, 1e11, n)),
            pl.Series("bps", np.random.uniform(1, 20, n)),
            pl.Series("pe", np.random.uniform(5, 50, n)),
            pl.Series("grossprofit_margin", np.random.uniform(0.1, 0.6, n)),
            pl.Series("netprofit_yoy", np.random.normal(0, 0.2, n)),
            pl.Series("industry", np.random.choice(industries, n)),
            pl.Series("sub_industry", np.random.choice(industries, n)),
        ])
        .sort(["stock_code", "trade_date"])
        .with_columns([
            (pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1).fill_null(0).alias("returns"),
            (pl.col("amount") * 10 / pl.col("volume").clip(lower_bound=1)).alias("vwap"),
        ])
    )
    for window in (5, 10, 15, 20, 30, 40, 50, 60, 80, 81, 90, 100, 120, 150, 180):
        panel = panel.with_columns(
            pl.col("volume").rolling_mean(window).over("stock_code").alias(f"adv{window}")
        )
    return panel, n


def main() -> int:
    panel, n = build_synthetic_panel()
    print(f"panel: {panel.shape}")

    total = 0
    failures: list[tuple[str, str, str]] = []
    for group, expected in GROUPS.items():
        files = sorted(p for p in (LIB / group).glob("*.py") if p.name != "__init__.py")
        if len(files) != expected:
            failures.append((group, "count", f"{len(files)} != {expected}"))
            continue
        for path in files:
            module_name = f"factorlib.{group}.{path.stem}"
            try:
                module = importlib.import_module(module_name)
                fn = getattr(module, "compute", None)
                assert callable(fn), "compute not callable"
                values = fn(panel)
                assert isinstance(values, pl.Series), f"returned {type(values).__name__}"
                assert len(values) == n, f"len {len(values)} != {n}"
                total += 1
            except Exception:
                failures.append((module_name, "runtime", traceback.format_exc(limit=2).splitlines()[-1][:200]))
    print(f"executed: {total} / {sum(GROUPS.values())}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for name, kind, message in failures[:40]:
            print(f"  FAIL [{kind}] {name}: {message}")
        return 1
    print("VERIFY_PASS — all modules computed on real panel execution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
