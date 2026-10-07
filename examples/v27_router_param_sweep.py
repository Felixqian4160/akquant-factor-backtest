"""V27 Router 三参数 sweep — 用 ZigZag ground truth 验证每个组合.

参数 sweep:
  BEAR_VOTE_THRESHOLD ∈ {2, 3, 4}
  BEAR_LOOKBACK ∈ {5, 10, 15, 20}
  BEAR_CUMULATIVE_THRESHOLD ∈ {3, 4, 5, 6}

每个组合重算 router 在 daily idx_close 上的 bear/bull_neutral,
然后 vs ZigZag ground truth (hs300_index_pivots.json, legs ≥100d)
算 Precision/Recall/Specificity/F1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

PIVOTS_PATH = "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/hs300_index_pivots_clean_20260919/hs300_index_pivots.json"
OUT = Path("evidence/v27_router_param_sweep_20261006")
OUT.mkdir(parents=True, exist_ok=True)

BEAR_SIGNAL_SPECS = [
    ("close_lt_MA5", lambda df: df["idx_close"] < df["ma5"]),
    ("close_lt_MA10", lambda df: df["idx_close"] < df["ma10"]),
    ("close_lt_MA20", lambda df: df["idx_close"] < df["ma20"]),
    ("close_lt_MA60", lambda df: df["idx_close"] < df["ma60"]),
    ("close_lt_MA120", lambda df: df["idx_close"] < df["ma120"]),
    ("MA20_lt_MA60", lambda df: df["ma20"] < df["ma60"]),
    ("MA60_lt_MA120", lambda df: df["ma60"] < df["ma120"]),
    ("ret10_lt_0", lambda df: df["ret10"] < 0),
    ("ret20_lt_0", lambda df: df["ret20"] < 0),
    ("ret40_lt_0", lambda df: df["ret40"] < 0),
    ("ret60_lt_0", lambda df: df["ret60"] < 0),
    ("dist_ma20_lt_neg2", lambda df: df["dist_ma20"] < -0.02),
    ("dist_ma60_lt_neg5", lambda df: df["dist_ma60"] < -0.05),
    ("slope_ma60_lt_0", lambda df: df["slope_ma60"] < 0),
    ("dd_60d_lt_neg5", lambda df: df["dd_60d"] < -0.05),
    ("dd_120d_lt_neg10", lambda df: df["dd_120d"] < -0.10),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def build_daily_idx() -> pd.DataFrame:
    """Build daily idx with 16 V27 signals."""
    df = pl.read_parquet("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet",
                        columns=["trade_date", "idx_close"]).filter(
        pl.col("idx_close").is_not_null()
    ).unique("trade_date").sort("trade_date")

    pdf = df.with_columns([
        pl.col("idx_close").rolling_mean(5).alias("ma5"),
        pl.col("idx_close").rolling_mean(10).alias("ma10"),
        pl.col("idx_close").rolling_mean(20).alias("ma20"),
        pl.col("idx_close").rolling_mean(60).alias("ma60"),
        pl.col("idx_close").rolling_mean(120).alias("ma120"),
        pl.col("idx_close").pct_change(10).alias("ret10"),
        pl.col("idx_close").pct_change(20).alias("ret20"),
        pl.col("idx_close").pct_change(40).alias("ret40"),
        pl.col("idx_close").pct_change(60).alias("ret60"),
    ]).with_columns([
        ((pl.col("idx_close") - pl.col("ma20")) / pl.col("ma20")).alias("dist_ma20"),
        ((pl.col("idx_close") - pl.col("ma60")) / pl.col("ma60")).alias("dist_ma60"),
        pl.col("ma60").diff().alias("slope_ma60"),
        pl.col("idx_close").rolling_max(60).alias("high_60d"),
        pl.col("idx_close").rolling_max(120).alias("high_120d"),
    ]).with_columns([
        (pl.col("idx_close") / pl.col("high_60d") - 1).alias("dd_60d"),
        (pl.col("idx_close") / pl.col("high_120d") - 1).alias("dd_120d"),
    ]).to_pandas()
    pdf["date"] = pd.to_datetime(pdf["trade_date"]).dt.date
    return pdf


def load_truth() -> dict:
    """ZigZag ground truth (legs ≥100d)."""
    data = json.loads(open(PIVOTS_PATH).read())
    truth = {}
    for leg in data["legs"]:
        if leg["duration_days"] < 100:
            continue
        start = date.fromisoformat(leg["start"])
        end = date.fromisoformat(leg["end"])
        kind = "bear" if leg["kind"] == "down" else "bull"
        d = start
        while d <= end:
            truth[d] = kind
            d += timedelta(days=1)
    return truth


def apply_router(pdf: pd.DataFrame, vote_threshold: int, lookback: int, cum_threshold: int) -> pd.Series:
    """Apply V27 router with given params."""
    sig_count = pd.DataFrame()
    for name, fn in BEAR_SIGNAL_SPECS:
        sig_count[name] = fn(pdf).fillna(False)
    pdf = pdf.copy()
    pdf["v27_vote_count"] = sig_count.sum(axis=1)
    pdf["v27_v2_flag"] = (pdf["v27_vote_count"] >= vote_threshold).astype(int)
    pdf["v27_cum"] = pdf["v27_v2_flag"].rolling(lookback, min_periods=1).sum()
    pdf["is_bear"] = (pdf["v27_cum"] >= cum_threshold).astype(int)
    return pdf.set_index("date")["is_bear"]


def evaluate(pred: pd.Series, truth: dict) -> dict:
    common = set(truth.keys()) & set(pred.index)
    if not common:
        return {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "precision": 0, "recall": 0, "specificity": 0,
                "f1": 0, "accuracy": 0, "n_bear_pred": 0, "n_common": 0}
    tp = sum(1 for d in common if pred.loc[d] == 1 and truth[d] == "bear")
    fp = sum(1 for d in common if pred.loc[d] == 1 and truth[d] == "bull")
    tn = sum(1 for d in common if pred.loc[d] == 0 and truth[d] == "bull")
    fn = sum(1 for d in common if pred.loc[d] == 0 and truth[d] == "bear")
    p_ = tp / (tp + fp) if (tp + fp) > 0 else 0
    r_ = tp / (tp + fn) if (tp + fn) > 0 else 0
    s_ = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": p_, "recall": r_, "specificity": s_,
            "f1": f1_, "accuracy": acc,
            "n_bear_pred": int(pred.sum()),
            "n_common": len(common)}


def main():
    log("=" * 80)
    log("V27 Router Param Sweep (ZigZag ground truth 验证)")
    log("=" * 80)
    log("Loading daily idx + ground truth...")
    pdf = build_daily_idx()
    truth = load_truth()
    log(f"  daily rows: {len(pdf)}")
    log(f"  truth dates: {len(truth)} ({sum(1 for v in truth.values() if v=='bear')} bear)")

    sweep_grid = []
    for vt in [2, 3, 4]:
        for lb in [5, 10, 15, 20]:
            for ct in [3, 4, 5, 6]:
                sweep_grid.append((vt, lb, ct))
    log(f"Sweep grid: {len(sweep_grid)} combos")

    results = {}
    for vt, lb, ct in sweep_grid:
        try:
            pred = apply_router(pdf, vt, lb, ct)
            m = evaluate(pred, truth)
            results[(vt, lb, ct)] = m
        except Exception as e:
            results[(vt, lb, ct)] = {"error": str(e)}

    # Sort by F1
    rows = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: -x[1]["f1"]
    )

    log("\n" + "=" * 80)
    log("Sweep 结果 (按 F1 score 排序)")
    log("=" * 80)
    log(f"{'vote':>5} {'lookback':>8} {'cum':>4} | {'Precision':>9} {'Recall':>7} {'Spec':>6} {'F1':>6} {'Acc':>6} | {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'bear_pred':>9}")
    for (vt, lb, ct), m in rows:
        log(f"  {vt:>3}  {lb:>7}  {ct:>3} |   {m['precision']*100:>7.1f}%  {m['recall']*100:>5.1f}%  {m['specificity']*100:>4.1f}%  {m['f1']:>5.3f}  {m['accuracy']*100:>4.1f}% | {m['tp']:>4} {m['fp']:>4} {m['tn']:>4} {m['fn']:>4} {m['n_bear_pred']:>9}")

    # Best by different metrics
    log("\n=== Best by Precision (≥0.85) ===")
    for k, m in sorted(rows, key=lambda x: -x[1]["precision"])[:5]:
        log(f"  vt={k[0]} lb={k[1]} ct={k[2]}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% F1={m['f1']:.3f}")

    log("\n=== Best by F1 ===")
    for k, m in rows[:5]:
        log(f"  vt={k[0]} lb={k[1]} ct={k[2]}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% F1={m['f1']:.3f}")

    log("\n=== Best by Specificity (≥0.5) ===")
    for k, m in sorted([(k, v) for k, v in results.items() if "error" not in v and v["specificity"] >= 0.5],
                       key=lambda x: -x[1]["f1"])[:5]:
        log(f"  vt={k[0]} lb={k[1]} ct={k[2]}: P={m['precision']*100:.1f}% R={m['recall']*100:.1f}% Spec={m['specificity']*100:.1f}% F1={m['f1']:.3f}")

    # Save
    serial = {f"vt{k[0]}_lb{k[1]}_ct{k[2]}": v for k, v in results.items()}
    with open(OUT / "sweep_summary.json", "w") as f:
        json.dump(serial, f, indent=2, ensure_ascii=False)
    log(f"\nSummary saved: {OUT/'sweep_summary.json'}")


if __name__ == "__main__":
    from pathlib import Path
    main()