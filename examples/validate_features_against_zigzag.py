"""Step 1: Validate whether panel features can discriminate between ZigZag regime labels.

Methodology (verification-driven):
  1. Build per-trade_date label from ZigZag pivots:
       - bull   = date is inside an 'up' leg
       - bear   = date is inside a 'down' leg
       - sideways = date is outside any ≥100d leg
  2. Pick a small set of features that are (a) cross-sectional means so
     we get one value per trade_date, and (b) computable from data
     observable at the close of trade_date (no lookahead).
  3. For each feature, report:
       - per-class mean / median / std
       - KS test between bull and bear (small p ⇒ separable)
       - effect size (mean diff / pooled std)
  4. If KS p > 0.01 for ALL features ⇒ the feature set cannot
     discriminate regime and the classifier path is not viable.
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats  # KS test

PIVOTS = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "hs300_index_pivots_clean_20260919/"
    "hs300_index_pivots.json"
)
PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)
OUT = Path("/media/felix/f/quant/akquant-factor-backtest/evidence/regime_classifier_validation")
OUT.mkdir(parents=True, exist_ok=True)

MIN_LEG_DAYS = 100  # legs shorter than this count as sideways material


def build_labels(pivots_path: Path, min_leg_days: int) -> dict[date, str]:
    """Per-day ZigZag label. Bull/bear only for legs ≥ min_leg_days."""
    data = json.load(open(pivots_path))
    labels: dict[date, str] = {}
    for leg in data["legs"]:
        if leg["duration_days"] < min_leg_days:
            continue  # too short — treat as sideways
        start = date.fromisoformat(leg["start"])
        end = date.fromisoformat(leg["end"])
        kind = "bull" if leg["kind"] == "up" else "bear"
        d = start
        while d <= end:
            labels[d] = kind
            d += timedelta(days=1)
    return labels


def main() -> int:
    print("=" * 60)
    print("Step 1: feature separation under ZigZag regime labels")
    print("=" * 60)

    # 1. Build ZigZag labels.
    print("\n[1/4] building per-day ZigZag labels (legs ≥ "
          f"{MIN_LEG_DAYS}d are bull/bear, rest = sideways)...")
    labels = build_labels(PIVOTS, MIN_LEG_DAYS)
    dist = Counter(labels.values())
    print(f"  labeled days: {len(labels)}")
    print(f"  bull: {dist.get('bull', 0)}, bear: {dist.get('bear', 0)}, "
          f"sideways (no leg): not in dict")

    # 2. Aggregate panel features to per-trade_date (cross-section mean).
    print("\n[2/4] loading panel features and aggregating to per-day mean...")
    # Cross-section means: each of the 354 stocks contributes one value per
    # date; we collapse to a single market-level series by averaging.
    daily_features = (
        pl.scan_parquet(str(PANEL))
        .select([
            "trade_date",
            "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
            "mw_above_ma5_xs_rank", "mw_above_ma20_xs_rank",
            "mw_above_ma60_xs_rank",
            "mw_vol_20d_xs_rank",
            "mw_align_20d_xs_rank", "mw_align_60d_xs_rank",
            "mw_ret_20d_xs_rank", "mw_ret_60d_xs_rank",
            "talib_RSI", "talib_ATR",
        ])
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .group_by("trade_date")
        .agg([
            pl.col(c).mean().alias(c) for c in [
                "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
                "mw_above_ma5_xs_rank", "mw_above_ma20_xs_rank",
                "mw_above_ma60_xs_rank",
                "mw_vol_20d_xs_rank",
                "mw_align_20d_xs_rank", "mw_align_60d_xs_rank",
                "mw_ret_20d_xs_rank", "mw_ret_60d_xs_rank",
                "talib_RSI", "talib_ATR",
            ]
        ])
        .sort("trade_date")
        .collect()
    )
    print(f"  daily rows: {daily_features.shape[0]}")

    # 3. Per-class summary stats + KS test (bull vs bear).
    print("\n[3/4] per-feature separation between bull and bear labels...")
    feature_cols = [
        "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
        "mw_above_ma5_xs_rank", "mw_above_ma20_xs_rank",
        "mw_above_ma60_xs_rank",
        "mw_vol_20d_xs_rank",
        "mw_align_20d_xs_rank", "mw_align_60d_xs_rank",
        "mw_ret_20d_xs_rank", "mw_ret_60d_xs_rank",
        "talib_RSI", "talib_ATR",
    ]

    rows = []
    for col in feature_cols:
        # Build per-class arrays from rows where label is known.
        per_class = defaultdict(list)
        for r in daily_features.iter_rows(named=True):
            d = r["trade_date"]
            v = r[col]
            if v is None or np.isnan(v):
                continue
            if d in labels:  # bull or bear only — sideways ignored
                per_class[labels[d]].append(float(v))

        # Per-class summary.
        summary = {}
        for cls in ["bull", "bear"]:
            arr = np.array(per_class[cls])
            if len(arr) == 0:
                summary[cls] = {"n": 0}
                continue
            summary[cls] = {
                "n": len(arr),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr)),
            }

        # KS test bull vs bear.
        bull_arr = np.array(per_class["bull"])
        bear_arr = np.array(per_class["bear"])
        if len(bull_arr) > 5 and len(bear_arr) > 5:
            ks = stats.ks_2samp(bull_arr, bear_arr)
            ks_p = float(ks.pvalue)
            ks_stat = float(ks.statistic)
            pooled_var = (np.var(bull_arr) + np.var(bear_arr)) / 2
            pooled_std = float(np.sqrt(pooled_var)) if pooled_var > 0 else 1.0
            if pooled_std > 0:
                effect_size = float(
                    (np.mean(bull_arr) - np.mean(bear_arr)) / pooled_std
                )
            else:
                effect_size = 0.0
        else:
            ks_p = 1.0
            ks_stat = 0.0
            effect_size = 0.0

        rows.append({
            "feature": col,
            "n_bull": summary["bull"].get("n", 0),
            "n_bear": summary["bear"].get("n", 0),
            "bull_mean": summary["bull"].get("mean"),
            "bull_median": summary["bull"].get("median"),
            "bear_mean": summary["bear"].get("mean"),
            "bear_median": summary["bear"].get("median"),
            "mean_diff": (
                (summary["bull"].get("mean") or 0) - (summary["bear"].get("mean") or 0)
            ),
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "effect_size_cohen_d": float(effect_size),
            "separable": ks_p < 0.01 and abs(effect_size) > 0.2,
        })

    # Print table.
    print(f"\n  {'feature':<28}  {'bull mean':>10}  {'bear mean':>10}  "
          f"{'mean_diff':>10}  {'cohen_d':>8}  {'KS p':>10}  {'sep':>4}")
    print("  " + "-" * 100)
    for r in rows:
        print(
            f"  {r['feature']:<28}  "
            f"{(r['bull_mean'] or 0):>+10.4f}  "
            f"{(r['bear_mean'] or 0):>+10.4f}  "
            f"{(r['mean_diff'] or 0):>+10.4f}  "
            f"{r['effect_size_cohen_d']:>+8.3f}  "
            f"{r['ks_pvalue']:>10.2e}  "
            f"{'YES' if r['separable'] else 'no':>4}"
        )

    n_sep = sum(1 for r in rows if r["separable"])
    print(f"\n  Separable features (KS p<0.01 AND |d|>0.2): {n_sep}/{len(rows)}")

    if n_sep == 0:
        print("  ⚠️  NO feature separates bull vs bear. Regime classifier is "
              "not viable with current feature set.")
    elif n_sep < 3:
        print(f"  ⚠️  Only {n_sep} feature(s) separate. Marginal signal — "
              "classifier may not beat random.")
    else:
        print(f"  ✓  {n_sep} features show separation — classifier worth trying.")

    # 4. Save artefact.
    (OUT / "feature_separation.json").write_text(json.dumps(rows, indent=2))
    print(f"\n  artefact: {OUT / 'feature_separation.json'}")

    # Bonus: cross-sectional dispersion per regime (do stocks disperse
    # differently in bull vs bear vs sideways?)
    print("\n[4/4] cross-sectional dispersion by regime...")
    # We measure the std of mw_ret_20d across stocks per day (breadth).
    daily_breadth = (
        pl.scan_parquet(str(PANEL))
        .select(["trade_date", "mw_ret_20d_xs_rank"])
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .group_by("trade_date")
        .agg(pl.col("mw_ret_20d_xs_rank").std().alias("breadth_std"))
        .sort("trade_date")
        .collect()
    )
    per_class_breadth = defaultdict(list)
    for r in daily_breadth.iter_rows(named=True):
        d = r["trade_date"]
        v = r["breadth_std"]
        if v is None or np.isnan(v):
            continue
        if d in labels:
            per_class_breadth[labels[d]].append(float(v))

    print(f"  Cross-sectional std of mw_ret_20d_xs_rank (breadth proxy):")
    for cls in ["bull", "bear"]:
        arr = np.array(per_class_breadth[cls])
        if len(arr) > 0:
            print(
                f"    {cls:>5}: n={len(arr)}, mean={np.mean(arr):.4f}, "
                f"median={np.median(arr):.4f}, std={np.std(arr):.4f}"
            )
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())