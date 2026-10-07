"""Validate idx_mom_60 regime classifier against zigzag ground-truth labels.

Hypothesis:
  My previous rule-based classifier
      bull if idx_mom_60 > 5%, bear if idx_mom_60 < -5%, else sideways
  is *bad* — using ZigZag-derived leg labels as ground truth will expose it.

Plan:
  1. Build per-trade_date label from hs300_index_pivots.json:
       - 'bull' if date is inside an 'up' leg
       - 'bear' if date is inside a 'down' leg
       - 'sideways' otherwise (short legs, gaps, early data)
  2. Build per-trade_date prediction from idx_mom_60:
       - 'bull' if idx_ret_60d > 5%
       - 'bear' if idx_ret_60d < -5%
       - 'sideways' otherwise
  3. Show confusion matrix, per-class precision/recall, agreement rate.
"""
from __future__ import annotations

import json
import polars as pl
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict, Counter

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


def build_truth_labels(pivots_path: Path) -> dict[date, str]:
    """Map trade_date → label based on which leg it falls in."""
    data = json.load(open(pivots_path))
    legs = data["legs"]

    label_map: dict[date, str] = {}
    for leg in legs:
        start = date.fromisoformat(leg["start"])
        end = date.fromisoformat(leg["end"])
        kind = "bull" if leg["kind"] == "up" else "bear"
        d = start
        while d <= end:
            label_map[d] = kind
            d += timedelta(days=1)
    return label_map


def main() -> int:
    print("=" * 60)
    print("Validating idx_mom_60 classifier against ZigZag ground truth")
    print("=" * 60)

    # 1. Truth labels from ZigZag pivots.
    print("\n[1/3] building per-day truth labels from ZigZag pivots...")
    truth = build_truth_labels(PIVOTS)
    truth_dist = Counter(truth.values())
    print(f"  labeled dates: {len(truth)}")
    print(f"  truth distribution: {dict(truth_dist)}")
    print(f"    bull: {truth_dist.get('bull', 0)}")
    print(f"    bear: {truth_dist.get('bear', 0)}")
    print(f"  (note: sideways dates are simply dates with no leg cover)")

    # 2. Get idx_ret_60d from panel.
    print("\n[2/3] loading idx_ret_60d from panel...")
    # Per-trade_date idx_ret_60d — take first ts_code's row (same per day).
    df = (
        pl.scan_parquet(str(PANEL))
        .select(["trade_date", "idx_ret_60d"])
        .filter(pl.col("idx_ret_60d").is_not_null())
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .unique(subset=["trade_date"])
        .sort("trade_date")
        .collect()
    )
    print(f"  daily rows: {df.shape[0]}")
    print(f"  idx_ret_60d: mean={df['idx_ret_60d'].mean():.3f}, "
          f"std={df['idx_ret_60d'].std():.3f}, "
          f"min={df['idx_ret_60d'].min():.3f}, max={df['idx_ret_60d'].max():.3f}")

    # 3. Apply my classifier and compare.
    print("\n[3/3] applying classifier + computing confusion matrix...")
    THRESHOLD = 0.05  # ±5%
    predicted = {}
    for row in df.iter_rows(named=True):
        d = row["trade_date"]
        r = row["idx_ret_60d"]
        if r > THRESHOLD:
            predicted[d] = "bull"
        elif r < -THRESHOLD:
            predicted[d] = "bear"
        else:
            predicted[d] = "sideways"

    pred_dist = Counter(predicted.values())
    print(f"  predicted distribution: {dict(pred_dist)}")

    # Build confusion matrix on the dates that have BOTH a truth label AND a prediction.
    LABELS = ["bull", "sideways", "bear"]
    cm = {gt: {pred: 0 for pred in LABELS} for gt in LABELS}
    matched = 0
    total_eval = 0
    for d, gt_label in truth.items():
        if d in predicted:
            pred_label = predicted[d]
            cm[gt_label][pred_label] += 1
            total_eval += 1
            if gt_label == pred_label:
                matched += 1

    print(f"\n  eval days: {total_eval}")
    print(f"  agreement (diagonal): {matched}/{total_eval} = {matched/total_eval:.1%}")

    # Confusion matrix table.
    print("\n  Confusion Matrix (rows=truth from ZigZag, cols=prediction from idx_mom_60):")
    print(f"  {'':>12}  {'bull':>8}  {'sideways':>10}  {'bear':>8}  {'total':>8}")
    for gt in LABELS:
        row = cm[gt]
        s = sum(row.values())
        print(
            f"  {gt:>12}  {row['bull']:>8}  {row['sideways']:>10}  "
            f"{row['bear']:>8}  {s:>8}"
        )

    # Per-class metrics.
    print("\n  Per-class precision / recall / F1:")
    print(f"  {'class':>10}  {'P':>6}  {'R':>6}  {'F1':>6}  {'support':>8}")
    for cls in LABELS:
        tp = cm[cls][cls]
        fp = sum(cm[other][cls] for other in LABELS if other != cls)
        fn = sum(cm[cls][other] for other in LABELS if other != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = sum(cm[cls].values())
        print(f"  {cls:>10}  {prec:>6.2f}  {rec:>6.2f}  {f1:>6.2f}  {support:>8}")

    # Failure modes worth flagging.
    bull_days = sum(cm["bull"].values())
    bull_pred_bull = cm["bull"]["bull"]
    print(f"\n  Bull-leg days: {bull_days}  correctly predicted bull: {bull_pred_bull} "
          f"({bull_pred_bull/bull_days:.1%})")
    bear_days = sum(cm["bear"].values())
    bear_pred_bear = cm["bear"]["bear"]
    print(f"  Bear-leg days: {bear_days}  correctly predicted bear: {bear_pred_bear} "
          f"({bear_pred_bear/bear_days:.1%})")

    # Look at idx_ret_60d distribution within each truth class.
    by_class: dict[str, list[float]] = defaultdict(list)
    for d, gt in truth.items():
        if d in predicted:
            row = df.filter(pl.col("trade_date") == d)
            if row.shape[0] > 0:
                by_class[gt].append(float(row["idx_ret_60d"][0]))

    print(f"\n  idx_ret_60d distribution by truth class:")
    for cls in LABELS:
        vals = by_class[cls]
        if vals:
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            print(
                f"    {cls:>10}: n={n}, "
                f"min={vals_sorted[0]:+.3f}, "
                f"median={vals_sorted[n//2]:+.3f}, "
                f"max={vals_sorted[-1]:+.3f}"
            )

    # Save artefacts.
    out = Path("/media/felix/f/quant/akquant-factor-backtest/evidence/regime_classifier_validation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "confusion_matrix.json").write_text(json.dumps(cm, indent=2))
    print(f"\n  confusion_matrix.json: {out / 'confusion_matrix.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())