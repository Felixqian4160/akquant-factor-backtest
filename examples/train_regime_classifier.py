"""Step 1.5: LightGBM 3-class regime classifier, 5-fold walk-forward.

Verification:
  - Build ZigZag labels (bull/bear/sideways, legs ≥100d)
  - Build feature set: idx_ret_* + talib_RSI + talib_ATR + talib_BBANDS_*
    + talib_CCI + talib_WILLR + talib_ADX + talib_MACD_0/1/2 (cross-section
    means, no lookahead). Skip the xs_rank columns (they are
    cross-section normalized to mean 0.5 by construction and useless).
  - 5-fold walk-forward (no random shuffle): each fold trains on all
    rows up to fold_end and predicts the next ~20% window.
  - Per-fold macro AUC + accuracy; combined OOF confusion matrix.
  - Baseline: idx_ret_60d rule (bull>5%, bear<-5%, sideways else)
    on the same OOF rows for head-to-head comparison.

Output: 5-fold AUC, accuracy, confusion matrix JSON, comparison table.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

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

MIN_LEG_DAYS = 100
LABELS = ["bear", "sideways", "bull"]  # alpha-sorted for stable AUC


def build_labels(pivots_path: Path, all_dates: list[date]) -> dict[date, str]:
    """Per-day ZigZag label covering EVERY date in all_dates.

    - bull   = date is inside an 'up' leg of ≥100d
    - bear   = date is inside a 'down' leg of ≥100d
    - sideways = every other date (including short legs and gaps)

    The previous version only emitted bull/bear, leaving 'sideways' as
    'no label' which crashed the multi-class AUC.
    """
    data = json.load(open(pivots_path))
    labels: dict[date, str] = {d: "sideways" for d in all_dates}
    for leg in data["legs"]:
        if leg["duration_days"] < MIN_LEG_DAYS:
            continue
        start = date.fromisoformat(leg["start"])
        end = date.fromisoformat(leg["end"])
        kind = "bull" if leg["kind"] == "up" else "bear"
        d = start
        while d <= end:
            if d in labels:
                labels[d] = kind
            d += timedelta(days=1)
    return labels


FEATURE_COLS = [
    # Index returns (true market signals, not normalized)
    "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
    # TA-Lib cross-section means (real values, not xs_rank)
    "talib_RSI", "talib_ATR",
    "talib_BBANDS_0", "talib_BBANDS_2",  # upper / lower
    "talib_CCI", "talib_WILLR", "talib_ADX",
    "talib_MACD_0", "talib_MACD_1", "talib_MACD_2",
    # Cross-section realized vol from mw_* (these are NOT xs_rank)
    "mw_vol_20d",
    "mw_align_20d", "mw_align_60d",
]

# Extra cross-section features computed via polars rolling (not
# present in the panel; aggregated from raw OHLCV to one value per
# trade_date).
EXTRA_FEATURE_COLS = [
    "xs_ts_mean_5", "xs_ts_mean_20", "xs_ts_mean_60",
    "xs_ts_std_20", "xs_ts_std_60",
    "xs_intraday_range",
    "xs_ret_20d", "xs_ret_60d",
]

ALL_FEATURE_COLS = FEATURE_COLS + EXTRA_FEATURE_COLS


def main() -> int:
    print("=" * 60)
    print("Step 1.5: LightGBM 3-class regime classifier + 5-fold WF")
    print("=" * 60)

    # 2. Daily features.
    print("\n[2/5] loading + aggregating per-day features...")
    # Step 2a: panel-level features (cross-section means of existing cols).
    daily_panel = (
        pl.scan_parquet(str(PANEL))
        .select(["trade_date"] + FEATURE_COLS)
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .group_by("trade_date")
        .agg([pl.col(c).mean().alias(c) for c in FEATURE_COLS])
        .sort("trade_date")
        .collect()
    )
    # Step 2b: extra rolling features computed from raw OHLCV.
    raw = pl.scan_parquet(str(PANEL)).select(
        ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
    ).with_columns(pl.col("trade_date").cast(pl.Date)).collect()
    raw_with = raw.sort(["ts_code", "trade_date"]).with_columns([
        pl.col("close").rolling_mean(5).over("ts_code").alias("ts_mean_5"),
        pl.col("close").rolling_mean(20).over("ts_code").alias("ts_mean_20"),
        pl.col("close").rolling_mean(60).over("ts_code").alias("ts_mean_60"),
        pl.col("close").rolling_std(20).over("ts_code").alias("ts_std_20"),
        pl.col("close").rolling_std(60).over("ts_code").alias("ts_std_60"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("intraday_range"),
        (pl.col("close") / pl.col("close").shift(20).over("ts_code") - 1).alias("ret_20d"),
        (pl.col("close") / pl.col("close").shift(60).over("ts_code") - 1).alias("ret_60d"),
    ])
    daily_extra = (
        raw_with.group_by("trade_date")
        .agg([pl.col(c).mean().alias("xs_" + c) for c in [
            "ts_mean_5", "ts_mean_20", "ts_mean_60",
            "ts_std_20", "ts_std_60",
            "intraday_range",
            "ret_20d", "ret_60d",
        ]])
        .sort("trade_date")
    )
    # Inner join on trade_date.
    daily = daily_panel.join(daily_extra, on="trade_date", how="inner")
    print(f"  daily rows: {daily.shape[0]}  (panel={len(daily_panel)} extra={len(daily_extra)})")
    print(f"  feature count: {len(ALL_FEATURE_COLS)} ({len(FEATURE_COLS)} panel + {len(EXTRA_FEATURE_COLS)} computed)")

    # 1. Labels (after we know all dates).
    print("\n[1/5] building ZigZag labels...")
    all_dates = [r["trade_date"] for r in daily.iter_rows(named=True)]
    labels = build_labels(PIVOTS, all_dates)
    label_dist = Counter(labels.values())
    print(f"  labeled days: {len(labels)}  "
          f"bull={label_dist.get('bull', 0)}  "
          f"bear={label_dist.get('bear', 0)}  "
          f"sideways={label_dist.get('sideways', 0)}")

    # Build the modelling frame: rows that have a label + features.
    rows = []
    for r in daily.iter_rows(named=True):
        d = r["trade_date"]
        if d not in labels:
            continue
        feat = [r[c] for c in ALL_FEATURE_COLS]
        if any(v is None or np.isnan(v) for v in feat):
            continue
        rows.append((d, labels[d], feat))

    rows.sort(key=lambda x: x[0])
    print(f"  labelled + non-null rows: {len(rows)}")

    dates_arr = np.array([r[0] for r in rows])
    y_arr = np.array([r[1] for r in rows])
    X_arr = np.array([r[2] for r in rows])
    label_codes = np.array([LABELS.index(y) for y in y_arr])

    # 3. 5-fold walk-forward split (time-ordered).
    print("\n[3/5] 5-fold walk-forward split (time-ordered)...")
    n = len(rows)
    fold_size = n // 5
    fold_splits = []
    for k in range(5):
        # Train: rows 0 .. train_end
        # OOS: rows [oof_start .. oof_end]
        # Each fold expands training set progressively.
        train_end = (k + 1) * fold_size  # expanding
        oof_start = train_end
        oof_end = min(train_end + fold_size, n)
        if oof_start >= n:
            break
        fold_splits.append((k, train_end, oof_start, oof_end))
        print(f"  fold {k}: train=[0..{train_end-1}]  OOF=[{oof_start}..{oof_end-1}]  "
              f"({oof_end-oof_start} OOF rows)")

    # 4. Train + OOF predict.
    print("\n[4/5] training LightGBM per fold...")
    oof_pred_proba = np.full((n, 3), np.nan)
    fold_metrics = []
    for k, train_end, oof_start, oof_end in fold_splits:
        X_train = X_arr[:train_end]
        y_train = label_codes[:train_end]
        X_oof = X_arr[oof_start:oof_end]
        y_oof = label_codes[oof_start:oof_end]

        # Drop NaN in y_train (none expected after filtering).
        valid = ~np.isnan(y_train)
        X_train = X_train[valid]
        y_train = y_train[valid]

        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            random_state=42 + k,
            verbosity=-1,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_oof)
        oof_pred_proba[oof_start:oof_end] = proba
        pred = np.argmax(proba, axis=1)

        # Per-fold AUC (one-vs-rest macro).
        try:
            auc = roc_auc_score(y_oof, proba, multi_class="ovr", average="macro",
                                labels=[0, 1, 2])
        except Exception:
            auc = float("nan")
        acc = accuracy_score(y_oof, pred)
        fold_metrics.append({
            "fold": k,
            "train_rows": int(train_end),
            "oof_rows": int(oof_end - oof_start),
            "auc_macro": float(auc),
            "accuracy": float(acc),
        })
        print(f"  fold {k}: AUC={auc:.3f}  Acc={acc:.3f}  "
              f"n_train={train_end}  n_oof={oof_end-oof_start}")

    # 5. Combined OOF metrics.
    print("\n[5/5] combined OOF metrics + baseline comparison...")
    valid_oof = ~np.isnan(oof_pred_proba[:, 0])
    y_oof_all = label_codes[valid_oof]
    proba_all = oof_pred_proba[valid_oof]
    pred_all = np.argmax(proba_all, axis=1)

    auc_macro = roc_auc_score(y_oof_all, proba_all, multi_class="ovr",
                              average="macro", labels=[0, 1, 2])
    auc_weighted = roc_auc_score(y_oof_all, proba_all, multi_class="ovr",
                                 average="weighted", labels=[0, 1, 2])
    acc_all = accuracy_score(y_oof_all, pred_all)
    cm = confusion_matrix(y_oof_all, pred_all, labels=[0, 1, 2])

    print(f"\n  === Combined OOF metrics ===")
    print(f"  AUC macro:    {auc_macro:.3f}")
    print(f"  AUC weighted: {auc_weighted:.3f}")
    print(f"  Accuracy:     {acc_all:.3f}")
    print(f"  OOF rows:     {len(y_oof_all)}")

    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {'':>12}  {'bear':>8}  {'sideways':>10}  {'bull':>8}")
    for i, lab in enumerate(LABELS):
        print(f"  {lab:>12}  {cm[i, 0]:>8}  {cm[i, 1]:>10}  {cm[i, 2]:>8}")

    # Baseline: idx_ret_60d threshold rule on the same OOF rows.
    idx_ret_60d_idx = FEATURE_COLS.index("idx_ret_60d")
    baseline_pred = []
    for k, train_end, oof_start, oof_end in fold_splits:
        for i in range(oof_start, oof_end):
            r = X_arr[i, idx_ret_60d_idx]
            if r > 0.05:
                baseline_pred.append(2)  # bull
            elif r < -0.05:
                baseline_pred.append(0)  # bear
            else:
                baseline_pred.append(1)  # sideways
    baseline_pred = np.array(baseline_pred)
    base_acc = accuracy_score(y_oof_all, baseline_pred[: len(y_oof_all)])

    print(f"\n  === Baseline (idx_ret_60d >5%/<-5% rule) ===")
    print(f"  Accuracy:  {base_acc:.3f}")

    # Head-to-head.
    print(f"\n  === Head-to-head (LightGBM vs idx_ret_60d rule) ===")
    print(f"  LightGBM AUC macro: {auc_macro:.3f}    Acc: {acc_all:.3f}")
    print(f"  idx_ret_60d Acc:    {base_acc:.3f}    (AUC not defined for hard rule)")
    print(f"  Verdict: " + (
        "✅ LightGBM BEATS baseline" if auc_macro > 0.55 and acc_all > base_acc + 0.05
        else "⚠️  LightGBM MARGINAL" if auc_macro > 0.5
        else "❌ LightGBM LOSES to random"
    ))

    # Feature importance (use last fold's model).
    last_fold_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=3, random_state=42 + 4, verbosity=-1,
    )
    last_fold_model.fit(X_arr, label_codes)
    fi = last_fold_model.feature_importances_
    fi_sorted = sorted(zip(ALL_FEATURE_COLS, fi), key=lambda x: -x[1])

    print(f"\n  Feature importance (gain across all classes, last fold):")
    for fname, fimp in fi_sorted[:8]:
        print(f"    {fname:<28}  {fimp:>6.0f}")

    # Save artefact.
    artefact = {
        "feature_cols": ALL_FEATURE_COLS,
        "n_total_rows": n,
        "n_oof_rows": int(len(y_oof_all)),
        "labels": LABELS,
        "fold_metrics": fold_metrics,
        "combined": {
            "auc_macro": float(auc_macro),
            "auc_weighted": float(auc_weighted),
            "accuracy": float(acc_all),
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABELS,
        "baseline_idx_ret_60d_accuracy": float(base_acc),
        "feature_importance": [
            {"feature": f, "importance": float(v)}
            for f, v in fi_sorted
        ],
    }
    out_path = OUT / "lightgbm_regime_classifier.json"
    out_path.write_text(json.dumps(artefact, indent=2))
    print(f"\n  artefact: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())