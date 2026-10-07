"""Regime-specific factor lift audit.

For each regime (bull / bear / sideways, derived from ZigZag legs):
  1. For each factor column, compute cross-section rank at trade_date t
     and forward 20d return (close[t+21]/close[t] - 1) at t.
  2. Within regime, split days into top decile and bottom decile by
     factor rank; report top-decile vs bottom-decile avg fwd return.
  3. lift = top decile avg / bottom decile avg (ratio).
  4. Sort by lift descending; show top-10 per regime.

Output: per-regime top-10 table + cross-regime comparison.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

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
OUT = Path("/media/felix/f/quant/akquant-factor-backtest/evidence/regime_factor_lift")
OUT.mkdir(parents=True, exist_ok=True)

MIN_LEG_DAYS = 100
FWD_DAYS = 20  # 20d forward return
DECILE = 0.1   # top / bottom 10%


def build_regime_labels(pivots_path: Path) -> dict[date, str]:
    data = json.load(open(pivots_path))
    labels: dict[date, str] = {}
    for leg in data["legs"]:
        if leg["duration_days"] < MIN_LEG_DAYS:
            continue
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
    print("Regime-specific factor lift audit")
    print("=" * 60)

    # 1. Build per-day regime labels.
    print("\n[1/4] building regime labels from ZigZag legs (≥100d)...")
    all_dates_panel = (
        pl.scan_parquet(str(PANEL))
        .select(["trade_date"])
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .unique(subset=["trade_date"])
        .sort("trade_date")
        .collect()
    )
    all_dates = [r["trade_date"] for r in all_dates_panel.iter_rows(named=True)]
    labels = build_regime_labels(PIVOTS)
    for d in all_dates:
        if d not in labels:
            labels[d] = "sideways"
    from collections import Counter
    dist = Counter(labels.values())
    print(f"  total days: {len(labels)}")
    print(f"  bull: {dist.get('bull', 0)}, bear: {dist.get('bear', 0)}, "
          f"sideways: {dist.get('sideways', 0)}")

    # 2. Load panel with forward 20d return (keep ALL factor columns).
    print("\n[2/4] loading panel + computing forward 20d return per (date, stock)...")
    schema_keys = pl.scan_parquet(str(PANEL)).collect_schema().keys()
    base = {
        "trade_date", "ts_code", "open", "high", "low", "close", "vol",
        "amount", "adj_factor", "adj_close", "adj_factor_inferred",
        "pct_chg", "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d",
    }
    factor_cols = [c for c in schema_keys if c not in base]
    cols_to_load = ["trade_date", "ts_code", "close"] + factor_cols

    df = (
        pl.scan_parquet(str(PANEL))
        .select(cols_to_load)
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .sort(["ts_code", "trade_date"])
        .with_columns([
            (pl.col("close").shift(-FWD_DAYS).over("ts_code") / pl.col("close") - 1)
            .alias("fwd_ret")
        ])
        .collect()
    )
    print(f"  panel: {df.shape}, factor_cols={len(factor_cols)}")

    # 3. Per factor × regime lift.
    print("\n[3/4] computing lift per factor × regime...")

    # Pre-join labels.
    label_df = pl.DataFrame(
        {"trade_date": list(labels.keys()), "regime": list(labels.values())}
    ).with_columns(pl.col("trade_date").cast(pl.Date))

    # Pre-merge labels + fwd_ret into one frame.
    base_frame = df.join(label_df, on="trade_date", how="inner").filter(
        pl.col("fwd_ret").is_not_null()
    )
    print(f"  base frame after label join + fwd filter: {base_frame.shape}")

    # For each factor, compute cross-section rank + lift.
    REGIMES = ["bull", "bear", "sideways"]
    results: dict[str, list[dict]] = {r: [] for r in REGIMES}

    for fi, factor in enumerate(factor_cols):
        if (fi + 1) % 50 == 0 or fi == 0:
            print(f"  [{fi+1}/{len(factor_cols)}] {factor}")
        try:
            sub = base_frame.select(["trade_date", "regime", "fwd_ret", factor])
            # Drop NaN factor values.
            sub = sub.filter(pl.col(factor).is_not_null())
            # Compute cross-section rank (pct 0-1).
            sub = sub.with_columns(
                pl.col(factor).rank(method="ordinal").over("trade_date").alias("rk")
            )
            n_per_day = sub.group_by("trade_date").agg(pl.len().alias("n"))
            sub = sub.join(n_per_day, on="trade_date")
            # Per regime, compute top/bottom decile fwd_ret.
            regime_metrics = []
            for regime in REGIMES:
                rsub = sub.filter(pl.col("regime") == regime)
                if rsub.shape[0] == 0:
                    continue
                # Need at least 30 days of valid samples to be meaningful.
                if rsub["trade_date"].n_unique() < 30:
                    continue
                # Top / bottom decile by rank within day.
                rsub = rsub.with_columns([
                    pl.col("n").alias("_n"),
                ])
                # Decile cutoffs: top 10% = rk >= 0.9*n, bottom 10% = rk <= 0.1*n
                rsub = rsub.with_columns([
                    ((pl.col("rk") / pl.col("_n")) >= (1 - DECILE)).alias("is_top"),
                    ((pl.col("rk") / pl.col("_n")) <= DECILE).alias("is_bot"),
                ])
                top_avg = float(
                    rsub.filter(pl.col("is_top"))["fwd_ret"].mean()
                ) if rsub.filter(pl.col("is_top")).shape[0] > 0 else float("nan")
                bot_avg = float(
                    rsub.filter(pl.col("is_bot"))["fwd_ret"].mean()
                ) if rsub.filter(pl.col("is_bot")).shape[0] > 0 else float("nan")
                n_top = rsub.filter(pl.col("is_top")).shape[0]
                n_bot = rsub.filter(pl.col("is_bot")).shape[0]
                # lift = top / bot in ratio; bot < 0 ⇒ use diff instead.
                if bot_avg < 0 and not np.isnan(bot_avg):
                    lift = top_avg - bot_avg  # absolute spread
                    lift_kind = "diff"
                elif bot_avg > 0:
                    lift = top_avg / bot_avg
                    lift_kind = "ratio"
                else:
                    lift = top_avg - bot_avg
                    lift_kind = "diff"
                results[regime].append({
                    "factor": factor,
                    "n_days": rsub["trade_date"].n_unique(),
                    "n_top": n_top,
                    "n_bot": n_bot,
                    "top_avg_fwd": top_avg,
                    "bot_avg_fwd": bot_avg,
                    "lift": lift,
                    "lift_kind": lift_kind,
                })
        except Exception as exc:  # noqa: BLE001
            continue

    # 4. Per-regime top 10 + comparison.
    print("\n[4/4] top-10 factors per regime...")
    summary = {}
    for regime in REGIMES:
        rows = results[regime]
        rows.sort(key=lambda r: (r["lift"] if r["lift_kind"] == "ratio" else r["lift"]),
                  reverse=True)
        top10 = rows[:10]
        summary[regime] = {
            "n_factors_evaluated": len(rows),
            "top_10": top10,
        }
        print(f"\n  === regime = {regime} ===")
        print(f"  factors evaluated: {len(rows)}")
        print(f"  {'factor':<28}  {'n_days':>7}  {'top_fwd':>8}  "
              f"{'bot_fwd':>8}  {'lift':>7}  {'kind':>6}")
        print("  " + "-" * 80)
        for r in top10:
            print(
                f"  {r['factor']:<28}  {r['n_days']:>7}  "
                f"{r['top_avg_fwd']*100:>+7.2f}%  "
                f"{r['bot_avg_fwd']*100:>+7.2f}%  "
                f"{r['lift']:>+7.2f}  "
                f"{r['lift_kind']:>6}"
            )

    # Cross-regime overlap: which factors appear in top-10 of multiple regimes?
    print("\n  === Cross-regime top-10 overlap ===")
    top_sets = {r: {x['factor'] for x in summary[r]['top_10']} for r in REGIMES}
    all_top = top_sets["bull"] | top_sets["bear"] | top_sets["sideways"]
    in_all_3 = top_sets["bull"] & top_sets["bear"] & top_sets["sideways"]
    in_bull_bear = (top_sets["bull"] & top_sets["bear"]) - in_all_3
    in_bull_only = top_sets["bull"] - top_sets["bear"] - top_sets["sideways"]
    print(f"  Total unique in any top-10: {len(all_top)}")
    print(f"  In ALL 3 regimes: {len(in_all_3)}  {sorted(in_all_3)}")
    print(f"  In bull & bear only: {len(in_bull_bear)}  {sorted(in_bull_bear)}")
    print(f"  Bull-only (regime-specific): {len(in_bull_only)}  {sorted(in_bull_only)}")

    # Save artefact.
    artefact = {
        "regimes": REGIMES,
        "fwd_days": FWD_DAYS,
        "decile": DECILE,
        "min_leg_days": MIN_LEG_DAYS,
        "summary": summary,
        "overlap": {
            "in_all_3": sorted(in_all_3),
            "in_bull_bear_only": sorted(in_bull_bear),
            "bull_only": sorted(in_bull_only),
        },
    }
    out_path = OUT / "regime_factor_lift.json"
    out_path.write_text(json.dumps(artefact, indent=2, default=str))
    print(f"\n  artefact: {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())