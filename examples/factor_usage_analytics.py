"""Factor usage analytics — re-derive per-rebal-date top-10 factor selection
and aggregate usage statistics.

Reads:
  evidence/sweep/v33_factorrank_20261007/V14_0/picks_meta.json  (for regime info)
  evidence/causal_zigzag_router_20261006/router_map.json

Writes:
  evidence/sweep/v33_factorrank_20261007/factor_usage_stats.json
  evidence/sweep/v33_factorrank_20261007/factor_usage_top50.csv
  evidence/sweep/v33_factorrank_20261007/factor_usage_by_regime.json
  evidence/sweep/v33_factorrank_20261007/factor_usage_turnover.csv
"""
from __future__ import annotations

import json
import time
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
ROUTER_MAP = Path("evidence/causal_zigzag_router_20261006/router_map.json")
OUT_DIR = Path("evidence/sweep/v33_factorrank_20261007")

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
ENTRY_TO_EXIT_SHIFT = 21
LOOKBACK_SESSIONS = 60
TOP_K = 10
K_NOM = 10
ROUND_TRIP_COST = 0.005


def log(msg: str) -> None:
    ts = _dt.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def dt_expr(d) -> pl.Expr:
    if isinstance(d, str):
        ts = pd.Timestamp(d).to_pydatetime()
    else:
        ts = pd.Timestamp(d).to_pydatetime()
    return pl.lit(ts, dtype=pl.Datetime("ms"))


def trading_dates() -> list[str]:
    df = pl.scan_parquet(PANEL).select(["trade_date"]).unique().sort("trade_date").collect()
    return [str(x)[:10] for x in df["trade_date"].to_list()]


def load_router() -> dict[str, str]:
    return json.loads(ROUTER_MAP.read_text())


def compute_factor_returns_polars(panel_full: pl.DataFrame, factors: list[str],
                                   all_dates_iso: list[str], lookback: int
                                   ) -> dict[str, dict[str, float]]:
    log(f"  Polars factor returns, {len(factors)} factors × {len(all_dates_iso)} dates")
    t0 = time.time()
    n_dates = len(all_dates_iso)
    out: dict[str, dict[str, float]] = {d: {} for d in all_dates_iso[lookback:]}
    for fi, f in enumerate(factors, start=1):
        df = panel_full.select(["trade_date", "_fwd_net", f])
        df = df.filter(pl.col(f).is_not_null() & pl.col("_fwd_net").is_not_null())
        df = df.with_columns(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk")
        )
        per_date = df.group_by("trade_date").agg([
            (pl.col("_fwd_net").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top_mean"),
            (pl.col("_fwd_net").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot_mean"),
        ]).sort("trade_date").select(["trade_date", (pl.col("_top_mean") - pl.col("_bot_mean")).alias("_diff")])
        date_pos = []
        diffs = []
        for row in per_date.iter_rows(named=True):
            d_str = str(row["trade_date"])[:10]
            pos = all_dates_iso.index(d_str) if d_str in all_dates_iso else -1
            if pos >= 0 and np.isfinite(row["_diff"]):
                date_pos.append(pos)
                diffs.append(row["_diff"])
        if not date_pos:
            continue
        date_pos_arr = np.asarray(date_pos, dtype=np.int64)
        diffs_arr = np.asarray(diffs, dtype=np.float64)
        for i in range(lookback, n_dates):
            d_iso = all_dates_iso[i]
            mask = date_pos_arr < i
            if mask.sum() < 10:
                continue
            window = diffs_arr[mask][-lookback:]
            window = window[np.isfinite(window)]
            if len(window) >= 10:
                out[d_iso][f] = float(np.mean(window))
        if fi % 25 == 0:
            log(f"    factor {fi}/{len(factors)}: {f}, elapsed {time.time()-t0:.0f}s")
    log(f"  factor_returns done in {time.time()-t0:.0f}s")
    return out


def main():
    log("=== Factor usage analytics ===")

    # Read v33 schema
    v33_schema = pl.read_parquet_schema(PANEL)
    v33_cols = list(v33_schema.keys())
    base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
                 "vol", "amount", "pct_chg", "adj_factor", "adj_close",
                 "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
                 "turnover_rate", "circ_cap", "cap", "symbol", "volume",
                 "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
    zigzag_labels = {"v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
                     "v10_1_b1_start", "v10_1_b1_interval",
                     "v10_1_down_start", "v10_1_down_interval",
                     "v10_1_peak_zone", "v10_1_valley_zone",
                     "v10_1_zig_peak", "v10_1_zig_valley"}
    factors = [c for c in v33_cols if c not in (base_cols | zigzag_labels)]
    log(f"v33 panel {len(v33_cols)} cols; voting factors={len(factors)}")

    log("Building _fwd_net on full panel...")
    t0 = time.time()
    panel_full = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"] + factors)
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
            pl.col("close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
        ])
        .with_columns(
            ((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_net"))
        )
        .select(["trade_date", "ts_code", "_fwd_net"] + factors)
        .collect()
    )
    log(f"_fwd_net done in {time.time()-t0:.0f}s")

    all_dates = trading_dates()
    start_dt = pd.Timestamp(START).date()
    end_dt = pd.Timestamp(END).date()
    rebal_dates = all_dates[0::REBAL_STEP]
    rebal_dates = [d for d in rebal_dates if start_dt <= pd.Timestamp(d).date() <= end_dt]

    factor_returns = compute_factor_returns_polars(panel_full, factors, all_dates, LOOKBACK_SESSIONS)
    fr_rebal = {d: factor_returns.get(d, {}) for d in rebal_dates}
    log(f"factor_returns coverage: {sum(1 for v in fr_rebal.values() if v)}/{len(fr_rebal)}")

    router = load_router()

    # === Per-rebal-date factor usage tracking ===
    factor_usage_count: dict[str, int] = {}
    factor_usage_count_bull: dict[str, int] = {}
    factor_usage_count_bear: dict[str, int] = {}
    factor_usage_score_sum: dict[str, float] = {}
    factor_dates: dict[str, list[str]] = {}
    daily_factors: dict[str, list[str]] = {}

    for rd in rebal_dates:
        fr = fr_rebal.get(rd, {})
        if not fr:
            daily_factors[rd] = []
            continue
        ranked = sorted(fr.items(), key=lambda kv: -kv[1])
        active = [f for f, _ in ranked[:TOP_K]]
        regime = router.get(rd, "bull_neutral")
        daily_factors[rd] = active
        for f, score in ranked[:TOP_K]:
            factor_usage_count[f] = factor_usage_count.get(f, 0) + 1
            factor_usage_score_sum[f] = factor_usage_score_sum.get(f, 0.0) + score
            factor_dates.setdefault(f, []).append(rd)
            if regime == "bull_neutral":
                factor_usage_count_bull[f] = factor_usage_count_bull.get(f, 0) + 1
            else:
                factor_usage_count_bear[f] = factor_usage_count_bear.get(f, 0) + 1

    # === Output 1: factor_usage_stats.json (overall + by regime) ===
    n_rebal = len(rebal_dates)
    n_bull = sum(1 for r in router.values() if r == "bull_neutral" and r in router)
    n_bear = sum(1 for r in router.values() if r == "bear" and r in router)
    # Use rebal-specific regime count
    bull_rebal = sum(1 for d in rebal_dates if router.get(d, "bull_neutral") == "bull_neutral")
    bear_rebal = sum(1 for d in rebal_dates if router.get(d, "bull_neutral") == "bear")

    factor_stats = {
        "metadata": {
            "panel": str(PANEL),
            "panel_version": "v33_with_new_factors_20261003",
            "panel_total_cols": len(v33_cols),
            "voting_factors": len(factors),
            "rebal_dates": n_rebal,
            "bull_neutral_rebal": bull_rebal,
            "bear_rebal": bear_rebal,
            "top_k_per_date": TOP_K,
            "max_slots_per_factor_date": TOP_K * n_rebal,
        },
        "factors": {
            f: {
                "n_used_total": factor_usage_count.get(f, 0),
                "n_used_bull": factor_usage_count_bull.get(f, 0),
                "n_used_bear": factor_usage_count_bear.get(f, 0),
                "pct_used_total": factor_usage_count.get(f, 0) / n_rebal * 100,
                "pct_used_bull": factor_usage_count_bull.get(f, 0) / max(bull_rebal, 1) * 100,
                "pct_used_bear": factor_usage_count_bear.get(f, 0) / max(bear_rebal, 1) * 100,
                "avg_score_when_used": factor_usage_score_sum.get(f, 0) / max(factor_usage_count.get(f, 0), 1),
            }
            for f in factors
        }
    }
    (OUT_DIR / "factor_usage_stats.json").write_text(json.dumps(factor_stats, indent=2, ensure_ascii=False))
    log(f"Wrote factor_usage_stats.json ({len(factors)} factors)")

    # === Output 2: factor_usage_top50.csv (sorted by usage) ===
    rows = []
    for f in factors:
        s = factor_stats["factors"][f]
        rows.append({
            "factor": f,
            "n_used_total": s["n_used_total"],
            "n_used_bull": s["n_used_bull"],
            "n_used_bear": s["n_used_bear"],
            "pct_used_total": round(s["pct_used_total"], 2),
            "pct_used_bull": round(s["pct_used_bull"], 2),
            "pct_used_bear": round(s["pct_used_bear"], 2),
            "avg_score": round(s["avg_score_when_used"], 5),
        })
    rows.sort(key=lambda x: -x["n_used_total"])
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "factor_usage_top50.csv", index=False)
    log(f"Wrote factor_usage_top50.csv (top 50 factors)")

    # === Output 3: factor_usage_by_regime.json (top 30 per regime) ===
    by_regime = {
        "all": sorted(rows, key=lambda x: -x["n_used_total"])[:30],
        "bull_neutral": sorted(rows, key=lambda x: -x["n_used_bull"])[:30],
        "bear": sorted(rows, key=lambda x: -x["n_used_bear"])[:30],
    }
    (OUT_DIR / "factor_usage_by_regime.json").write_text(
        json.dumps(by_regime, indent=2, ensure_ascii=False))
    log(f"Wrote factor_usage_by_regime.json (top 30 per regime)")

    # === Output 4: factor_usage_turnover.csv (per-factor date coverage) ===
    turnover_rows = []
    for f in factors:
        used_dates = factor_dates.get(f, [])
        if not used_dates:
            turnover_rows.append({"factor": f, "n_used": 0, "first_used": None,
                                  "last_used": None, "n_unique_months": 0,
                                  "consistency_score": 0.0})
            continue
        used_dates_sorted = sorted(used_dates)
        # months
        months = set(d[:7] for d in used_dates_sorted)
        turnover_rows.append({
            "factor": f,
            "n_used": len(used_dates_sorted),
            "first_used": used_dates_sorted[0],
            "last_used": used_dates_sorted[-1],
            "n_unique_months": len(months),
            "consistency_score": len(months) / max((pd.Timestamp(used_dates_sorted[-1]).year -
                                                     pd.Timestamp(used_dates_sorted[0]).year + 1) * 12, 1),
        })
    df2 = pd.DataFrame(turnover_rows).sort_values("n_used", ascending=False)
    df2.to_csv(OUT_DIR / "factor_usage_turnover.csv", index=False)
    log(f"Wrote factor_usage_turnover.csv (all factors)")

    # === Print summary to console ===
    log("\n=== TOP 20 FACTORS BY USAGE (all regime) ===")
    for i, r in enumerate(by_regime["all"][:20], 1):
        log(f"  {i:>2}. {r['factor']:<35} {r['n_used_total']:>4} ({r['pct_used_total']:>5.1f}%) | bull={r['n_used_bull']:>3} bear={r['n_used_bear']:>3} | avg_score={r['avg_score']:>+7.5f}")

    log("\n=== TOP 20 BULL FACTORS ===")
    for i, r in enumerate(by_regime["bull_neutral"][:20], 1):
        log(f"  {i:>2}. {r['factor']:<35} bull={r['n_used_bull']:>4} ({r['pct_used_bull']:>5.1f}%) | total={r['n_used_total']:>4} | avg_score={r['avg_score']:>+7.5f}")

    log("\n=== TOP 20 BEAR FACTORS ===")
    for i, r in enumerate(by_regime["bear"][:20], 1):
        log(f"  {i:>2}. {r['factor']:<35} bear={r['n_used_bear']:>4} ({r['pct_used_bear']:>5.1f}%) | total={r['n_used_total']:>4} | avg_score={r['avg_score']:>+7.5f}")

    log("\n=== USAGE DISTRIBUTION ===")
    used_50plus = sum(1 for r in rows if r["n_used_total"] >= 50)
    used_100plus = sum(1 for r in rows if r["n_used_total"] >= 100)
    used_never = sum(1 for r in rows if r["n_used_total"] == 0)
    log(f"  factors used 0 times:        {used_never} / {len(factors)}")
    log(f"  factors used >= 50 times:    {used_50plus} / {len(factors)}")
    log(f"  factors used >= 100 times:   {used_100plus} / {len(factors)}")
    log(f"  mean usage per factor:       {np.mean([r['n_used_total'] for r in rows]):.1f}")
    log(f"  median usage per factor:     {np.median([r['n_used_total'] for r in rows]):.0f}")
    log(f"  max usage:                   {max(r['n_used_total'] for r in rows)}")
    log(f"  min usage (non-zero):        {min(r['n_used_total'] for r in rows if r['n_used_total']>0)}")
    log(f"  total factor slots filled:   {sum(r['n_used_total'] for r in rows)} / {TOP_K * n_rebal} expected")

    log("\nDONE")


if __name__ == "__main__":
    main()