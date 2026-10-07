"""Build picks using causal ZigZag router + V14 IC + V19 directional bear.

Identical scoring logic to build_picks_v27balance_directional.py, but
regime is loaded from a precomputed JSON router map built by
causal_zigzag_router.py (uses ZigZag leg START dates, no look-ahead).

The bear factor score cache is the directional V27_BALANCE cache at
evidence/v37_v14_causal_v27balance_directional_20261006/_cache/offset_{N}/
selected_directions.tsv, which carries High/Low direction per factor
+ selected_score. The picks builder reads selected_score and selects
nlarge vs nsmallest based on direction.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
SCORES = Path("evidence/v34_ic_voting_2010_2025_correct/ic_scores.csv")  # not used — IC computed inline
DIRECTIONAL_CACHE_BASE = Path(
    "evidence/v37_v14_causal_v27balance_directional_20261006/_cache"
)
ROUTER_MAP_PATH = Path("evidence/causal_zigzag_router_20261006/router_map.json")

START = "2010-01-01"
END = "2025-12-31"
IC_START = "2009-08-01"
REBAL_STEP = 20
CAUSAL_LAG = 21
IC_WINDOW = 60
MIN_IC_OBS = 30
K_NOM = 10
TOP_K = 10
MIN_VOTES = 2
MIN_STOCKS = 5
MAX_STOCKS = 10
ROUND_TRIP_COST = 0.005
BEAR_TOP_FACTORS = 10
BEAR_TOP_STOCKS_PER_FACTOR = 10
BEAR_MIN_VOTES = 3
BEAR_MIN_STOCKS = 5
BEAR_MAX_STOCKS = 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dt_expr(value: str) -> pl.Expr:
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def trading_dates() -> list[str]:
    dates = (
        pl.scan_parquet(PANEL)
        .select("trade_date")
        .filter((pl.col("trade_date") >= dt_expr(START)) & (pl.col("trade_date") <= dt_expr(END)))
        .unique()
        .sort("trade_date")
        .collect()
        .get_column("trade_date")
        .to_list()
    )
    return [pd.Timestamp(d).date().isoformat() for d in dates]



def main() -> None:
    global MIN_VOTES, BEAR_MIN_VOTES, OUT_ROOT  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebal-offset", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--min-votes-bull", type=int, default=2)
    parser.add_argument("--min-votes-bear", type=int, default=4)
    args = parser.parse_args()
    offset = args.rebal_offset
    MIN_VOTES = args.min_votes_bull
    BEAR_MIN_VOTES = args.min_votes_bear
    OUT_ROOT = pathlib.Path(args.out_root)

    t0 = time.time()
    # Read v33 panel schema directly — single source of truth, no separate manifest.
    v33_schema = pl.read_parquet_schema(PANEL)
    v33_cols = list(v33_schema.keys())
    # Exclude base OHLCV / metadata columns and zigzag labels (look-ahead bias)
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
    excluded = base_cols | zigzag_labels
    factors = [c for c in v33_cols if c not in excluded]
    log(f"offset={offset}; v33 panel {len(v33_cols)} cols, voting factors={len(factors)} "
        f"(excluded {len(excluded)} base/zigzag)")

    # Load causal ZigZag regime
    router_map = json.loads(ROUTER_MAP_PATH.read_text())
    all_dates = trading_dates()
    bear_dates = sorted([d for d in all_dates if router_map.get(d) == "bear"])
    rebal_dates = all_dates[offset::REBAL_STEP]
    log(f"rebalances={len(rebal_dates)}, bear dates in window={len(bear_dates)}")

    # No directional bear cache needed — bear regime uses simple
    # IC voting with stricter MIN_VOTES.

    # Causal Spearman IC scores (60-day rolling window, 21-day causal lag,
    # 20-day forward net return). Computed inline so each offset's
    # rebal dates (different from baseline) get correct scores.
    log("Computing causal Spearman IC scores inline...")
    t_ic = time.time()
    base = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"] + factors)
        .filter((pl.col("trade_date") >= pl.datetime(2009, 8, 1)) &
                (pl.col("trade_date") <= pl.datetime(2025, 12, 31)))
        .sort(["ts_code", "trade_date"])
        .collect()
    )
    base = base.with_columns([
        pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
        pl.col("close").shift(-21).over("ts_code").alias("_close_t21"),
    ]).with_columns(
        (pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("net20")
    )
    # Compute per-day Spearman corr of each factor vs net20 across stocks
    ic_wide = (
        base.group_by("trade_date")
        .agg([pl.corr(pl.col(c), pl.col("net20"), method="spearman").alias(c) for c in factors])
        .sort("trade_date")
    )
    raw_dates_np = np.asarray(
        [np.datetime64(pd.Timestamp(d).date(), "D") for d in ic_wide["trade_date"].to_list()]
    )
    raw_values = ic_wide.select(factors).to_numpy()
    score_map: dict[str, dict[str, float]] = {}
    rebal_set = set(rebal_dates)
    for rd in rebal_dates:
        rd_np = np.datetime64(rd, "D")
        count_le = int(np.searchsorted(raw_dates_np, rd_np, side="right"))
        end_exclusive = count_le - CAUSAL_LAG
        if end_exclusive < MIN_IC_OBS:
            continue
        begin = max(0, end_exclusive - IC_WINDOW)
        window = raw_values[begin:end_exclusive]
        with np.errstate(invalid="ignore"):
            scores = np.nanmean(window, axis=0)
        d_entry: dict[str, float] = {}
        for fac, sc in zip(factors, scores):
            if np.isfinite(sc):
                d_entry[fac] = float(sc)
        score_map[rd] = d_entry
    log(f"IC scores computed for {len(score_map)} rebal dates in {time.time()-t_ic:.1f}s")

    rebal_panel_dates = [pd.Timestamp(d).to_pydatetime() for d in rebal_dates]
    date_series = pl.Series("_dates", rebal_panel_dates, dtype=pl.Datetime("ms"))
    panel = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(date_series))
        .collect()
    )
    log(f"panel subset rows={panel.height:,}")

    picks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict] = {}
    bull_count = bear_count = 0
    for idx, rd in enumerate(rebal_dates, start=1):
        regime_tag = router_map.get(rd, "bull_neutral")
        if regime_tag == "bull_neutral":
            bull_count += 1
            factor_scores = score_map.get(rd, {})
            if not factor_scores:
                continue
            ranked = sorted(factor_scores.items(), key=lambda kv: -abs(kv[1]))
            active = [f for f, _ in ranked[:TOP_K]]
            day = panel.filter(pl.col("trade_date") == dt_expr(rd))
            if day.height == 0:
                continue
            codes = day.get_column("ts_code").to_list()
            values = day.select(active).to_numpy()
            votes = np.zeros(day.height, dtype=np.int32)
            for j, factor in enumerate(active):
                column = values[:, j]
                valid = np.flatnonzero(np.isfinite(column))
                if len(valid) < K_NOM:
                    continue
                if factor_scores[factor] >= 0:
                    chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
                else:
                    chosen = valid[np.argpartition(column[valid], K_NOM - 1)[:K_NOM]]
                votes[chosen] += 1
            order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
            selected_idx = [i for i in order if int(votes[i]) >= MIN_VOTES]
            if len(selected_idx) < MIN_STOCKS:
                selected_idx = order[:MIN_STOCKS]
            selected_idx = selected_idx[:MAX_STOCKS]
            if not selected_idx:
                continue
            picks[rd] = {str(codes[i]): int(votes[i]) for i in selected_idx}
            metadata[rd] = {
                "regime": "bull_neutral",
                "selector": "v14_ic_voting",
                "n_active_factors": len(active),
                "n_picks": len(selected_idx),
                "max_votes": int(votes.max()),
                "n_at_or_above_threshold": int((votes >= MIN_VOTES).sum()),
            }
        else:
            bear_count += 1
            # Bear regime: simple IC voting (same as bull, just stricter min_votes)
            factor_scores = score_map.get(rd, {})
            if not factor_scores:
                continue
            ranked = sorted(factor_scores.items(), key=lambda kv: -abs(kv[1]))
            active = [f for f, _ in ranked[:TOP_K]]
            day = panel.filter(pl.col("trade_date") == dt_expr(rd))
            if day.height == 0:
                continue
            codes = day.get_column("ts_code").to_list()
            values = day.select(active).to_numpy()
            votes = np.zeros(day.height, dtype=np.int32)
            for j, factor in enumerate(active):
                column = values[:, j]
                valid = np.flatnonzero(np.isfinite(column))
                if len(valid) < K_NOM:
                    continue
                if factor_scores[factor] >= 0:
                    chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
                else:
                    chosen = valid[np.argpartition(column[valid], K_NOM - 1)[:K_NOM]]
                votes[chosen] += 1
            order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
            selected_idx = [i for i in order if int(votes[i]) >= BEAR_MIN_VOTES]
            if len(selected_idx) < BEAR_MIN_STOCKS:
                selected_idx = order[:BEAR_MIN_STOCKS]
            selected_idx = selected_idx[:BEAR_MAX_STOCKS]
            if not selected_idx:
                continue
            picks[rd] = {str(codes[i]): int(votes[i]) for i in selected_idx}
            metadata[rd] = {
                "regime": "bear",
                "selector": "v14_ic_voting",
                "n_active_factors": len(active),
                "n_picks": len(selected_idx),
                "max_votes": int(votes.max()),
                "n_at_or_above_threshold": int((votes >= BEAR_MIN_VOTES).sum()),
            }
        if idx == 1 or idx % 25 == 0:
            sel_n = metadata[rd]["n_picks"]
            mv = metadata[rd]["max_votes"]
            log(f"  picks {idx}/{len(rebal_dates)}, date={rd}, regime={regime_tag}, "
                f"selected={sel_n}, max_votes={mv}")

    out = OUT_ROOT / f"V14_{offset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (out / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))
    summary = {
        "tag": f"V14_{offset}",
        "panel": str(PANEL),
        "panel_version": "v33_with_new_factors_20261003",
        "panel_total_cols": len(v33_cols),
        "window": f"{START} ~ {END}",
        "router": "causal ZigZag leg.start (≥100d legs, lookback-zero)",
        "bull_selector": "V14 IC voting (60d Spearman, lag=21, MIN_VOTES=2, K_NOM=10)",
        "bear_selector": "V14 IC voting (60d Spearman, lag=21, MIN_VOTES=4, K_NOM=10)",
        "factor_count_voting": len(factors),
        "excluded_zigzag_labels": sorted(zigzag_labels),
        "excluded_base_cols": sorted(base_cols),
        "rebalance_step": REBAL_STEP,
        "top_k": TOP_K,
        "k_nom": K_NOM,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
        "bear_top_factors": BEAR_TOP_FACTORS,
        "bear_min_votes": BEAR_MIN_VOTES,
        "bear_min_factor_mean_return": 0.005,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "n_bull_neutral_rebalances": bull_count,
        "n_bear_rebalances": bear_count,
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "runtime_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"saved {out}: {len(picks)} dates bull_neutral={bull_count} bear={bear_count}")


if __name__ == "__main__":
    main()
