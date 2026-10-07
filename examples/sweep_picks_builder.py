"""Build picks using V27_BALANCE router + directional Bear factor scores.

Picks builder:
  bull/neutral regime: V14 IC voting picks (causal 60-day rolling Spearman
    IC, top-K = 10 factors, K_NOM = 10 stocks per factor, MIN_VOTES = 2)
  bear regime: directional bear factor voting
    - per-factor directional score already cached (High vs Low history),
      keeps the better direction with strict eligibility gate
    - current bear date picks: top 10 stocks from each active factor under
      its selected direction (high => nlargest; low => nsmallest)
    - MIN_VOTES = 3, MIN_STOCKS = 5, MAX_STOCKS = 10

Routing uses the V27_BALANCE parameters previously fixed:
  BEAR_VOTE_THRESHOLD = 2, BEAR_LOOKBACK = 5, BEAR_CUMULATIVE_THRESHOLD = 5

The execution contract (T+1 NextOpen, lot=100, 0.25% commission + 0.10%
slippage, 90% target) and IC artifact are unchanged from V37.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
SCORES = Path("evidence/v34_ic_voting_2010_2025_correct/ic_scores.csv")
IC_START = "2009-08-01"
START = "2010-01-01"
END = "2025-12-31"
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

OUT_ROOT = Path("evidence/v37_v14_causal_v27balance_directional_20261006")

BEAR_VOTE_THRESHOLD = 2
BEAR_LOOKBACK = 5
BEAR_CUMULATIVE_THRESHOLD = 5

ACADEMIC22 = [
    "l_size", "l_size3", "l_turnm", "l_turna", "l_ami", "l_dtvm", "l_dtva",
    "l_vdtv", "r_tv", "r_beta", "p_m1", "p_m3", "p_m6", "p_m11", "p_m24",
    "p_mchg", "p_52w", "p_mdr", "p_pr", "p_season", "v_bm", "v_ep",
]
NEW17 = [
    "winner_ratio", "efficiency_ratio", "fractal_dimension", "alpha191_040",
    "alpha191_095", "mom12m_jt", "maxret_bcw", "accruals_sloan", "idiovola_clmx",
    "gp_novymarx", "overnight_intraday_spread", "skew21_lottery", "pvcorr_21",
    "kurt21_returns", "coskew60", "hl_52w_disposition", "resmom_6m",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dt_expr(value: str) -> pl.Expr:
    return pl.lit(value).str.strptime(pl.Datetime("ms"), "%Y-%m-%d")


def router_map(daily: pd.DataFrame) -> dict[str, str]:
    d = daily.copy()
    d["ma5"] = d["idx_close"].rolling(5).mean()
    d["ma10"] = d["idx_close"].rolling(10).mean()
    d["ma20"] = d["idx_close"].rolling(20).mean()
    d["ma60"] = d["idx_close"].rolling(60).mean()
    d["ma120"] = d["idx_close"].rolling(120).mean()
    d["ret10"] = d["idx_close"].pct_change(10)
    d["ret20"] = d["idx_close"].pct_change(20)
    d["ret40"] = d["idx_close"].pct_change(40)
    d["ret60"] = d["idx_close"].pct_change(60)
    d["dist_ma20"] = (d["idx_close"] - d["ma20"]) / d["ma20"]
    d["dist_ma60"] = (d["idx_close"] - d["ma60"]) / d["ma60"]
    d["slope_ma60"] = d["ma60"].diff()
    d["high_60d"] = d["idx_close"].rolling(60).max()
    d["high_120d"] = d["idx_close"].rolling(120).max()
    d["dd_60d"] = d["idx_close"] / d["high_60d"] - 1.0
    d["dd_120d"] = d["idx_close"] / d["high_120d"] - 1.0
    sig = pd.concat([
        d["idx_close"] < d["ma5"],
        d["idx_close"] < d["ma10"],
        d["idx_close"] < d["ma20"],
        d["idx_close"] < d["ma60"],
        d["idx_close"] < d["ma120"],
        d["ma20"] < d["ma60"],
        d["ma60"] < d["ma120"],
        d["ret10"] < 0,
        d["ret20"] < 0,
        d["ret40"] < 0,
        d["ret60"] < 0,
        d["dist_ma20"] < -0.02,
        d["dist_ma60"] < -0.05,
        d["slope_ma60"] < 0,
        d["dd_60d"] < -0.05,
        d["dd_120d"] < -0.10,
    ], axis=1).fillna(False)
    votes = sig.sum(axis=1)
    cumulative = (votes >= BEAR_VOTE_THRESHOLD).astype(int).rolling(BEAR_LOOKBACK).sum()
    d["is_bear"] = cumulative >= BEAR_CUMULATIVE_THRESHOLD
    return {d_: "bear" if v else "bull_neutral" for d_, v in zip(d["date"], d["is_bear"].fillna(False))}


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


def build_bear_picks(
    panel: pl.DataFrame,
    factors: list[str],
    current_date: str,
    selected_path: Path,
    bear_idx_map: dict[str, dict[str, tuple[str, float]]],
) -> dict[str, int]:
    bear_map = bear_idx_map.get(current_date, {})
    if not bear_map:
        return {}
    selected = sorted(bear_map.items(), key=lambda kv: -kv[1][1])[:BEAR_TOP_FACTORS]
    active_factors = [f for f, _ in selected]
    day = panel.filter(pl.col("trade_date") == dt_expr(current_date))
    if day.height == 0:
        return {}
    codes = day.get_column("ts_code").to_list()
    values = day.select(active_factors).to_numpy()
    votes = np.zeros(day.height, dtype=np.int32)
    for j, factor in enumerate(active_factors):
        column = values[:, j]
        valid = np.flatnonzero(np.isfinite(column))
        if len(valid) < BEAR_TOP_STOCKS_PER_FACTOR * 2:
            continue
        direction, _ = bear_map[factor]
        if direction == "high":
            chosen = valid[np.argpartition(column[valid], -BEAR_TOP_STOCKS_PER_FACTOR)[-BEAR_TOP_STOCKS_PER_FACTOR:]]
        else:
            chosen = valid[np.argpartition(column[valid], BEAR_TOP_STOCKS_PER_FACTOR - 1)[:BEAR_TOP_STOCKS_PER_FACTOR]]
        votes[chosen] += 1
    order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
    selected_idx = [i for i in order if int(votes[i]) >= BEAR_MIN_VOTES]
    if len(selected_idx) < BEAR_MIN_STOCKS:
        selected_idx = order[:BEAR_MIN_STOCKS]
    selected_idx = selected_idx[:BEAR_MAX_STOCKS]
    return {str(codes[i]): int(votes[i]) for i in selected_idx}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebal-offset", type=int, required=True)
    parser.add_argument("--out-root", required=True, help="output picks root")
    parser.add_argument("--min-votes-bull", type=int, default=2)
    parser.add_argument("--min-votes-bear", type=int, default=3)
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument("--rebal-step", type=int, default=20)
    parser.add_argument("--ic-window", type=int, default=60)
    parser.add_argument("--bear-min-mean-return", type=float, default=0.005)
    parser.add_argument("--bear-lookback-sessions", type=int, default=30)
    args = parser.parse_args()
    offset = args.rebal_offset
    global MIN_VOTES, BEAR_MIN_VOTES, MAX_STOCKS, REBAL_STEP, IC_WINDOW, OUT_ROOT
    MIN_VOTES = args.min_votes_bull
    BEAR_MIN_VOTES = args.min_votes_bear
    MAX_STOCKS = args.max_stocks
    REBAL_STEP = args.rebal_step
    IC_WINDOW = args.ic_window
    OUT_ROOT = pathlib.Path(args.out_root)
    t0 = time.time()
    manifest = json.loads(Path("evidence/v37_v14_causal/factor_manifest.json").read_text())
    factors = manifest["factors"]
    log(f"offset={offset}; factors={len(factors)}")

    daily = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "idx_close"])
        .filter(pl.col("idx_close").is_not_null())
        .unique("trade_date")
        .sort("trade_date")
        .collect()
        .to_pandas()
    )
    daily["date"] = pd.to_datetime(daily["trade_date"]).dt.date.astype(str)
    all_dates = [d for d in daily["date"].tolist() if START <= d <= END]
    regime = router_map(daily[["date", "idx_close"]])
    bear_dates = [d for d in all_dates if regime.get(d, "bull_neutral") == "bear"]
    rebal_dates = all_dates[offset::REBAL_STEP]
    log(f"rebalances={len(rebal_dates)}, bear={len(bear_dates)}")

    # Reuse the baseline directional bear cache (built for baseline
    # MIN_FACTOR_MEAN_RETURN=0.005, LOOKBACK_SESSIONS=30, MIN_OBS=10;
    # these aren't being sweeped in this harness). Only --min-votes-bear
    # affects picks generation here.
    selected_path = pathlib.Path(
        "evidence/v37_v14_causal_v27balance_directional_20261006"
    ) / "_cache" / f"offset_{offset}" / "selected_directions.tsv"
    bear_idx_map: dict[str, dict[str, tuple[str, float]]] = {}
    if selected_path.exists():
        with selected_path.open() as fh:
            next(fh)
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 9:
                    continue
                cur, factor, direction, score, *_ = parts
                bear_idx_map.setdefault(cur, {})[factor] = (direction, float(score))
    else:
        log(f"selected directions cache missing: {selected_path}")

    score_map: dict[str, dict[str, float]] = {}
    raw_df = pd.read_csv(SCORES, usecols=["signal_date", "factor", "causal_ic_score"])
    mask = (
        raw_df["signal_date"].between(IC_START, END)
        & raw_df["causal_ic_score"].notna()
        & np.isfinite(raw_df["causal_ic_score"])
    )
    sub = raw_df.loc[mask, ["signal_date", "factor", "causal_ic_score"]]
    log(f"score rows={len(sub):,}")
    for d, fac, sc in zip(sub["signal_date"].to_numpy(), sub["factor"].to_numpy(), sub["causal_ic_score"].to_numpy(dtype=float)):
        score_map.setdefault(str(d)[:10], {})[str(fac)] = float(sc)

    rebal_panel_dates = [
        pd.Timestamp(d).to_pydatetime() for d in rebal_dates
    ]
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
        regime_tag = regime.get(rd, "bull_neutral")
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
            bear_picks = build_bear_picks(panel, factors, rd, selected_path, bear_idx_map)
            if not bear_picks:
                continue
            picks[rd] = bear_picks
            metadata[rd] = {
                "regime": "bear",
                "selector": "directional_bear_voting",
                "n_active_factors": sum(1 for f in bear_idx_map.get(rd, {}).values() if f[1] > 0),
                "n_picks": len(bear_picks),
                "max_votes": max(bear_picks.values()) if bear_picks else 0,
                "n_at_or_above_threshold": sum(1 for v in bear_picks.values() if v >= BEAR_MIN_VOTES),
            }
        if idx == 1 or idx % 25 == 0:
            log(
                f"  picks {idx}/{len(rebal_dates)} {rd} regime={regime_tag} "
                f"picks={metadata[rd]['n_picks']} max_votes={metadata[rd]['max_votes']}"
            )

    OUT_ROOT = pathlib.Path(args.out_root)
    out = OUT_ROOT / f"V14_{offset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (out / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))
    summary = {
        "tag": f"V14_{offset}",
        "panel": str(PANEL),
        "window": f"{START} ~ {END}",
        "factor_count_declared": len(factors),
        "all_factors_vote": True,
        "ic_window": IC_WINDOW,
        "causal_lag": CAUSAL_LAG,
        "fwd_contract": "T+1 raw open -> T+21 close",
        "rebalance_step": REBAL_STEP,
        "top_k": TOP_K,
        "k_nom": K_NOM,
        "min_votes": MIN_VOTES,
        "min_stocks": MIN_STOCKS,
        "max_stocks": MAX_STOCKS,
        "bear_router": "V27_BALANCE 5d cumulative V2/16 bear signals >= 5",
        "bear_selector": "directional bear factor voting (high vs low history)",
        "bear_cache": str(selected_path),
        "bear_top_factors": BEAR_TOP_FACTORS,
        "bear_min_votes": BEAR_MIN_VOTES,
        "bear_min_factor_mean_return": 0.005,
        "bear_min_observations": 10,
        "bear_lookback_sessions": 30,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "n_bull_neutral_rebalances": bull_count,
        "n_bear_rebalances": bear_count,
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "runtime_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"saved {out}: {len(picks)} dates bull={bull_count} bear={bear_count}")


if __name__ == "__main__":
    main()
