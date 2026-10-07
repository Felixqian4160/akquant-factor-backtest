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
SCORES = Path("evidence/v34_ic_voting_2010_2025_correct/ic_scores.csv")
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


def build_bear_factor_returns(bear_session_dates: list[str], factors: list[str], offset: int) -> dict:
    """Per current bear date, build a dict {factor: (direction, selected_score)}.

    Reads the directional cache built by
    examples/build_bear_directional_cache_v27balance.py, which scored
    each (date, factor) by both High and Low historical net-return
    portfolios and selected the better direction with eligibility gates.
    """
    cache_path = DIRECTIONAL_CACHE_BASE / f"offset_{offset}" / "selected_directions.tsv"
    if not cache_path.exists():
        log(f"  directional cache missing: {cache_path}")
        return {}
    t0 = time.time()
    out: dict[str, dict[str, tuple[str, float]]] = {}
    with cache_path.open() as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            cur, fac, direction, score = parts[0], parts[1], parts[2], float(parts[3])
            if cur not in bear_session_dates:
                continue
            if fac not in factors:
                continue
            out.setdefault(cur, {})[fac] = (direction, score)
    log(f"  directional cache loaded for {len(out)} bear dates in {time.time()-t0:.1f}s")
    return out


def main() -> None:
    global MIN_VOTES, BEAR_MIN_VOTES, OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebal-offset", type=int, required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--min-votes-bull", type=int, default=MIN_VOTES)
    parser.add_argument("--min-votes-bear", type=int, default=BEAR_MIN_VOTES)
    args = parser.parse_args()
    offset = args.rebal_offset
    MIN_VOTES = args.min_votes_bull
    BEAR_MIN_VOTES = args.min_votes_bear
    OUT_ROOT = pathlib.Path(args.out_root)

    t0 = time.time()
    manifest = json.loads(Path("evidence/v37_v14_causal/factor_manifest.json").read_text())
    factors = manifest["factors"]
    log(f"offset={offset}; factors={len(factors)}")

    # Load causal ZigZag regime
    router_map = json.loads(ROUTER_MAP_PATH.read_text())
    all_dates = trading_dates()
    bear_dates = sorted([d for d in all_dates if router_map.get(d) == "bear"])
    rebal_dates = all_dates[offset::REBAL_STEP]
    log(f"rebalances={len(rebal_dates)}, bear dates in window={len(bear_dates)}")

    # Bear factor score cache (directional High/Low per (date, factor))
    bear_idx_map = build_bear_factor_returns(bear_dates, factors, offset)

    # IC scores for bull/neutral regime
    score_map: dict[str, dict[str, float]] = {}
    raw_df = pd.read_csv(SCORES, usecols=["signal_date", "factor", "causal_ic_score"])
    mask = (
        raw_df["signal_date"].between(IC_START, END)
        & raw_df["causal_ic_score"].notna()
        & np.isfinite(raw_df["causal_ic_score"])
    )
    sub = raw_df.loc[mask, ["signal_date", "factor", "causal_ic_score"]]
    log(f"score rows={len(sub):,}")
    for d, fac, sc in zip(sub["signal_date"].to_numpy(),
                          sub["factor"].to_numpy(),
                          sub["causal_ic_score"].to_numpy(dtype=float)):
        score_map.setdefault(str(d)[:10], {})[str(fac)] = float(sc)

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
            bear_picks_active = bear_idx_map.get(rd, {})
            if not bear_picks_active:
                continue
            eligible = [(f, direction, score)
                        for f, (direction, score) in bear_picks_active.items()]
            eligible.sort(key=lambda x: -x[2])
            active = [f for f, _, _ in eligible[:BEAR_TOP_FACTORS]]
            day = panel.filter(pl.col("trade_date") == dt_expr(rd))
            if day.height == 0:
                continue
            codes = day.get_column("ts_code").to_list()
            values = day.select(active).to_numpy()
            votes = np.zeros(day.height, dtype=np.int32)
            direction_map = {f: d for f, d, _ in eligible[:BEAR_TOP_FACTORS]}
            score_map_bear = {f: s for f, _, s in eligible[:BEAR_TOP_FACTORS]}
            for j, factor in enumerate(active):
                column = values[:, j]
                valid = np.flatnonzero(np.isfinite(column))
                if len(valid) < BEAR_TOP_STOCKS_PER_FACTOR * 2:
                    continue
                direction = direction_map[factor]
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
            if not selected_idx:
                continue
            picks[rd] = {str(codes[i]): int(votes[i]) for i in selected_idx}
            metadata[rd] = {
                "regime": "bear",
                "selector": "v19_factor_return_voting",
                "n_eligible_factors": len(bear_picks_active),
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
        "window": f"{START} ~ {END}",
        "router": "causal ZigZag leg.start (≥100d legs, lookback-zero)",
        "bull_selector": "V14 IC voting (60d Spearman, lag=21)",
        "bear_selector": "v19_factor_return_voting (historical mean return top-10 stocks)",
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
