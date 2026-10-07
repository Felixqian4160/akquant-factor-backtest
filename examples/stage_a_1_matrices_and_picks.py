"""Stage A-1: scoring matrices (4 variants) + picks derivation (all variants).

Matrices (per-date top10-bot10 net20 diff, dates x 428 factors):
  M_v33raw : v33 factor values + RAW fwd_net
  M_v33adj : v33 factor values + ADJ fwd_net (adj prices joined from v34)
  M_v34raw : v34 factor values + RAW fwd_net
  M_v34adj : v34 factor values + ADJ fwd_net   (canonical)

Picks variants (all saved under evidence/stage_a_20261007/picks/<name>/V14_0/):
  repro_v34 (M_v34adj, top10) -- compare vs evidence/sweep/v34_factorrank_lb20_20261007
  repro_v33 (M_v33raw, top10) -- compare vs v33 picks
  p2_v34rawfwd (M_v34raw, top10)
  p3_v33adjfwd (M_v33adj, top10)
  null_01..null_10 (M_v34adj, 10 RANDOM factors per date)
  subsample (M_v34adj, score over even-indexed sessions only)
"""
from __future__ import annotations
import json
import pathlib
import sys
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V33 = AKQ / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = AKQ / "evidence" / "stage_a_20261007"
OUT.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
ENTRY_TO_EXIT_SHIFT = 21
LOOKBACK_SESSIONS = 20
TOP_K = 10
K_NOM = 10
MIN_VOTES_BULL = 2
MIN_STOCKS_BULL = 5
MAX_STOCKS_BULL = 10
MIN_VOTES_BEAR = 4
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10
ROUND_TRIP_COST = 0.005


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_factors(panel_path):
    cols = pl.read_parquet_schema(panel_path).keys()
    base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
                 "vol", "amount", "pct_chg", "adj_factor", "adj_close",
                 "adj_open", "adj_high", "adj_low",
                 "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
                 "turnover_rate", "circ_cap", "cap", "symbol", "volume",
                 "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
    zigzag = {"v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
              "v10_1_b1_start", "v10_1_b1_interval",
              "v10_1_down_start", "v10_1_down_interval",
              "v10_1_peak_zone", "v10_1_valley_zone",
              "v10_1_zig_peak", "v10_1_zig_valley"}
    excluded = base_cols | zigzag
    return [c for c in cols if c not in excluded]


def compute_matrix(panel_path, factors, out_path):
    """One pass over panel computing per-date diff for BOTH raw and adj fwd_net."""
    tag = out_path.stem
    p_raw = out_path.with_name(tag + "_RAW.parquet")
    p_adj = out_path.with_name(tag + "_ADJ.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if p_raw.exists() and p_adj.exists():
        log(f"matrices exist, skip: {p_raw.name}, {p_adj.name}")
        return p_raw, p_adj

    # need raw px + adj px (adj from v34 regardless of panel)
    t0 = time.time()
    adj_cols = pl.read_parquet(V34, columns=["ts_code", "trade_date", "adj_open", "adj_close"])
    adj_cols = adj_cols.with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
    pf = (
        pl.scan_parquet(panel_path)
        .select(["trade_date", "ts_code", "open", "close"] + factors)
        .with_columns(pl.col("trade_date").cast(pl.Datetime("ms")))
        .collect()
        .join(adj_cols, on=["ts_code", "trade_date"], how="left")
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_o1"),
            pl.col("close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_c21"),
            pl.col("adj_open").shift(-1).over("ts_code").alias("_ao1"),
            pl.col("adj_close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_ac21"),
        ])
        .with_columns([
            (pl.col("_c21") / pl.col("_o1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_raw"),
            (pl.col("_ac21") / pl.col("_ao1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_adj"),
        ])
        .select(["trade_date", "ts_code", "_fwd_raw", "_fwd_adj"] + factors)
    )
    log(f"  panel loaded {pf.shape} ({time.time()-t0:.0f}s)")

    base = pf.select("trade_date").unique().sort("trade_date")
    cols_raw, cols_adj = [], []
    t1 = time.time()
    for fi, f in enumerate(factors, 1):
        d = pf.select(["trade_date", "_fwd_raw", "_fwd_adj", f])
        d = d.filter(pl.col(f).is_finite())
        d = d.with_columns(pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk"))
        agg = d.group_by("trade_date").agg([
            (pl.col("_fwd_raw").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top_raw"),
            (pl.col("_fwd_raw").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot_raw"),
            (pl.col("_fwd_adj").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top_adj"),
            (pl.col("_fwd_adj").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot_adj"),
        ])
        agg = agg.select([
            pl.col("trade_date"),
            (pl.col("_top_raw") - pl.col("_bot_raw")).alias("_diff_raw"),
            (pl.col("_top_adj") - pl.col("_bot_adj")).alias("_diff_adj"),
        ])
        j = base.join(agg, on="trade_date", how="left")
        cols_raw.append(j["_diff_raw"].alias(f))
        cols_adj.append(j["_diff_adj"].alias(f))
        if fi % 100 == 0:
            log(f"    factor {fi}/{len(factors)} ({time.time()-t1:.0f}s)")

    m_raw = base.with_columns(cols_raw)
    m_adj = base.with_columns(cols_adj)

    tag = out_path.stem
    p_raw = out_path.with_name(tag + "_RAW.parquet")
    p_adj = out_path.with_name(tag + "_ADJ.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m_raw.write_parquet(p_raw)
    m_adj.write_parquet(p_adj)
    log(f"  saved {p_raw.name} / {p_adj.name} ({time.time()-t1:.0f}s)")
    del pf
    import gc; gc.collect()
    return p_raw, p_adj


# ─────────────────────── picks derivation ───────────────────────

def trading_dates():
    df = pl.scan_parquet(V34).select(["trade_date"]).unique().sort("trade_date").collect()
    return [str(x)[:10] for x in df["trade_date"].to_list()]


def derive_picks(matrix_path, panel_path, selector, out_dir, label, session_subsample=False):
    if not pathlib.Path(matrix_path).exists():
        log(f"matrix missing, skip: {label} ({pathlib.Path(matrix_path).name})")
        return None
    OUT_ROOT = pathlib.Path(out_dir) / label / "V14_0"
    if (OUT_ROOT / "picks.json").exists():
        log(f"picks exist, skip: {label}")
        return OUT_ROOT
    log(f"deriving picks: {label}")
    t0 = time.time()

    mat = pl.read_parquet(matrix_path)
    dates_iso = [str(x)[:10] for x in mat["trade_date"].to_list()]
    factors = [c for c in mat.columns if c != "trade_date"]

    all_dates = trading_dates()
    assert all_dates == dates_iso, "matrix dates != panel dates"
    n_dates = len(all_dates)

    import pandas as pd
    start_dt = pd.Timestamp(START).date()
    end_dt = pd.Timestamp(END).date()
    rebal_dates = [d for d in all_dates[0::REBAL_STEP] if start_dt <= pd.Timestamp(d).date() <= end_dt]

    # numeric matrix
    M = mat.select(factors).to_numpy()  # (n_dates, n_factors)
    date_pos = {d: i for i, d in enumerate(all_dates)}

    router = json.loads(ROUTER_MAP.read_text())

    # panel slice for voting (rebal dates only)
    import datetime as _dt
    rdt = pl.Series("trade_date", [_dt.datetime.fromisoformat(d) for d in rebal_dates]).dt.cast_time_unit("ms")
    day_panels = (
        pl.read_parquet(panel_path, columns=["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(rdt))
    )
    reb_datetime = list(day_panels["trade_date"].unique().sort().to_list())

    picks, meta = {}, {}
    for i_rd, rd in enumerate(rebal_dates):
        i = date_pos[rd]
        d = rd
        regime = router.get(d, "bull_neutral")
        min_votes = MIN_VOTES_BULL if regime == "bull_neutral" else MIN_VOTES_BEAR
        min_stocks = MIN_STOCKS_BULL if regime == "bull_neutral" else MIN_STOCKS_BEAR
        max_stocks = MAX_STOCKS_BULL if regime == "bull_neutral" else MAX_STOCKS_BEAR

        # scores: mean of last LOOKBACK sessions before i
        lo = max(0, i - LOOKBACK_SESSIONS)
        win = M[lo:i, :][::2, :] if session_subsample else M[lo:i, :]
        with np.errstate(all="ignore"):
            counts = np.isfinite(win).sum(axis=0)
            scores = np.where(counts >= 10, np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
        scored = [(fi, scores[fi]) for fi in range(len(factors)) if np.isfinite(scores[fi])]
        if not scored:
            continue
        ranked = sorted(scored, key=lambda kv: -kv[1])

        active_fi = selector(ranked, i_rd, d)
        active = [factors[fi] for fi in active_fi]

        # votes
        day = day_panels.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == rd)
        if day.height == 0:
            continue
        codes = day["ts_code"].to_list()
        values = day.select(active).to_numpy()
        votes = np.zeros(day.height, dtype=np.int32)
        for j in range(len(active)):
            column = values[:, j]
            valid = np.flatnonzero(np.isfinite(column))
            if len(valid) < K_NOM:
                continue
            chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
            votes[chosen] += 1

        order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
        selected = [k for k in order if int(votes[k]) >= min_votes]
        if len(selected) < min_stocks:
            selected = order[:min_stocks]
        selected = selected[:max_stocks]
        if not selected:
            continue
        picks[d] = {str(codes[k]): int(votes[k]) for k in selected}
        meta[d] = {"regime": regime, "selector": label, "n_active_factors": len(active),
                   "n_picks": len(selected), "max_votes": int(votes.max()),
                   "n_at_or_above_threshold": int((votes >= min_votes).sum())}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (OUT_ROOT / "picks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "label": label, "matrix": str(matrix_path), "panel": str(panel_path),
        "n_rebalances": len(picks), "window": f"{START}~{END}",
    }, indent=2))
    log(f"  saved {OUT_ROOT} ({len(picks)} dates, {time.time()-t0:.0f}s)")
    return OUT_ROOT


def sel_top10(ranked, i_rd, d):
    return [fi for fi, _ in ranked[:TOP_K]]


def make_sel_rand(seed):
    def sel(ranked, i_rd, d):
        rng = np.random.RandomState(seed * 100003 + i_rd)
        return list(rng.choice(len(ranked), size=TOP_K, replace=False))
    return sel


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", choices=["v33", "v34", "all"], default="all")
    ap.add_argument("--picks-only", action="store_true")
    args = ap.parse_args()

    factors_v33 = get_factors(V33)
    factors_v34 = get_factors(V34)
    log(f"factors: v33={len(factors_v33)}, v34={len(factors_v34)}")

    if not args.picks_only:
        if args.panel in ("v33", "all"):
            compute_matrix(V33, factors_v33, OUT / "matrix_v33.parquet")
        if args.panel in ("v34", "all"):
            compute_matrix(V34, factors_v34, OUT / "matrix_v34.parquet")

    m34adj = OUT / "matrix_v34_ADJ.parquet"
    m34raw = OUT / "matrix_v34_RAW.parquet"
    m33raw = OUT / "matrix_v33_RAW.parquet"
    m33adj = OUT / "matrix_v33_ADJ.parquet"

    # canonical v34 (repro) & v33 (repro)
    derive_picks(m34adj, V34, sel_top10, OUT / "picks", "repro_v34")
    derive_picks(m33raw, V33, sel_top10, OUT / "picks", "repro_v33")
    # attribution
    derive_picks(m34raw, V34, sel_top10, OUT / "picks", "p2_v34rawfwd")
    derive_picks(m33adj, V33, sel_top10, OUT / "picks", "p3_v33adjfwd")
    # null draws (random factor selection on canonical matrix)
    for s in range(1, 11):
        derive_picks(m34adj, V34, make_sel_rand(s), OUT / "picks", f"null_{s:02d}")
    # subsample selector: score over even-indexed sessions only
    derive_picks(m34adj, V34, sel_top10, OUT / "picks", "subsample", session_subsample=True)
    log("DONE stage A-1")


if __name__ == "__main__":
    main()
