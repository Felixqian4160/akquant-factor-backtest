"""Lookahead audit — score-window settle lag (v34-lb20 factor-return voting), 2026-10-10.

User question: 有没有未来参数 / 从几个角度验证这个收益的正确性。

Finding under test:
  At rebalance date i:  score[f, i] = mean(diff[f, i-20 : i])
  Each diff row t = top10-bot10 FORWARD return over [t+1 -> t+21]  (settles at t+21).
  => rows i-20..i-1 are UNSETTLED at i; the newest row settles at i+20, i.e. up to
     20 trading sessions AFTER the decision day.

This script:
  1. numerically re-verifies row semantics on matrix_v34_ADJ (manual recompute == matrix),
  2. reproduces the archived LEAKY picks (must be 100% identical -> single-variable proof),
  3. builds CAUSAL picks: same pipeline with the score window shifted back
     (lag=21 sessions, fully settled: window [i-41:i-21]) — single variable changed,
  4. quantifies factor-set / stock-pick overlap + score forward-predictiveness diagnostics.

Outputs:
  evidence/audit_lookahead_20261010/score_window_stats.json
  evidence/audit_lookahead_20261010/picks_causal/V14_{off}/{picks,picks_meta,summary}.json
"""
from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
STAGE5_PICKS = AKQ / "evidence" / "stage5_20261007" / "picks"
OUT = AKQ / "evidence" / "audit_lookahead_20261010"

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
LOOKBACK = 20
SETTLE_LAG = 21  # conservative: score window ends 21 sessions before i (all rows settled)
TOP_K = 10
K_NOM = 10
MIN_VOTES_BULL = 2
MIN_STOCKS_BULL = 5
MAX_STOCKS_BULL = 10
MIN_VOTES_BEAR = 4
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ────────────────────────── data ──────────────────────────

def trading_dates() -> list[str]:
    df = pl.scan_parquet(V34).select(["trade_date"]).unique().sort("trade_date").collect()
    return [str(x)[:10] for x in df["trade_date"].to_list()]


def rebal_dates(all_dates: list[str], offset: int) -> list[str]:
    import pandas as pd
    start_dt = pd.Timestamp(START).date()
    end_dt = pd.Timestamp(END).date()
    return [d for d in all_dates[offset::REBAL_STEP]
            if start_dt <= pd.Timestamp(d).date() <= end_dt]


# ────────────────────────── 1. semantics check ──────────────────────────

def semantics_check(mat: pl.DataFrame, all_dates: list[str]) -> dict:
    """Manually recompute matrix cell M[t, f] from the raw panel for 3 factors x 2 dates."""
    factors_chk = ["alpha_alpha001", "gtja_gtja_002", "talib_RSI"]
    probe_dates = [all_dates[300], all_dates[1500]]
    pf = (
        pl.scan_parquet(V34)
        .select(["trade_date", "ts_code", "adj_open", "adj_close"] + factors_chk)
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("adj_open").shift(-1).over("ts_code").alias("_o1"),
            pl.col("adj_close").shift(-21).over("ts_code").alias("_c21"),
        ])
        .with_columns((pl.col("_c21") / pl.col("_o1") - 1.0 - 0.005).alias("_fwd"))
        .collect()
    )
    out = []
    for d in probe_dates:
        sub = pf.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == d)
        for f in factors_chk:
            dd = sub.filter(pl.col(f).is_finite() & pl.col("_fwd").is_finite())
            dd = dd.with_columns(pl.col(f).rank(method="ordinal", descending=True).alias("_rk"))
            top = dd.filter(pl.col("_rk") <= 10)["_fwd"].sum() / 10.0
            bot = dd.filter(pl.col("_rk") > (dd["_rk"].max() - 10))["_fwd"].sum() / 10.0
            manual = float(top - bot)
            mv = mat.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == d)[f].to_list()
            if mv:
                out.append({"date": d, "factor": f, "manual_recompute": manual,
                            "matrix": float(mv[0]), "abs_diff": abs(manual - float(mv[0]))})
    return {"cells": out,
            "max_abs_diff": max(c["abs_diff"] for c in out),
            "note": "manual recompute of a forward window [t+1 -> t+21]; row t value settles at t+21"}


# ────────────────────────── 2/3. picks derivation ──────────────────────────

def derive(all_dates, date_pos, M, factors, day_panels, router, offset, mode):
    """mode='leaky' -> window [i-20:i] (archived);  mode='causal' -> window [i-41:i-21]."""
    picks, meta = {}, {}
    for rd in rebal_dates(all_dates, offset):
        i = date_pos[rd]
        if mode == "leaky":
            lo, hi = max(0, i - LOOKBACK), i
        else:  # causal: shift window back SETTLE_LAG sessions
            hi = i - SETTLE_LAG
            lo = max(0, hi - LOOKBACK)
        if hi - lo < 10:
            continue
        win = M[lo:hi, :]
        with np.errstate(all="ignore"):
            counts = np.isfinite(win).sum(axis=0)
            scores = np.where(counts >= 10, np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
        scored = [(fi, scores[fi]) for fi in range(len(factors)) if np.isfinite(scores[fi])]
        if not scored:
            continue
        ranked = sorted(scored, key=lambda kv: -kv[1])
        active = [factors[fi] for fi, _ in ranked[:TOP_K]]

        regime = router.get(rd, "bull_neutral")
        if regime == "bull_neutral":
            mv_, ms, mx = MIN_VOTES_BULL, MIN_STOCKS_BULL, MAX_STOCKS_BULL
        else:
            mv_, ms, mx = MIN_VOTES_BEAR, MIN_STOCKS_BEAR, MAX_STOCKS_BEAR

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
        selected = [k for k in order if int(votes[k]) >= mv_]
        if len(selected) < ms:
            selected = order[:ms]
        selected = selected[:mx]
        if not selected:
            continue
        picks[rd] = {str(codes[k]): int(votes[k]) for k in selected}
        meta[rd] = {"regime": regime, "selector": f"factor_return_voting_{mode}",
                    "offset": offset, "n_active_factors": len(active),
                    "n_picks": len(selected), "max_votes": int(votes.max()),
                    "active_factors": active}
    return picks, meta


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {}

    log("loading matrix...")
    mat = pl.read_parquet(MAT)
    dates_iso = [str(x)[:10] for x in mat["trade_date"].to_list()]
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy()
    all_dates = trading_dates()
    assert all_dates == dates_iso
    date_pos = {d: i for i, d in enumerate(all_dates)}
    log(f"matrix: {M.shape}, factors={len(factors)}")

    # 1) semantics
    log("== semantics check ==")
    stats["semantics"] = semantics_check(mat, all_dates)
    log(f"  max|manual-matrix| = {stats['semantics']['max_abs_diff']}")

    router = json.loads(ROUTER_MAP.read_text())

    # 2) leaky repro (offset 0) — must equal archived stage5 picks exactly
    log("== leaky repro check (offset 0) ==")
    _rd_all = sorted(set(rebal_dates(all_dates, 0)) | set(rebal_dates(all_dates, 10)))
    rdt = pl.Series("trade_date", [
        __import__("datetime").datetime.fromisoformat(d) for d in _rd_all
    ]).dt.cast_time_unit("ms")
    day_panels = (
        pl.read_parquet(V34, columns=["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(rdt))
    )
    picks_leaky, meta_leaky = derive(all_dates, date_pos, M, factors, day_panels, router, 0, "leaky")
    arch = json.loads((STAGE5_PICKS / "V14_0" / "picks.json").read_text())
    common = set(picks_leaky) & set(arch)
    unequal = [d for d in common if set(picks_leaky[d]) != set(arch[d])]
    stats["leaky_repro"] = {"dates_compared": len(common),
                            "dates_equal": len(common) - len(unequal),
                            "mismatch_dates": unequal[:10]}
    log(f"  repro: {len(common)-len(unequal)}/{len(common)} dates identical (mismatches={len(unequal)})")

    # 4a) factor-set & pick overlap leaky vs causal (offset 0), + score diagnostics
    log("== leaky vs causal overlap + diagnostics ==")
    picks_causal0, meta_causal0 = derive(all_dates, date_pos, M, factors, day_panels, router, 0, "causal")
    overlaps_f, overlaps_s = [], []
    for d in sorted(set(picks_leaky) & set(picks_causal0)):
        overlaps_f.append(jaccard(meta_leaky[d]["active_factors"], meta_causal0[d]["active_factors"]))
        overlaps_s.append(jaccard(list(picks_leaky[d]), list(picks_causal0[d])))
    stats["overlap_offset0"] = {
        "n_dates": len(overlaps_f),
        "mean_factor_jaccard": float(np.mean(overlaps_f)) if overlaps_f else None,
        "mean_stock_jaccard": float(np.mean(overlaps_s)) if overlaps_s else None,
    }
    log(f"  mean factor-set jaccard={stats['overlap_offset0']['mean_factor_jaccard']:.3f}, "
        f"stock-pick jaccard={stats['overlap_offset0']['mean_stock_jaccard']:.3f}")

    # diagnostics: corr(score, M[i]) (next 20d) and corr(score, M[i-42]) (settled past)
    diag = []
    sample_idx = [i for i in [400, 900, 1400, 1900, 2400, 2900, 3400] if i < len(all_dates)]
    for i in sample_idx:
        if i - SETTLE_LAG - LOOKBACK < 0:
            continue
        sl = np.nanmean(np.where(np.isfinite(M[i - LOOKBACK:i]), M[i - LOOKBACK:i], np.nan), axis=0)
        sc = np.nanmean(np.where(np.isfinite(M[i - 41:i - 21]), M[i - 41:i - 21], np.nan), axis=0)
        d_next = M[i]
        d_past = M[i - 42]
        ok = np.isfinite(sl) & np.isfinite(sc) & np.isfinite(d_next) & np.isfinite(d_past)
        if ok.sum() < 50:
            continue
        diag.append({
            "date": all_dates[i], "n_factors": int(ok.sum()),
            "corr_leaky_vs_next20d": float(np.corrcoef(sl[ok], d_next[ok])[0, 1]),
            "corr_causal_vs_next20d": float(np.corrcoef(sc[ok], d_next[ok])[0, 1]),
            "corr_leaky_vs_settledpast": float(np.corrcoef(sl[ok], d_past[ok])[0, 1]),
        })
    stats["score_diagnostics"] = diag
    for r in diag:
        log(f"  {r['date']}: leaky~next20d={r['corr_leaky_vs_next20d']:.3f}  "
            f"causal~next20d={r['corr_causal_vs_next20d']:.3f}  leaky~settled={r['corr_leaky_vs_settledpast']:.3f}")

    # 3) write causal picks for offsets 0 & 10
    for off in (0, 10):
        log(f"== building causal picks offset={off} ==")
        if off == 0:
            pk, mt = picks_causal0, meta_causal0
        else:
            pk, mt = derive(all_dates, date_pos, M, factors, day_panels, router, off, "causal")
        outdir = OUT / "picks_causal" / f"V14_{off}"
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "picks.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1))
        (outdir / "picks_meta.json").write_text(json.dumps(mt, ensure_ascii=False, indent=1))
        (outdir / "summary.json").write_text(json.dumps({
            "tag": f"V14_{off}", "mode": "causal_settled_lag21",
            "window": f"score = mean(diff[i-41:i-21]) — fully settled at decision i",
            "n_rebalances": len(pk),
            "avg_n_stocks": float(np.mean([len(v) for v in pk.values()])) if pk else 0.0,
        }, indent=2))
        log(f"  saved {outdir} ({len(pk)} dates)")

    stats["causal_offset0_dates"] = len(picks_causal0)
    stats["leaky_offset0_dates"] = len(picks_leaky)
    (OUT / "score_window_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    log(f"stats saved: {OUT/'score_window_stats.json'}")
    log("DONE audit script")


if __name__ == "__main__":
    main()
