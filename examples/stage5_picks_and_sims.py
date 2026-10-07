"""Stage 5: 10 offsets × 2026 OOS × v34-lb20 canonical baseline.

Outputs:
  evidence/stage5_20261007/picks/V14_{off}/{picks,meta,summary}.json  (10 sets)
  evidence/stage5_20261007/sims/V14_{off}/...                            (10 sims)
  evidence/stage5_20261007/sims_2026/V14_{off}/...                       (10 OOS sims, end=2026-08-27)

Picks derivation reuses matrix_v34_ADJ + derive_picks logic from stage_a_1.
Runner = examples/v41_run_akquant_v34.py.
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"

START = "2010-01-01"
END = "2025-12-31"
OOS_END = "2026-08-27"
REBAL_STEP = 20
LOOKBACK = 20
TOP_K = 10
K_NOM = 10
MIN_VOTES_BULL = 2
MIN_STOCKS_BULL = 5
MAX_STOCKS_BULL = 10
MIN_VOTES_BEAR = 4
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10

OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
OUT_BASE = AKQ / "evidence" / "stage5_20261007"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def trading_dates_from_panel():
    df = pl.scan_parquet(V34).select(["trade_date"]).unique().sort("trade_date").collect()
    return [str(x)[:10] for x in df["trade_date"].to_list()]


def derive_picks_offset(matrix_path, panel_path, offset, out_root):
    OUT_ROOT = out_root / f"V14_{offset}"
    if (OUT_ROOT / "picks.json").exists():
        log(f"picks exist, skip: V14_{offset}")
        return OUT_ROOT
    log(f"deriving picks: V14_{offset} (offset={offset})")
    t0 = time.time()

    mat = pl.read_parquet(matrix_path)
    dates_iso = [str(x)[:10] for x in mat["trade_date"].to_list()]
    _drop = [c for c in mat.columns if c.startswith("v10_1_")]
    if _drop: mat = mat.drop(_drop)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy()

    all_dates = trading_dates_from_panel()
    assert all_dates == dates_iso
    import pandas as pd
    start_dt = pd.Timestamp(START).date()
    end_dt = pd.Timestamp(END).date()
    rebal_dates = [d for d in all_dates[offset::REBAL_STEP] if start_dt <= pd.Timestamp(d).date() <= end_dt]

    date_pos = {d: i for i, d in enumerate(all_dates)}
    router = json.loads(ROUTER_MAP.read_text())

    import datetime as _dt
    rdt = pl.Series("trade_date", [_dt.datetime.fromisoformat(d) for d in rebal_dates]).dt.cast_time_unit("ms")
    day_panels = (
        pl.read_parquet(panel_path, columns=["trade_date", "ts_code"] + factors)
        .filter(pl.col("trade_date").is_in(rdt))
    )

    picks, meta = {}, {}
    for rd in rebal_dates:
        i = date_pos[rd]
        regime = router.get(rd, "bull_neutral")
        if regime == "bull_neutral":
            mv, ms, mx = MIN_VOTES_BULL, MIN_STOCKS_BULL, MAX_STOCKS_BULL
        else:
            mv, ms, mx = MIN_VOTES_BEAR, MIN_STOCKS_BEAR, MAX_STOCKS_BEAR

        lo = max(0, i - LOOKBACK)
        win = M[lo:i, :]
        with np.errstate(all="ignore"):
            counts = np.isfinite(win).sum(axis=0)
            scores = np.where(counts >= 10, np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
        scored = [(fi, scores[fi]) for fi in range(len(factors)) if np.isfinite(scores[fi])]
        if not scored: continue
        ranked = sorted(scored, key=lambda kv: -kv[1])
        active_fi = [fi for fi, _ in ranked[:TOP_K]]
        active = [factors[fi] for fi in active_fi]

        day = day_panels.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == rd)
        if day.height == 0: continue
        codes = day["ts_code"].to_list()
        values = day.select(active).to_numpy()
        votes = np.zeros(day.height, dtype=np.int32)
        for j in range(len(active)):
            column = values[:, j]
            valid = np.flatnonzero(np.isfinite(column))
            if len(valid) < K_NOM: continue
            chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
            votes[chosen] += 1
        order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
        selected = [k for k in order if int(votes[k]) >= mv]
        if len(selected) < ms: selected = order[:ms]
        selected = selected[:mx]
        if not selected: continue
        picks[rd] = {str(codes[k]): int(votes[k]) for k in selected}
        meta[rd] = {"regime": regime, "selector": "factor_return_voting_offset",
                    "offset": offset, "n_active_factors": len(active),
                    "n_picks": len(selected), "max_votes": int(votes.max()),
                    "n_at_or_above_threshold": int((votes >= mv).sum())}

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (OUT_ROOT / "picks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (OUT_ROOT / "summary.json").write_text(json.dumps({
        "tag": f"V14_{offset}", "offset": offset, "lookback_sessions": LOOKBACK,
        "panel": str(panel_path), "panel_version": "v34_adj_20261007",
        "n_rebalances": len(picks), "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "window": f"{START}~{END}",
    }, indent=2))
    log(f"  saved {OUT_ROOT} ({len(picks)} dates, {time.time()-t0:.0f}s)")
    return OUT_ROOT


def run_sim(offset, out_base, end_date):
    cmd = [
        sys.executable, "-u", str(AKQ / "examples" / "v41_run_akquant_v34.py"),
        "--tag", f"V14_{offset}",
        "--picks-base", str(OUT_BASE / "picks"),
        "--actions", "on", "--ca-mode", "all", "--prices", "raw",
        "--end", end_date,
        "--out-base", str(out_base),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
    rt = time.time() - t0
    ok = (out_base / f"V14_{offset}" / "result.json").exists()
    log(f"  sim offset={offset} end={end_date}: {'OK' if ok else 'FAIL'} ({rt:.0f}s)")
    if not ok:
        log(f"  tail: {(r.stdout or '')[-400:]}")
    return ok


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    log(f"matrix: {MAT}")
    log(f"offsets: {OFFSETS}")

    # 1) derive 10 picks sets (offset-aware, from matrix, ~3s each)
    log("===== Step 1: derive 10 picks sets =====")
    for off in OFFSETS:
        derive_picks_offset(MAT, V34, off, OUT_BASE / "picks")

    # 2) full-window sims (2010-2025)
    log("===== Step 2: 10 full-window sims (2010-2025, +CA) =====")
    for off in OFFSETS:
        run_sim(off, OUT_BASE / "sims", END)

    # 3) 2026 OOS sims (2010-2026-08-27)
    log("===== Step 3: 10 2026 OOS sims (2010-2026-08-27, +CA) =====")
    for off in OFFSETS:
        run_sim(off, OUT_BASE / "sims_2026", OOS_END)

    log("DONE stage5 picks + sims")


if __name__ == "__main__":
    main()
