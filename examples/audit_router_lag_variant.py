"""Router-lag variant: causal score window + CONFIRMATION-DELAYED router (offset 0).

The archived router applies ZigZag leg.start on the pivot date itself; in real time a
pivot is only knowable after a 10% reversal from it (confirmation). This variant delays
the regime switch to the confirmation date. Everything else = causal-score pipeline.

Outputs: evidence/audit_lookahead_20261010/picks_causal_routerlag/V14_0/
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
OUT = AKQ / "evidence" / "audit_lookahead_20261010" / "picks_causal_routerlag"
PIVOTS = pathlib.Path("/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
                      "hs300_index_pivots_clean_20260919/hs300_index_pivots.json")
OFFSET = 0
START, END, STEP, LOOKBACK, SETTLE_LAG = "2010-01-01", "2025-12-31", 20, 20, 21
TOP_K, K_NOM = 10, 10
MV_BULL, MS_BULL, MX_BULL = 2, 5, 10
MV_BEAR, MS_BEAR, MX_BEAR = 4, 5, 10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    mat = pl.read_parquet(MAT)
    factors = [c for c in mat.columns if c != "trade_date"]
    M = mat.select(factors).to_numpy()
    all_dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
    dpos = {d: i for i, d in enumerate(all_dates)}

    # ---- confirmation-delayed router ----
    ix = (
        pl.read_parquet(V34, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
        .filter(pl.col("idx_close").is_not_null())
        .with_columns(pl.col("trade_date").dt.date().alias("d"))
        .select(["d", "idx_close"]).to_pandas()
    )
    import pandas as pd
    ix["d"] = pd.to_datetime(ix["d"]).dt.date
    ixdates = list(ix["d"]); idxv = ix["idx_close"].to_numpy(dtype=float)
    ixpos = {d: i for i, d in enumerate(ixdates)}

    piv = json.loads(PIVOTS.read_text())["pivots"]
    conf = []  # (confirm_date, kind) kind: 'up'->bull_neutral, 'down'->bear
    for p in piv:
        pd_ = pd.Timestamp(p["date"]).date()
        if pd_ not in ixpos:
            continue
        i0 = ixpos[pd_]
        cond = idxv[i0:] <= p["price"] * 0.9 if p["type"] == "peak" else idxv[i0:] >= p["price"] * 1.1
        hits = np.flatnonzero(cond)
        if len(hits) == 0:
            continue
        cdate = ixdates[i0 + int(hits[0])]
        conf.append((cdate, "down" if p["type"] == "peak" else "up"))
    conf.sort()
    log(f"confirmed pivots: {len(conf)} of {len(piv)}")
    delayed = {}
    ci = 0
    for d in all_dates:
        d_ = pd.Timestamp(d).date()
        while ci < len(conf) and conf[ci][0] <= d_:
            ci += 1
        delayed[d] = "bear" if ci and conf[ci - 1][1] == "down" else "bull_neutral"

    router_orig = json.loads(ROUTER_MAP.read_text())
    rebal_dates = [d for d in all_dates[OFFSET::STEP] if START <= d <= END]
    n_diff = sum(1 for d in rebal_dates if delayed.get(d) != router_orig.get(d, "bull_neutral"))
    log(f"rebalances where delayed regime != archived regime: {n_diff}/{len(rebal_dates)}")

    # ---- picks: causal score + delayed router ----
    rdt = pl.Series("trade_date", [pd.Timestamp(d).to_pydatetime() for d in rebal_dates]).dt.cast_time_unit("ms")
    day_panels = pl.read_parquet(V34, columns=["trade_date", "ts_code"] + factors).filter(
        pl.col("trade_date").is_in(rdt)
    )
    picks, meta = {}, {}
    for rd in rebal_dates:
        i = dpos[rd]
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
        regime = delayed.get(rd, "bull_neutral")
        if regime == "bull_neutral":
            mv_, ms, mx = MV_BULL, MS_BULL, MX_BULL
        else:
            mv_, ms, mx = MV_BEAR, MS_BEAR, MX_BEAR
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
        meta[rd] = {"regime": regime, "selector": "factor_return_voting_causal_routerlag",
                    "offset": OFFSET, "n_active_factors": len(active), "n_picks": len(selected)}

    outdir = OUT / f"V14_{OFFSET}"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (outdir / "picks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log(f"saved {outdir} ({len(picks)} dates)")
    log("DONE")


if __name__ == "__main__":
    main()
