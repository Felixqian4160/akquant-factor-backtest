"""Causal (settle-lag corrected) v34-lb20 — FULL 10-offset formal evaluation.

- picks : score window [i-41:i-21] (all vintages settled before decision day i).
          Built into evidence/audit_lookahead_20261010/picks_causal/ (resumable,
          V14_0/V14_10 already exist from the audit).
- sims  : 10 offsets x 2 windows (2010-2025 / 2010-2026.08) via v41 runner into
          evidence/audit_lookahead_20261010/{sims,sims_2026}/ (resumable).
- report: aggregate table + CAUSAL_EVAL.md + comparison chart vs archived baseline.

Usage:
  python3.12 -u examples/causal_full_eval_20261010.py --build
  python3.12 -u examples/causal_full_eval_20261010.py --run 4
  python3.12 -u examples/causal_full_eval_20261010.py --status
  python3.12 -u examples/causal_full_eval_20261010.py --report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import polars as pl

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
MAT = AKQ / "evidence" / "stage_a_20261007" / "matrix_v34_ADJ.parquet"
V34 = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
ROUTER_MAP = AKQ / "evidence" / "causal_zigzag_router_20261006" / "router_map.json"
OUT = AKQ / "evidence" / "audit_lookahead_20261010"
PICKS = OUT / "picks_causal"
ARCH_SIMS = AKQ / "evidence" / "stage5_20261007"

OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
WINDOWS = [("sims", "2025-12-31"), ("sims_2026", "2026-08-27")]
START, END = "2010-01-01", "2025-12-31"
STEP, LOOKBACK, SETTLE_LAG = 20, 20, 21
TOP_K, K_NOM = 10, 10
MV_BULL, MS_BULL, MX_BULL = 2, 5, 10
MV_BEAR, MS_BEAR, MX_BEAR = 4, 5, 10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Data:
    def __init__(self):
        mat = pl.read_parquet(MAT)
        self.factors = [c for c in mat.columns if c != "trade_date"]
        self.M = mat.select(self.factors).to_numpy()
        self.dates = [str(d)[:10] for d in mat["trade_date"].to_list()]
        self.dpos = {d: i for i, d in enumerate(self.dates)}
        self.router = json.loads(ROUTER_MAP.read_text())
        import pandas as pd
        self._pd = pd
        # union of all rebalance dates across 10 offsets
        rd = set()
        for off in OFFSETS:
            for d in self.dates[off::STEP]:
                if START <= d <= END:
                    rd.add(d)
        self.rd_all = sorted(rd)
        rdt = pl.Series("trade_date", [pd.Timestamp(d).to_pydatetime() for d in self.rd_all]).dt.cast_time_unit("ms")
        self.day_panels = pl.read_parquet(V34, columns=["trade_date", "ts_code"] + self.factors).filter(
            pl.col("trade_date").is_in(rdt)
        )


def derive_causal(data: Data, offset: int):
    picks, meta = {}, {}
    for rd in [d for d in data.dates[offset::STEP] if START <= d <= END]:
        i = data.dpos[rd]
        hi = i - SETTLE_LAG
        lo = max(0, hi - LOOKBACK)
        if hi - lo < 10:
            continue
        win = data.M[lo:hi]
        with np.errstate(all="ignore"):
            counts = np.isfinite(win).sum(axis=0)
            scores = np.where(counts >= 10, np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=0), np.nan)
        scored = [(fi, scores[fi]) for fi in range(len(data.factors)) if np.isfinite(scores[fi])]
        if not scored:
            continue
        ranked = sorted(scored, key=lambda kv: -kv[1])
        active = [data.factors[fi] for fi, _ in ranked[:TOP_K]]
        regime = data.router.get(rd, "bull_neutral")
        if regime == "bull_neutral":
            mv_, ms, mx = MV_BULL, MS_BULL, MX_BULL
        else:
            mv_, ms, mx = MV_BEAR, MS_BEAR, MX_BEAR
        day = data.day_panels.filter(pl.col("trade_date").dt.strftime("%Y-%m-%d") == rd)
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
            votes[valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]] += 1
        order = sorted(range(len(codes)), key=lambda k: (-int(votes[k]), str(codes[k])))
        selected = [k for k in order if int(votes[k]) >= mv_]
        if len(selected) < ms:
            selected = order[:ms]
        selected = selected[:mx]
        if not selected:
            continue
        picks[rd] = {str(codes[k]): int(votes[k]) for k in selected}
        meta[rd] = {"regime": regime, "selector": "factor_return_voting_causal", "offset": offset,
                    "n_active_factors": len(active), "n_picks": len(selected), "max_votes": int(votes.max())}
    return picks, meta


def build(data: Data):
    for off in OFFSETS:
        outdir = PICKS / f"V14_{off}"
        if (outdir / "picks.json").exists():
            log(f"picks exist, skip V14_{off}")
            continue
        pk, mt = derive_causal(data, off)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "picks.json").write_text(json.dumps(pk, ensure_ascii=False, indent=1))
        (outdir / "picks_meta.json").write_text(json.dumps(mt, ensure_ascii=False, indent=1))
        log(f"built V14_{off}: {len(pk)} dates")


def result_path(off, wname):
    return OUT / wname / f"V14_{off}" / "result.json"


def run_sims(max_n: int):
    todo = [(o, w, e) for w, e in WINDOWS for o in OFFSETS if not result_path(o, w).exists()]
    log(f"missing sims: {len(todo)}; running up to {max_n}")
    ran = 0
    for off, wname, edate in todo:
        if ran >= max_n:
            break
        log(f"  run V14_{off} / {wname} ...")
        t0 = time.time()
        cmd = [sys.executable, "-u", str(RUNNER), "--tag", f"V14_{off}",
               "--picks-base", str(PICKS), "--actions", "on", "--ca-mode", "all",
               "--prices", "raw", "--end", edate, "--out-base", str(OUT / wname)]
        subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
        ok = result_path(off, wname).exists()
        log(f"    {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)")
        if not ok:
            return
        ran += 1
    remaining = sum(1 for w, e in WINDOWS for o in OFFSETS if not result_path(o, w).exists())
    log(f"done this call: {ran}; remaining: {remaining}")


def collect(sims_dir):
    rows = []
    for off in OFFSETS:
        p = sims_dir / f"V14_{off}" / "result.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())["metrics"]
        rows.append({"offset": off, "total_pct": m["total_return_pct"], "ann": m["annualized_return"],
                     "sharpe": m.get("sharpe_ratio"), "mdd": m["max_drawdown_pct"],
                     "win": m["win_rate"], "trades": m["closed_trade_count"]})
    return rows


def report():
    res = {}
    for wname, label in (("sims", "2010-2025"), ("sims_2026", "2010-2026.08")):
        rows = collect(OUT / wname)
        res[wname] = rows
        anns = [r["ann"] for r in rows]
        log(f"[{label}] n={len(rows)}: ann_mean={np.mean(anns):.4f} +- {np.std(anns, ddof=1):.4f} "
            f"min={min(anns):.4f} max={max(anns):.4f}")

    lines = ["# 因果修正版 v34-lb20 — 全 10 offset 正式评测（2026-10-10）", "",
             "score 窗口 = [i-41:i-21]（全部已结算）；其余合同与归档版完全一致。", ""]
    for wname, label in (("sims", "2010-2025"), ("sims_2026", "2010-2026.08")):
        rows = res[wname]
        if not rows:
            continue
        anns = [r["ann"] for r in rows]
        lines += [f"## {label}", "",
                  "| offset | total% | 年化 | Sharpe | MDD% | win% | trades |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {r['offset']} | {r['total_pct']:,.0f} | {r['ann']:.4f} | {r['sharpe']:.3f} | "
                         f"{r['mdd']:.1f} | {r['win']:.1f} | {int(r['trades'])} |")
        lines += ["", f"- 年化均值 = **{np.mean(anns):.2%} ± {np.std(anns, ddof=1):.2%}**；"
                      f"区间 [{min(anns):.2%}, {max(anns):.2%}]；"
                      f"全正 = {'是' if all(a > 0 for a in anns) else '否'}", ""]
    lines += ["## 对照（归档原版 vs 因果修正）", "",
              "| | 年化均值 full | 年化均值 oos | Sharpe均值(full) | MDD worst |",
              "|---|---:|---:|---:|---:|"]
    import statistics
    def agg(rows):
        anns = [r["ann"] for r in rows]
        return np.mean(anns), (min([r["mdd"] for r in rows]) if rows else None), max([r["mdd"] for r in rows])
    a_full, am_mdd_min, am_mdd_max = agg(res["sims"])
    a_oos, ao_mdd_min, ao_mdd_max = agg(res["sims_2026"])
    lines += [f"| 归档原版(10-07) | +75.83% | +72.64% | 2.158 | 38.1% |",
              f"| **因果修正(本次)** | **{a_full:.2%}** | **{a_oos:.2%}** | "
              f"{np.mean([r['sharpe'] for r in res['sims']]):.3f} | {am_mdd_max:.1f}% |", ""]
    (OUT / "CAUSAL_EVAL.md").write_text("\n".join(lines))
    (OUT / "causal_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    log(f"report: {OUT/'CAUSAL_EVAL.md'}")

    # comparison chart (10-offset mean NAV: causal vs archived vs HS300)
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", "/home/felix/.hermes/profiles/buffett/cache/scratch/mplcache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import pandas as pd

        def mean_nav(base, wname):
            navs = []
            for off in OFFSETS:
                p = base / wname / f"V14_{off}" / "nav.csv"
                if p.exists():
                    df = pd.read_csv(p).assign(d=lambda x: pd.to_datetime(x["date"]))
                    navs.append(df.set_index("d")["value"])
            if not navs:
                return None
            return pd.concat(navs, axis=1).mean(axis=1)

        cm = mean_nav(OUT, "sims_2026")
        am = mean_nav(ARCH_SIMS, "sims_2026")
        pan = pl.read_parquet(AUDIT_PANEL if False else V34, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
        pan = pan.filter(pl.col("idx_close").is_not_null()).to_pandas()
        pan["d"] = pd.to_datetime(pan["trade_date"])
        hs = pan.set_index("d")["idx_close"]

        fig, ax = plt.subplots(figsize=(14, 7))
        if am is not None:
            ax.plot(am.index, am.values / 1e6, color="crimson", linewidth=2.5, label=f"archived (leaky) mean (final ¥{am.iloc[-1]/1e6:,.0f}M)")
        if cm is not None:
            ax.plot(cm.index, cm.values / 1e6, color="steelblue", linewidth=2.5, label=f"causal fix mean (final ¥{cm.iloc[-1]/1e6:,.1f}M)")
        common = cm.index.intersection(hs.index)
        hs_n = hs.loc[common] / hs.loc[common].iloc[0] * 100.0
        ax.plot(common, hs_n.values, color="gray", linewidth=2.0, label=f"HS300 buy-hold (final ¥{hs_n.iloc[-1]:,.0f}M per ¥100M)")
        ax.set_yscale("log")
        ax.set_ylabel("NAV (Million ¥, log scale)", fontsize=12)
        ax.set_title("v34-lb20 10-offset mean NAV — causal fix vs archived (leaky) vs HS300", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", fontsize=10)
        ch = OUT / "charts_causal"
        ch.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(ch / "compare_causal_vs_archived.png", dpi=110)
        log(f"chart: {ch/'compare_causal_vs_archived.png'}")
    except Exception as exc:  # noqa: BLE001
        log(f"chart skipped: {type(exc).__name__}: {exc}")


def status():
    done = sum(1 for w, e in WINDOWS for o in OFFSETS if result_path(o, w).exists())
    log(f"causal sims done: {done}/20")
    for w, _ in WINDOWS:
        marks = "".join("+" if result_path(o, w).exists() else "." for o in OFFSETS)
        log(f"  {w}: [{marks}]")
    pk = sum(1 for o in OFFSETS if (PICKS / f"V14_{o}" / "picks.json").exists())
    log(f"causal picks built: {pk}/10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        status()
    elif args.build:
        build(Data())
    elif args.run is not None:
        run_sims(args.run)
    elif args.report:
        report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
