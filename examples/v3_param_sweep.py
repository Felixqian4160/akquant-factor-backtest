"""v3 resonance 参数扫描程序 — 完整的组合逐一测试工具 (2026-10-10)

目标: 在「收益/回撤比」(seg_ann / |seg_mdd|, 2018-2025) 上寻找优于当前
最优 (hold=0.50, cool=3, cap=40, n_pos=10) 的参数组合。

预注册协议 (PLAN.md 同步):
  - Wave-1 (筛选): 54 组全因子组合, 单相位 (offset 0), 只做排序筛选, 不下结论
  - Wave-2 (验证): 按 Wave-1 收益/回撤比 top-K (+当前最优) 做 5 相位 (offset 0-4),
    以相位均值作为最终裁决
  - 网格: hold_pctl {0.40,0.50,0.65} x cooldown {2,3,5} x max_hold_bars {30,40,60}
          x n_pos {10,15};  grid_step=5 与 bear=0.90 固定
  - 执行: v42 runner (--holding-bars = max_hold_bars), 其余合同与 v3 框架一致
  - 全部序贯执行 (不并行), 断点续传 (journal.jsonl)

用法:
  --plan          生成网格 + 打印
  --status        完成进度
  --run N         序贯执行最多 N 个缺失的 Wave-1 配置 (build+sim)
  --validate K    取 Wave-1 收益/回撤比 top-K (加当前最优) 跑相位 1-4
  --report        汇总排名 (含相位均值) -> sweep_report.md
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
V3 = AKQ / "examples" / "v3_resonance_framework.py"
V42 = AKQ / "examples" / "v42_run_akquant_v34.py"
OUT = AKQ / "evidence" / "v3_sweep_20261010"
V3OUT = AKQ / "evidence" / "v3_resonance_20261010"
PICKS = V3OUT / "picks"
SIMS = V3OUT / "sims"
JOURNAL = OUT / "journal.jsonl"

HOLD_PCTL = [0.40, 0.50, 0.65]
COOLDOWN = [2, 3, 5]
MAX_HOLD = [30, 40, 60]
N_POS = [10, 15]
CHAMPION = {"hold_pctl": 0.50, "cooldown": 3, "max_hold_bars": 40, "n_pos": 10}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def make_grid():
    grid = []
    for hp, cd, mh, np_ in itertools.product(HOLD_PCTL, COOLDOWN, MAX_HOLD, N_POS):
        grid.append({"hold_pctl": hp, "cooldown": cd, "max_hold_bars": mh, "n_pos": np_})
    return grid


def tag_of(idx):
    return f"SW{idx:03d}"


def load_journal():
    entries = []
    if JOURNAL.exists():
        for line in JOURNAL.read_text().splitlines():
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def done_keys(entries):
    return {(e["idx"], e["offset"]) for e in entries if e.get("ok")}


def seg_metrics(nav_path, start="2018-01-01", end="2025-12-31"):
    df = pd.read_csv(nav_path)
    df["d"] = pd.to_datetime(df["date"])
    df = df.sort_values("d")
    df = df[df["d"] <= pd.Timestamp(end)]
    pre = df[df["d"] < pd.Timestamp(start)]["value"]
    if len(pre) == 0:
        return None
    base = float(pre.iloc[-1])
    s = df[df["d"] >= pd.Timestamp(start)]
    v = s["value"].to_numpy(dtype=float) / base
    curve = np.concatenate(([1.0], v))
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1.0))
    ret = float(v[-1] - 1.0)
    years = (s["d"].iloc[-1] - s["d"].iloc[0]).days / 365.25
    ann = float((1.0 + ret) ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    d = np.diff(v) / v[:-1]
    sr = float(d.mean() / d.std() * np.sqrt(244)) if d.std() > 0 else 0.0
    return {"ann": ann, "mdd": mdd, "sr": sr, "ret": ret}


def journal_append(rec):
    OUT.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(idx, params, offset, wave):
    tag = tag_of(idx)
    mh = params["max_hold_bars"]
    build_cmd = [sys.executable, "-u", str(V3), "--build",
                 "--hold-pctl", str(params["hold_pctl"]), "--cooldown", str(params["cooldown"]),
                 "--max-hold-bars", str(mh), "--n-pos", str(params["n_pos"]),
                 "--offset", str(offset), "--tag", tag]
    sim_cmd = [sys.executable, "-u", str(V42), "--tag", tag,
               "--picks-base", str(PICKS), "--actions", "on", "--ca-mode", "all",
               "--prices", "raw", "--end", "2025-12-31", "--out-base", str(SIMS),
               "--holding-bars", str(mh)]
    # NOTE: offset>0 uses distinct sim dirs via tag suffix — handled below
    if offset > 0:
        tag_off = f"{tag}_o{offset}"
        build_cmd[build_cmd.index("--tag") + 1] = tag_off
        sim_cmd[sim_cmd.index("--tag") + 1] = tag_off
        tag = tag_off
    r = subprocess.run(build_cmd, capture_output=True, text=True, cwd=str(AKQ), timeout=300)
    if r.returncode != 0:
        journal_append({"idx": idx, "tag": tag, "params": params, "offset": offset,
                        "wave": wave, "ok": False, "error": "build_failed",
                        "stderr": (r.stderr or "")[-300:]})
        return None
    r2 = subprocess.run(sim_cmd, capture_output=True, text=True, cwd=str(AKQ), timeout=400)
    rp = SIMS / tag / "result.json"
    nv = SIMS / tag / "nav.csv"
    if not rp.exists() or not nv.exists():
        journal_append({"idx": idx, "tag": tag, "params": params, "offset": offset,
                        "wave": wave, "ok": False, "error": "sim_failed",
                        "stderr": (r2.stderr or "")[-300:]})
        return None
    m = json.loads(rp.read_text())["metrics"]
    s = seg_metrics(nv)
    rec = {"idx": idx, "tag": tag, "params": params, "offset": offset, "wave": wave, "ok": True,
           "full": {"ann": m["annualized_return"], "sr": m["sharpe_ratio"],
                    "mdd": abs(m["max_drawdown_pct"]) / 100.0,
                    "trades": m["closed_trade_count"]},
           "seg": s,
           "seg_ratio": s["ann"] / abs(s["mdd"]) if s and s["mdd"] != 0 else None}
    journal_append(rec)
    return rec


def plan():
    grid = make_grid()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_plan.json").write_text(json.dumps({
        "objective": "seg_ann / |seg_mdd| (2018-2025)",
        "wave1": "54 combos, offset 0, screening only",
        "wave2": "top-K by ratio (+ champion) x offsets 0-4, verdict by phase mean",
        "grid": {"hold_pctl": HOLD_PCTL, "cooldown": COOLDOWN, "max_hold_bars": MAX_HOLD,
                 "n_pos": N_POS, "fixed": {"grid_step": 5, "bear_pctl": 0.90}},
        "champion": CHAMPION, "n_combos": len(grid)}, ensure_ascii=False, indent=1))
    log(f"grid: {len(grid)} combos -> {OUT/'sweep_plan.json'}")
    for i, g in enumerate(grid):
        log(f"  SW{i:03d}: {g}")


def status():
    entries = load_journal()
    done = done_keys(entries)
    n_done = len({k[0] for k in done})
    log(f"wave-1 done: {n_done}/54 configs ({len(entries)} journal entries)")


def run(n, max_secs=None):
    grid = make_grid()
    entries = load_journal()
    done = done_keys(entries)
    ran = 0
    t_start = time.time()
    for idx in range(len(grid)):
        if (idx, 0) in done:
            continue
        if ran >= n:
            break
        if max_secs and (time.time() - t_start) > max_secs:
            log("time budget reached; stopping before next config")
            break
        t0 = time.time()
        rec = run_one(idx, grid[idx], 0, 1)
        if rec:
            log(f"[{idx+1}/54] SW{idx:03d} {grid[idx]} -> seg {rec['seg']['ann']:.2%} "
                f"mdd {rec['seg']['mdd']:.1%} ratio {rec['seg_ratio']:.3f} ({time.time()-t0:.0f}s)")
        else:
            log(f"[{idx+1}/54] SW{idx:03d} FAILED")
        ran += 1
    log(f"batch done: {ran} configs this call")


def rank_wave1(entries):
    rows = {}
    for e in entries:
        if e.get("ok") and e["wave"] == 1:
            rows[e["idx"]] = e
    return sorted(rows.values(), key=lambda r: -(r["seg_ratio"] or 0))


def validate(k):
    grid = make_grid()
    entries = load_journal()
    done = done_keys(entries)
    ranked = rank_wave1(entries)
    # top-K by ratio + champion index
    champ_idx = next(i for i, g in enumerate(grid) if g == CHAMPION)
    picks = [r["idx"] for r in ranked[:k]]
    if champ_idx not in picks:
        picks.append(champ_idx)
    log(f"validate: idx {picks}")
    for idx in picks:
        for off in (1, 2, 3, 4):
            if (idx, off) in done:
                continue
            rec = run_one(idx, grid[idx], off, 2)
            if rec:
                log(f"  {tag_of(idx)} o{off}: seg {rec['seg']['ann']:.2%} ratio {rec['seg_ratio']:.3f}")
    log("validate batch done")


def report():
    entries = load_journal()
    grid = make_grid()
    by_idx = {}
    for e in entries:
        if not e.get("ok"):
            continue
        by_idx.setdefault(e["idx"], {})[e["offset"]] = e
    rows = []
    for idx, offs in by_idx.items():
        wave1 = offs.get(0)
        if not wave1:
            continue
        segs = [o["seg"] for o in offs.values() if o and o.get("seg")]
        ratios = [o["seg_ratio"] for o in offs.values() if o.get("seg_ratio") is not None]
        row = {"idx": idx, "params": grid[idx], "n_phase": len(offs),
               "seg_ann": wave1["seg"]["ann"], "seg_mdd": wave1["seg"]["mdd"],
               "ratio_w1": wave1["seg_ratio"],
               "full_ann": wave1["full"]["ann"], "full_mdd": wave1["full"]["mdd"],
               "trades": wave1["full"]["trades"]}
        if len(offs) >= 5:
            row["seg_ann_mean"] = float(np.mean([s["ann"] for s in segs]))
            row["seg_mdd_mean"] = float(np.mean([s["mdd"] for s in segs]))
            row["seg_mdd_worst"] = float(np.min([s["mdd"] for s in segs]))
            row["ratio_mean"] = float(np.mean(ratios))
            row["seg_sr_mean"] = float(np.mean([s["sr"] for s in segs]))
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("ratio_mean", r["ratio_w1"]) or 0))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    lines = ["# v3 参数扫描结果（收益/回撤比排名）", ""]
    lines.append("| # | idx | params | 相位 | seg ann | seg MDD | ratio | full ann | trades |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows[:30], 1):
        pm = json.dumps(r["params"], ensure_ascii=False)
        if "ratio_mean" in r:
            lines.append(f"| {i} | SW{r['idx']:03d} | {pm} | 5 | {r['seg_ann_mean']:.2%} | "
                         f"{r['seg_mdd_mean']:.1%} (w{r['seg_mdd_worst']:.1%}) | **{r['ratio_mean']:.3f}** | "
                         f"{r['full_ann']:.2%} | {int(r['trades'])} |")
        else:
            lines.append(f"| {i} | SW{r['idx']:03d} | {pm} | 1 | {r['seg_ann']:.2%} | "
                         f"{r['seg_mdd']:.1%} | {r['ratio_w1']:.3f} | {r['full_ann']:.2%} | {int(r['trades'])} |")
    (OUT / "sweep_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:1 + 3 + 30]))
    log(f"report -> {OUT/'sweep_report.md'} + sweep_results.json")


def curves(k=4):
    """Generate phase-mean NAV curves (top-K by ratio + champion) -> curves.json + PNG."""
    entries = load_journal()
    grid = make_grid()
    by_idx = {}
    for e in entries:
        if e.get("ok"):
            by_idx.setdefault(e["idx"], {})[e["offset"]] = e

    def ratio_of(idx):
        rs = [o["seg_ratio"] for o in by_idx[idx].values() if o.get("seg_ratio") is not None]
        return float(np.mean(rs))

    cand = [i for i in by_idx if 0 in by_idx[i]]
    ranked = sorted(cand, key=lambda i: -ratio_of(i))
    champ_idx = next(i for i, g in enumerate(grid) if g == CHAMPION)
    pick = ranked[:k]
    if champ_idx in by_idx and champ_idx not in pick:
        pick.append(champ_idx)

    def load_ser(tag):
        df = pd.read_csv(SIMS / tag / "nav.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp("2025-12-31")].sort_values("date")
        return df.set_index("date")["value"]

    def phase_mean(idx):
        segs, fulls = [], []
        for off, e in sorted(by_idx[idx].items()):
            s = load_ser(e["tag"])
            pre = s[s.index < pd.Timestamp("2018-01-01")]
            b = float(pre.iloc[-1]) if len(pre) else float(s.iloc[0])
            segs.append((s[s.index >= pd.Timestamp("2018-01-01")] / b).rename(off))
            fulls.append((s / float(s.iloc[0])).rename(off))
        ms = pd.concat(segs, axis=1).dropna().mean(axis=1)
        mf = pd.concat(fulls, axis=1).dropna().mean(axis=1)
        return ms, mf

    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": [], "benchmarks": []}
    for idx in pick:
        hp = by_idx[idx][0]["params"]
        label = f"SW{idx:03d} h{hp['hold_pctl']} c{hp['cooldown']} m{hp['max_hold_bars']} p{hp['n_pos']}"
        ms, mf = phase_mean(idx)
        out["configs"].append({
            "idx": idx, "label": label, "n_phases": len(by_idx[idx]), "ratio_mean": ratio_of(idx),
            "seg": {"dates": [d.strftime("%Y-%m-%d") for d in ms.index],
                    "values": [round(float(v), 4) for v in ms.values]},
            "full": {"dates": [d.strftime("%Y-%m-%d") for d in mf.index],
                     "values": [round(float(v), 4) for v in mf.values]},
        })
    BENCH = AKQ / "evidence" / "v2_screen_20261010" / "charts"
    for name, fname in [("HS300 index", "hs300_nav.csv"), ("EW portfolio", "ew_nav.csv")]:
        df = pd.read_csv(BENCH / fname)
        dcol = "date" if "date" in df.columns else "d"
        df["date"] = pd.to_datetime(df[dcol])
        s = df.set_index("date")["value"].sort_index()
        s = s[s.index <= pd.Timestamp("2025-12-31")]
        pre = s[s.index < pd.Timestamp("2018-01-01")]
        b = float(pre.iloc[-1]) if len(pre) else float(s.iloc[0])
        seg = s[s.index >= pd.Timestamp("2018-01-01")] / b
        full = s / float(s.iloc[0])
        out["benchmarks"].append({
            "label": name,
            "seg": {"dates": [d.strftime("%Y-%m-%d") for d in seg.index],
                    "values": [round(float(v), 4) for v in seg.values]},
            "full": {"dates": [d.strftime("%Y-%m-%d") for d in full.index],
                     "values": [round(float(v), 4) for v in full.values]},
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "curves.json").write_text(json.dumps(out, ensure_ascii=False))
    log(f"curves -> {OUT/'curves.json'} ({len(out['configs'])} configs)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=110)
        colors = ["#1e88e5", "#2e7d32", "#f57c00", "#8e24aa", "#e53935", "#00897b"]
        for j, c in enumerate(out["configs"]):
            axes[0].plot([pd.Timestamp(d) for d in c["seg"]["dates"]], c["seg"]["values"],
                         lw=2.2 if j == 0 else 1.4, color=colors[j % 6], label=c["label"])
            axes[1].plot([pd.Timestamp(d) for d in c["full"]["dates"]], c["full"]["values"],
                         lw=1.6, color=colors[j % 6], label=c["label"])
        for b in out["benchmarks"]:
            axes[0].plot([pd.Timestamp(d) for d in b["seg"]["dates"]], b["seg"]["values"],
                         lw=1.2, ls="--", color="#9e9e9e", label=b["label"])
            axes[1].plot([pd.Timestamp(d) for d in b["full"]["dates"]], b["full"]["values"],
                         lw=1.2, ls="--", color="#9e9e9e", label=b["label"])
        axes[0].set_title("v3 sweep - NAV segment 2018-2025 (phase-mean, base = 1)")
        axes[0].grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[0].set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-02-28"))
        axes[1].set_title("v3 sweep - NAV full 2010-2025 (log)")
        axes[1].set_yscale("log")
        axes[1].grid(alpha=0.3, which="both")
        axes[1].legend(fontsize=8)
        axes[1].set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-02-28"))
        fig.tight_layout()
        CH = OUT / "charts"
        CH.mkdir(parents=True, exist_ok=True)
        fig.savefig(CH / "sweep_curves.png")
        plt.close(fig)
        log(f"png -> {CH/'sweep_curves.png'}")
    except Exception as e:  # noqa: BLE001
        log(f"png failed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--max-secs", type=int, default=None)
    ap.add_argument("--validate", type=int, metavar="K")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()
    if args.plan:
        plan()
    elif args.status:
        status()
    elif args.run is not None:
        run(args.run, args.max_secs)
    elif args.validate is not None:
        validate(args.validate)
    elif args.report:
        report()
    elif args.curves:
        curves()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
