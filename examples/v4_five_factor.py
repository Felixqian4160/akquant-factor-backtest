"""v4 五因子共振投票 — 集合级驱动 (断点续传, 用户合同 2026-10-11).

对每个 5 因子集合: signals -> build (SW047 参数: hold=0.65 cool=3 cap=60 pos=15,
rank=score) -> v42 sim (holding-bars=60) -> 指标(含直线度 smoothness) -> journal.
参数统一, 单变量 = 因子集合本身。

用法:
  --plan            列出集合与状态
  --only s1,s2      只跑指定集合 (逗号分隔前缀)
  --offset N        网格偏移 (默认 0)
  --report          汇总报告 (ratio + 直线度)
  --curves          净值曲线图 (全部已完成集合 + SW047 + 基准)

产物: evidence/v4_five_factor_20261011/{journal.jsonl, v4_report.md, charts/}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
OUT = AKQ / "evidence" / "v4_five_factor_20261011"
FS = OUT / "run_factsets"
JOURNAL = OUT / "journal.jsonl"
FR = AKQ / "examples" / "v3_resonance_framework.py"
PICKS = AKQ / "evidence" / "v3_resonance_20261010" / "picks"
SIMS = AKQ / "evidence" / "v3_resonance_20261010" / "sims"
SWEEP_JOURNAL = AKQ / "evidence" / "v3_sweep_20261010" / "journal.jsonl"
BENCH = AKQ / "evidence" / "v2_screen_20261010" / "charts"
PY = sys.executable

BASE_PARAMS = ["--hold-pctl", "0.65", "--cooldown", "3", "--max-hold-bars", "60",
               "--n-pos", "15", "--rank-by", "score"]
ORDER = ["s1_top5_t", "s1d_top5_t_div", "s2_consist", "s3_family", "s4_control", "s1_n3", "s1_n8", "s1_n4"]

# 单变量筛选 (从 s1 基线出发, 变量 = 退出/冷却/买入门槛/持仓数; 目的 = 降低回撤提高直线度)
SCREEN_VARIANTS = [
    ("V4_S1_BP75_o0", ["--bear-pctl", "0.75"]),
    ("V4_S1_BP50_o0", ["--bear-pctl", "0.50"]),
    ("V4_S1_ENT99_o0", ["--ent-pctl", "0.99"]),
    ("V4_S1_CD6_o0", ["--cooldown", "6"]),
    ("V4_S1_CD10_o0", ["--cooldown", "10"]),
    ("V4_S1_CAP100_o0", ["--max-hold-bars", "100"]),
    ("V4_S1_POS10_o0", ["--n-pos", "10"]),
]

# n3 (3因子) 微调组: 从最直形态出发
SCREEN_VARIANTS_N3 = [
    ("V4_S1_N3_CD6_o0", ["--cooldown", "6"]),
    ("V4_S1_N3_POS10_o0", ["--n-pos", "10"]),
    ("V4_S1_N3_POS20_o0", ["--n-pos", "20"]),
    ("V4_S1_N3_CAP100_o0", ["--max-hold-bars", "100"]),
    ("V4_S1_N3_BP75_o0", ["--bear-pctl", "0.75"]),
    ("V4_S1_N3_CD6BP75_o0", ["--cooldown", "6", "--bear-pctl", "0.75"]),
    ("V4_S1_N3_UP70_o0", ["--up-th", "0.70"]),
    ("V4_S1_N3_NW3_o0", ["--need-win", "3"]),
    ("V4_S1_N3_ENT90_o0", ["--ent-pctl", "0.90"]),
    ("V4_S1_N3_M1SCORE_o0", ["--decide", "score"]),
    ("V4_S1_N3_M2GATE_o0", ["--decide", "gate"]),
    ("V4_S1_N3_M3DISP_o0", ["--decide", "disp"]),
]

# 筛选赢家的相位验证 (跨相位机制裁决)
PHASE_VARIANTS = [
    ("V4_S1_CD6", "s1_top5_t", ["--cooldown", "6"]),
    ("V4_S1D_CD6", "s1d_top5_t_div", ["--cooldown", "6"]),
    ("V4_S1_N3", "s1_n3", []),
    ("V4_S1_N3_CD6", "s1_n3", ["--cooldown", "6"]),
    ("V4_S1_N3_BP75", "s1_n3", ["--bear-pctl", "0.75"]),
    ("V4_S1_N3_M3DISP", "s1_n3", ["--decide", "disp"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_tag(set_tag, offset):
    return f"V4_{set_tag.upper()}_o{offset}"


def run_cmd(args, timeout=None):
    r = subprocess.run([PY, "-u", str(FR)] + args, capture_output=True, text=True,
                       cwd=str(AKQ), timeout=timeout)
    for ln in (r.stdout or "").strip().splitlines()[-6:]:
        log("  | " + ln)
    if r.returncode != 0:
        log("  !! exit " + str(r.returncode) + " stderr: " + (r.stderr or "")[-300:])
    return r.returncode == 0


def seg_metrics_and_smooth(nav_path):
    df = pd.read_csv(nav_path)
    df["d"] = pd.to_datetime(df["date"])
    df = df.sort_values("d")
    df = df[df["d"] <= pd.Timestamp("2025-12-31")]
    pre = df[df["d"] < pd.Timestamp("2018-01-01")]
    if not len(pre):
        return None
    base = float(pre["value"].iloc[-1])
    seg = df[df["d"] >= pd.Timestamp("2018-01-01")]
    vals = seg["value"].to_numpy(dtype=float) / base
    curve = np.concatenate(([1.0], vals))
    peak = np.maximum.accumulate(curve)
    mdd = float(np.min(curve / peak - 1.0))
    ret = float(vals[-1] - 1.0)
    years = (seg["d"].iloc[-1] - seg["d"].iloc[0]).days / 365.25
    ann = float((1.0 + ret) ** (1.0 / years) - 1.0)
    daily = np.diff(vals) / vals[:-1]
    sh = float(daily.mean() / daily.std() * np.sqrt(244)) if daily.std() > 0 else 0.0
    # 直线度: log NAV 线性拟合 R2 + 滚动窗口正收益率
    x = np.arange(len(curve))
    y = np.log(curve)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    roll60 = curve[60:] / curve[:-60] - 1
    roll120 = curve[120:] / curve[:-120] - 1
    return {"ret": ret, "ann": ann, "mdd": mdd, "sharpe": sh, "ratio": ann / abs(mdd),
            "r2_loglin": float(r2),
            "pos60": float((roll60 >= 0).mean()), "worst60": float(roll60.min()),
            "pos120": float((roll120 >= 0).mean()), "worst120": float(roll120.min())}


def load_journal():
    if not JOURNAL.exists():
        return []
    return [json.loads(ln) for ln in JOURNAL.read_text().splitlines() if ln.strip()]


def group_key(e):
    """归一化分组: 'V4_S1_CD6_o0' -> 'V4_S1_CD6'; s1_n3 -> V4_S1_N3 (journaling 别名)."""
    k = e["set"]
    m = re.match(r"^(.*)_o\d+$", k)
    if m:
        k = m.group(1)
    return {"s1_n3": "V4_S1_N3", "s1_n8": "V4_S1_N8"}.get(k, k)


def ensure_journal(set_tag, rtag, offset):
    entries = load_journal()
    if any(e["tag"] == rtag for e in entries):
        return
    res_p = SIMS / rtag / "result.json"
    if not res_p.exists():
        return
    m = json.loads(res_p.read_text())["metrics"]
    seg = seg_metrics_and_smooth(SIMS / rtag / "nav.csv")
    pm = json.loads((PICKS / rtag / "picks_meta.json").read_text())
    picks_day = float(np.mean([v["n_picks"] for v in pm["per_day"].values()]))
    c = pm.get("contract", {})
    params = {k: c.get(k) for k in ("ent_pctl", "ent_min", "hold_pctl", "hold_min",
                                    "bear_pctl", "bear_max", "cooldown_grids",
                                    "max_hold_bars", "n_pos", "rank_by")}
    ent = {"tag": rtag, "set": set_tag, "offset": offset,
           "full": {"ann": m["annualized_return"], "mdd": -m["max_drawdown_pct"] / 100.0,
                    "sharpe": m["sharpe_ratio"], "trades": m["closed_trade_count"],
                    "win": m["win_rate"] / 100.0},
           "seg": seg, "picks_day": picks_day, "params": params,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with JOURNAL.open("a") as f:
        f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    log(f"journal + {rtag}: seg ann {seg['ann']:.2%} mdd {seg['mdd']:.1%} ratio {seg['ratio']:.3f} "
        f"r2 {seg['r2_loglin']:.3f} pos60 {seg['pos60']:.0%}")


def sweep_ref():
    """SW047/SW026 o0 参考行 (与 v4 同 nav 计算口径)."""
    out = {}
    if not SWEEP_JOURNAL.exists():
        return out
    for ln in SWEEP_JOURNAL.read_text().splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("ok") and e.get("offset") == 0 and e.get("idx") in (41, 47, 26):
            rtag = e["tag"]
            nav = SIMS / rtag / "nav.csv"
            if nav.exists():
                s = seg_metrics_and_smooth(nav)
                if s:
                    p = e["params"]
                    out[f"SWe{ e['idx']:03d}"] = {"tag": rtag, "params": p, "seg": s}
    return out


def report():
    entries = load_journal()
    if not entries:
        log("no journal entries yet")
        return
    entries = sorted(entries, key=lambda e: -(e["seg"]["ratio"] if e["seg"] else 0))
    rows = []
    for e in entries:
        s = e["seg"]
        if s is None:
            continue
        rows.append(f"| {e['set']} | {e['offset']} | **{s['ratio']:.3f}** | {s['ann']:.2%} | "
                    f"{s['mdd']:.1%} | {s['r2_loglin']:.3f} | {s['pos60']:.0%} | {s['worst60']:.1%} | "
                    f"{e['full']['ann']:.2%} | {e['full']['trades']:.0f} | {e['picks_day']:.1f} |")
    refs = sweep_ref()
    ref_rows = []
    for k, v in sorted(refs.items()):
        s = v["seg"]
        ref_rows.append(f"| {k} (413因子) | 0 | {s['ratio']:.3f} | {s['ann']:.2%} | {s['mdd']:.1%} | "
                        f"{s['r2_loglin']:.3f} | {s['pos60']:.0%} | {s['worst60']:.1%} | - | - | - |")
    md = ["# v4 五因子共振投票 — 结果 (2018-2025 段)", "",
          "| 集合 | o | 收益/回撤比 | 段年化 | 段MDD | 直线度R² | 60d正收益率 | 60d最差 | 全期年化 | trades | picks/日 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + ref_rows
    by = {}
    for e in entries:
        if e["seg"]:
            by.setdefault(group_key(e), []).append(e)
    multi = {k: v for k, v in by.items() if len(v) >= 2}
    if multi:
        md += ["", "## 相位聚合 (多相位)", "",
               "| 集合 | 相位 | mean ratio [min,max] | mean 段年化 | worst MDD | mean R² | mean pos60 |",
               "|---|---|---|---|---|---|---|"]
        for k, es in sorted(multi.items(), key=lambda kv: -np.mean([x["seg"]["ratio"] for x in kv[1]])):
            rs = [x["seg"]["ratio"] for x in es]
            ans = [x["seg"]["ann"] for x in es]
            mdds = [x["seg"]["mdd"] for x in es]
            r2s = [x["seg"]["r2_loglin"] for x in es]
            p60 = [x["seg"]["pos60"] for x in es]
            md.append(f"| {k} | {len(es)} | **{np.mean(rs):.3f}** [{min(rs):.3f}, {max(rs):.3f}] | "
                      f"{np.mean(ans):.2%} | {min(mdds):.1%} | {np.mean(r2s):.3f} | {np.mean(p60):.0%} |")
    (OUT / "v4_report.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))
    log(f"saved {OUT/'v4_report.md'}")


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [e for e in load_journal() if e["seg"]]
    by = {}
    for e in entries:
        by.setdefault(group_key(e), []).append(e)
    PREFERRED = ["s1d_top5_t_div", "V4_S1_N3_M3DISP", "V4_S1_N3_BP75", "V4_S1_N3", "V4_S1_N3_CD6"]  # 直线候选优先
    order = sorted(by.keys(), key=lambda k: -np.mean([x["seg"]["ratio"] for x in by[k]]))
    order = [k for k in order if len(by[k]) >= 2]
    order = [k for k in PREFERRED if k in order] + [k for k in order if k not in PREFERRED]
    order = order[:6]
    if not order:
        log("no entries")
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=110)
    colors = ["#1e88e5", "#2e7d32", "#f57c00", "#8e24aa", "#e53935", "#00897b", "#6d4c41"]

    def load_ser(tag):
        df = pd.read_csv(SIMS / tag / "nav.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp("2025-12-31")].sort_values("date")
        return df.set_index("date")["value"]

    def phase_mean(es):
        segs, fulls = [], []
        for e in es:
            ser = load_ser(e["tag"])
            pre = ser[ser.index < pd.Timestamp("2018-01-01")]
            if not len(pre):
                continue
            b = float(pre.iloc[-1])
            segs.append((ser[ser.index >= pd.Timestamp("2018-01-01")] / b).rename(e["tag"]))
            fulls.append((ser / float(ser.iloc[0])).rename(e["tag"]))
        s = pd.concat(segs, axis=1).dropna().mean(axis=1)
        f = pd.concat(fulls, axis=1).dropna().mean(axis=1)
        return s, f

    for j, k in enumerate(order):
        es = by[k]
        seg, full = phase_mean(es)
        mean_r = float(np.mean([x["seg"]["ratio"] for x in es]))
        lab = f"{k} (mean ratio {mean_r:.3f}, {len(es)}相位)"
        axes[0].plot(seg.index, seg.values, lw=2.2 if j == 0 else 1.6, color=colors[j % 7], label=lab)
        axes[1].plot(full.index, full.values, lw=1.5, color=colors[j % 7], label=lab)
    # SW047 ref (phase-mean over its sweep offsets)
    sw = []
    sw_rs = []
    if SWEEP_JOURNAL.exists():
        for ln in SWEEP_JOURNAL.read_text().splitlines():
            if not ln.strip():
                continue
            e = json.loads(ln)
            if e.get("ok") and e.get("idx") == 47:
                sw.append({"tag": e["tag"], "offset": e["offset"]})
                if e.get("seg_ratio"):
                    sw_rs.append(e["seg_ratio"])
    if sw:
        seg, full = phase_mean(sw)
        lab = f"SW047 pool413 (mean ratio {np.mean(sw_rs):.3f}, {len(sw)}相位)"
        axes[0].plot(seg.index, seg.values, lw=1.4, color="#546e7a", ls="--", label=lab)
        axes[1].plot(full.index, full.values, lw=1.2, color="#546e7a", ls="--", label="SW047 pool413")
    for name, fname in [("HS300", "hs300_nav.csv"), ("EW portfolio", "ew_nav.csv")]:
        p = BENCH / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        dcol = "date" if "date" in df.columns else "d"
        df["date"] = pd.to_datetime(df[dcol])
        s = df.set_index("date")["value"].sort_index()
        s = s[s.index <= pd.Timestamp("2025-12-31")]
        pre = s[s.index < pd.Timestamp("2018-01-01")]
        if not len(pre):
            continue
        b = float(pre.iloc[-1])
        axes[0].plot(s[s.index >= pd.Timestamp("2018-01-01")].index,
                     (s[s.index >= pd.Timestamp("2018-01-01")] / b).values, lw=1.2, ls=":", color="#9e9e9e", label=name)
        axes[1].plot(s.index, (s / float(s.iloc[0])).values, lw=1.2, ls=":", color="#9e9e9e", label=name)
    axes[0].set_title("v4 five-factor resonance - segment 2018-2025 (base = 1)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-02-28"))
    axes[1].set_title("v4 five-factor resonance - full 2010-2025 (log)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-02-28"))
    fig.tight_layout()
    CH = OUT / "charts"
    CH.mkdir(parents=True, exist_ok=True)
    fig.savefig(CH / "v4_five_factor_o0.png")
    plt.close(fig)
    log(f"png -> {CH/'v4_five_factor_o0.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--phases", default=None)
    ap.add_argument("--screen", default=None, help="factset tag to run SCREEN_VARIANTS on")
    ap.add_argument("--screen-n3", default=None, help="factset tag to run SCREEN_VARIANTS_N3 on")
    ap.add_argument("--vphases", default=None, help="comma list of PHASE_VARIANTS keys")
    args = ap.parse_args()
    order = ORDER
    if args.only:
        order = [s for s in ORDER if any(s.startswith(x) for x in args.only.split(","))]
    if args.plan:
        for s in order:
            rtag = run_tag(s, args.offset)
            done = (SIMS / rtag / "result.json").exists()
            log(f"{s:18s} -> {rtag:28s} {'DONE' if done else 'pending'}")
        return
    if args.report:
        report()
        return
    if args.curves:
        curves()
        return
    if args.phases:
        order = [s for s in ORDER if any(s.startswith(x) for x in args.phases.split(","))]
        for s in order:
            ff = FS / f"{s}.json"
            for off in (1, 2, 3, 4):
                rtag = run_tag(s, off)
                if (SIMS / rtag / "result.json").exists():
                    log(f"{rtag}: sim exists, skip")
                    ensure_journal(s, rtag, off)
                    continue
                log(f"=== {s} -> {rtag} ===")
                if not run_cmd(["--factset", str(ff), "--build", "--tag", rtag,
                                "--offset", str(off)] + BASE_PARAMS):
                    continue
                if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", rtag,
                                "--runner", "v42", "--holding-bars", "60"], timeout=900):
                    continue
                log(f"{rtag} done")
                ensure_journal(s, rtag, off)
        return
    if args.screen:
        ff = FS / f"{args.screen}.json"
        for vtag, extra in SCREEN_VARIANTS:
            if (SIMS / vtag / "result.json").exists():
                log(f"{vtag}: sim exists, skip")
                ensure_journal(vtag, vtag, 0)
                continue
            log(f"=== {vtag} ===")
            if not run_cmd(["--factset", str(ff), "--build", "--tag", vtag,
                            "--offset", "0"] + BASE_PARAMS + extra):
                continue
            if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", vtag,
                            "--runner", "v42", "--holding-bars", "60"], timeout=900):
                continue
            ensure_journal(vtag, vtag, 0)
        return
    if args.screen_n3:
        ff = FS / f"{args.screen_n3}.json"
        for vtag, extra in SCREEN_VARIANTS_N3:
            if (SIMS / vtag / "result.json").exists():
                log(f"{vtag}: sim exists, skip")
                ensure_journal(vtag, vtag, 0)
                continue
            log(f"=== {vtag} ===")
            if not run_cmd(["--factset", str(ff), "--signals", "99"] + extra):
                continue
            if not run_cmd(["--factset", str(ff), "--build", "--tag", vtag,
                            "--offset", "0"] + BASE_PARAMS + extra):
                continue
            if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", vtag,
                            "--runner", "v42", "--holding-bars", "60"], timeout=900):
                continue
            ensure_journal(vtag, vtag, 0)
        return
    if args.vphases:
        wanted = set(args.vphases.split(","))
        for base, s, extra in PHASE_VARIANTS:
            if base not in wanted:
                continue
            ff = FS / f"{s}.json"
            for off in range(5):
                rtag = f"{base}_o{off}"
                if (SIMS / rtag / "result.json").exists():
                    log(f"{rtag}: sim exists, skip")
                    ensure_journal(base, rtag, off)
                    continue
                log(f"=== {rtag} ===")
                if not run_cmd(["--factset", str(ff), "--build", "--tag", rtag,
                                "--offset", str(off)] + BASE_PARAMS + extra):
                    continue
                if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", rtag,
                                "--runner", "v42", "--holding-bars", "60"], timeout=900):
                    continue
                ensure_journal(base, rtag, off)
        return
    for s in order:
        rtag = run_tag(s, args.offset)
        if (SIMS / rtag / "result.json").exists():
            log(f"{rtag}: sim exists, skip")
            ensure_journal(s, rtag, args.offset)
            continue
        t0 = time.time()
        ff = FS / f"{s}.json"
        log(f"=== {s} -> {rtag} ===")
        if not run_cmd(["--factset", str(ff), "--signals", "99"]):
            continue
        if not run_cmd(["--factset", str(ff), "--build", "--tag", rtag,
                        "--offset", str(args.offset)] + BASE_PARAMS):
            continue
        if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", rtag,
                        "--runner", "v42", "--holding-bars", "60"], timeout=900):
            continue
        log(f"{rtag} done in {time.time()-t0:.0f}s")
        ensure_journal(s, rtag, args.offset)


if __name__ == "__main__":
    main()
