"""v5 动态因子选择 — 每期按"近期效力"选 5 因子 → 共振投票 (用户合同 2026-10-11).

本质: 找到"在某个时期最有效的因子"的方法 → 选 5 个 → 投票.
三种效力度量 (全部 settled 因果口径, 窗口不含未结算行):
  V5_A60    : 近期 60 日收益均值 (快轮动)
  V5_B120IR : 近期 120 日信息比 mean/std (稳健画像)
  V5_C120HIT: 近期 120 日胜率 (正天数占比)
每 5 交易日网格: 用已结算数据给 pool20 打效力分 -> top-5 -> 其共振投票 (计数制).

用法:
  --run [--offset N]   signals -> build -> sim -> journal (断点续传)
  --plan / --report / --curves

产物: evidence/v5_dynamic_select_20261011/{journal.jsonl, v5_report.md, charts/}
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(AKQ / "examples"))
from v4_five_factor import seg_metrics_and_smooth, group_key  # noqa: E402

OUT = AKQ / "evidence" / "v5_dynamic_select_20261011"
FS = OUT / "run_factsets"
JOURNAL = OUT / "journal.jsonl"
CACHE_DIR = AKQ / "evidence" / "v3_resonance_20261010" / "_cache"
FR = AKQ / "examples" / "v3_resonance_framework.py"
PICKS = AKQ / "evidence" / "v3_resonance_20261010" / "picks"
SIMS = AKQ / "evidence" / "v3_resonance_20261010" / "sims"
V4_JOURNAL = AKQ / "evidence" / "v4_five_factor_20261011" / "journal.jsonl"
SWEEP_JOURNAL = AKQ / "evidence" / "v3_sweep_20261010" / "journal.jsonl"
BENCH = AKQ / "evidence" / "v2_screen_20261010" / "charts"
PY = sys.executable

BASE_PARAMS = ["--hold-pctl", "0.65", "--cooldown", "3", "--max-hold-bars", "60",
               "--n-pos", "15", "--rank-by", "score"]
POOL = "v5_pool20"
SELECT_VARIANTS = [
    ("V5_A60", ["--select-trail", "60", "--select-metric", "mean", "--select-topn", "5"]),
    ("V5_B120IR", ["--select-trail", "120", "--select-metric", "ir", "--select-topn", "5"]),
    ("V5_C120HIT", ["--select-trail", "120", "--select-metric", "hit", "--select-topn", "5"]),
    ("V5_C120HIT_M3", ["--select-trail", "120", "--select-metric", "hit", "--select-topn", "5",
                       "--decide", "disp"]),
    ("V5_D60MO", ["--select-trail", "60", "--select-metric", "mo", "--select-topn", "5"]),
    ("V5_E120MO", ["--select-trail", "120", "--select-metric", "mo", "--select-topn", "5"]),
    ("V5_F_WSUM_HIT", ["--select-trail", "120", "--select-metric", "hit", "--select-topn", "5",
                       "--select-mode", "wsum"]),
]
REF_SETS = ["s1d_top5_t_div", "V4_S1_N3", "V4_S1_N3_BP75", "s1_top5_t"]

# 池子大小单变量实验 (20 → 80 → 413)
POOL_VARIANTS = [
    ("V5_P80_C120HIT", "v5_pool80", ["--select-trail", "120", "--select-metric", "hit", "--select-topn", "5"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run_cmd(args, timeout=None):
    r = subprocess.run([PY, "-u", str(FR)] + args, capture_output=True, text=True,
                       cwd=str(AKQ), timeout=timeout)
    for ln in (r.stdout or "").strip().splitlines()[-7:]:
        log("  | " + ln)
    if r.returncode != 0:
        log("  !! exit " + str(r.returncode) + " stderr: " + (r.stderr or "")[-300:])
    return r.returncode == 0


def load_journal():
    if not JOURNAL.exists():
        return []
    return [json.loads(ln) for ln in JOURNAL.read_text().splitlines() if ln.strip()]


def ensure_journal(vtag, rtag, offset):
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
    ent = {"tag": rtag, "set": vtag, "offset": offset,
           "full": {"ann": m["annualized_return"], "mdd": -m["max_drawdown_pct"] / 100.0,
                    "sharpe": m["sharpe_ratio"], "trades": m["closed_trade_count"],
                    "win": m["win_rate"] / 100.0},
           "seg": seg, "picks_day": picks_day,
           "select": c.get("dynamic_select"),
           "params": {k: c.get(k) for k in ("ent_min", "hold_min", "bear_max", "cooldown_grids",
                                            "max_hold_bars", "n_pos", "rank_by")},
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    with JOURNAL.open("a") as f:
        f.write(json.dumps(ent, ensure_ascii=False) + "\n")
    log(f"journal + {rtag}: seg ann {seg['ann']:.2%} mdd {seg['mdd']:.1%} ratio {seg['ratio']:.3f} "
        f"R2 {seg['r2_loglin']:.3f} pos60 {seg['pos60']:.0%}")


def v4_refs():
    out = {}
    if not V4_JOURNAL.exists():
        return out
    by = {}
    for ln in V4_JOURNAL.read_text().splitlines():
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("seg"):
            by.setdefault(group_key(e), []).append(e)
    for k in REF_SETS:
        es = by.get(k, [])
        if len(es) < 2:
            continue
        out[k] = {"n": len(es),
                  "ratio": float(np.mean([e["seg"]["ratio"] for e in es])),
                  "ann": float(np.mean([e["seg"]["ann"] for e in es])),
                  "mdd": float(min(e["seg"]["mdd"] for e in es)),
                  "r2": float(np.mean([e["seg"]["r2_loglin"] for e in es])),
                  "pos60": float(np.mean([e["seg"]["pos60"] for e in es]))}
    return out


def report():
    entries = load_journal()
    rows = sorted(entries, key=lambda e: -(e["seg"]["ratio"] if e["seg"] else 0))
    md = ["# v5 动态因子选择 — 结果 (2018-2025 段, offset 0)" if entries else "", "",
          "| 变体 | 收益/回撤比 | 段年化 | 段MDD | 直线度R² | 60d正收益 | 全期年化 | trades |",
          "|---|---|---|---|---|---|---|---|"]
    for e in rows:
        s = e["seg"]
        if s is None:
            continue
        md.append(f"| {e['set']} | **{s['ratio']:.3f}** | {s['ann']:.2%} | {s['mdd']:.1%} | "
                  f"{s['r2_loglin']:.3f} | {s['pos60']:.0%} | {e['full']['ann']:.2%} | {e['full']['trades']:.0f} |")
    refs = v4_refs()
    if refs:
        md += ["", "## 静态对照 (v4, 5相位均值)", "",
               "| 配置 | ratio mean | 段年化 | worst MDD | mean R² | pos60 |", "|---|---|---|---|---|---|"]
        for k, v in refs.items():
            md.append(f"| {k} | {v['ratio']:.3f} | {v['ann']:.2%} | {v['mdd']:.1%} | {v['r2']:.3f} | {v['pos60']:.0%} |")
    (OUT / "v5_report.md").write_text("\n".join(md) + "\n")
    log("\n".join(md))
    log(f"saved {OUT/'v5_report.md'}")


def curves():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [e for e in load_journal() if e["seg"] and e["offset"] == 0]
    if not entries:
        log("no entries")
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=110)
    colors = ["#1e88e5", "#2e7d32", "#f57c00", "#8e24aa", "#e53935"]

    def load_ser(tag):
        df = pd.read_csv(SIMS / tag / "nav.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.Timestamp("2025-12-31")].sort_values("date")
        return df.set_index("date")["value"]

    for j, e in enumerate(sorted(entries, key=lambda x: x["tag"])):
        ser = load_ser(e["tag"])
        pre = ser[ser.index < pd.Timestamp("2018-01-01")]
        if not len(pre):
            continue
        b = float(pre.iloc[-1])
        seg = ser[ser.index >= pd.Timestamp("2018-01-01")] / b
        full = ser / float(ser.iloc[0])
        lab = f"{e['set']} (ratio {e['seg']['ratio']:.3f}, R² {e['seg']['r2_loglin']:.3f})"
        axes[0].plot(seg.index, seg.values, lw=2.0, color=colors[j % 5], label=lab)
        axes[1].plot(full.index, full.values, lw=1.5, color=colors[j % 5], label=lab)
    # refs
    for tag, lab, c in [("V4_S1_N3_o0", "n3 static (0.492)", "#546e7a"),
                        ("V4_S1D_TOP5_T_DIV_o0", "s1d static (0.666)", "#455a64")]:
        p = SIMS / tag
        if not p.exists():
            continue
        ser = load_ser(tag)
        pre = ser[ser.index < pd.Timestamp("2018-01-01")]
        b = float(pre.iloc[-1])
        seg = ser[ser.index >= pd.Timestamp("2018-01-01")] / b
        full = ser / float(ser.iloc[0])
        axes[0].plot(seg.index, seg.values, lw=1.3, ls="--", color=c, label=lab)
        axes[1].plot(full.index, full.values, lw=1.1, ls="--", color=c, label=lab)
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
                     (s[s.index >= pd.Timestamp("2018-01-01")] / b).values, lw=1.1, ls=":", color="#9e9e9e", label=name)
        axes[1].plot(s.index, (s / float(s.iloc[0])).values, lw=1.1, ls=":", color="#9e9e9e", label=name)
    axes[0].set_title("v5 dynamic factor select - segment 2018-2025 (base = 1)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-02-28"))
    axes[1].set_title("v5 dynamic factor select - full 2010-2025 (log)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-02-28"))
    fig.tight_layout()
    CH = OUT / "charts"
    CH.mkdir(parents=True, exist_ok=True)
    fig.savefig(CH / "v5_dynamic_select.png")
    plt.close(fig)
    log(f"png -> {CH/'v5_dynamic_select.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--curves", action="store_true")
    args = ap.parse_args()
    if args.plan:
        for vtag, pool, _x in [(v, POOL, x) for v, x in SELECT_VARIANTS] + POOL_VARIANTS:
            rtag = f"{vtag}_o{args.offset}"
            done = (SIMS / rtag / "result.json").exists()
            log(f"{vtag:16s} [{pool}] -> {rtag:20s} {'DONE' if done else 'pending'}")
        return
    if args.report:
        report()
        return
    if args.curves:
        curves()
        return
    jobs = [(v, POOL, x) for v, x in SELECT_VARIANTS] + POOL_VARIANTS
    for vtag, pool, extra in jobs:
        rtag = f"{vtag}_o{args.offset}"
        if (SIMS / rtag / "result.json").exists():
            log(f"{rtag}: sim exists, skip")
            ensure_journal(vtag, rtag, args.offset)
            continue
        t0 = time.time()
        ff = FS / f"{pool}.json"
        log(f"=== {vtag} -> {rtag} ===")
        ck = CACHE_DIR / f"votes__{pool}__any2_10_20_40_60_120.npz"
        if not ck.exists():
            if not run_cmd(["--factset", str(ff), "--signals", "99"]):
                continue
        if not run_cmd(["--factset", str(ff), "--build", "--tag", rtag,
                        "--offset", str(args.offset)] + BASE_PARAMS + extra):
            continue
        if not run_cmd(["--factset", str(ff), "--run", "1", "--tag", rtag,
                        "--runner", "v42", "--holding-bars", "60"], timeout=900):
            continue
        log(f"{rtag} done in {time.time()-t0:.0f}s")
        ensure_journal(vtag, rtag, args.offset)


if __name__ == "__main__":
    main()
