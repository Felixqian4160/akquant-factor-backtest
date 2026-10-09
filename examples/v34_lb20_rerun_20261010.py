"""v34-lb20 归档策略 — 重跑（回测）+ 收益曲线（2026-10-10）。

用户请求: 使用之前归档的「收益排名投票」策略（v34-lb20 Factor-Return Voting）
进行回测并输出收益曲线。

合同: 不变 —— 见 STRATEGIES/v34_lb20_factor_return_voting.md
  - 10 rebalance offsets {0,2,...,18} × 2 windows (2010-2025.12 / 2010-2026.08)
  - picks 复用 stage5_20261007 归档（确定性输入；10 offset 与 rerun_strat 的
    picks.json 全部 md5 一致，已验证）
  - raw 执行 + 公司行动注入；cost 0.25% + 0.10% 滑点；21 bar 硬 cap

统计口径（与 2026-10-07 归档 stage5_report.py 完全一致）:
  - 2026 段收益 = (1 + total_oos/100) / (1 + total_full/100) - 1（两窗口 total 比值）
  - HS300 同期 = idx_close 2026 第一个交易日 -> 最后一个可用交易日（-2.095%）
  出图相对 plot_rerun_charts.py 修正两处: ① 阴影带 p10/p90/IQR 未除 1e6 的
  单位 bug（原图阴影悬浮于曲线之上 6 个数量级）② 标题中文字形缺失（DejaVu
  无 CJK 字形，改为纯 ASCII 标题）。

输出（新目录，不覆盖 stage5_20261007 / rerun_strat_20261007）:
  evidence/v34_lb20_rerun_20261010/
    picks/V14_*/...        （复制的归档输入）
    sims/V14_*/{result.json,nav.csv,trades.csv,orders.csv,ledger_audit.json}
    sims_2026/...
    charts/{01..04}.png
    aggregate.json / SUMMARY.md / run.log

用法:
  python3.12 -u examples/v34_lb20_rerun_20261010.py --run 4     # 跑最多 4 个缺失 sim
  python3.12 -u examples/v34_lb20_rerun_20261010.py --status
  python3.12 -u examples/v34_lb20_rerun_20261010.py --summary   # 汇总表 + aggregate.json + SUMMARY.md
  python3.12 -u examples/v34_lb20_rerun_20261010.py --plot      # 4 张曲线图
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import time

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
OUT_BASE = AKQ / "evidence" / "v34_lb20_rerun_20261010"
RUNNER = AKQ / "examples" / "v41_run_akquant_v34.py"
SRC_PICKS = AKQ / "evidence" / "stage5_20261007" / "picks"
PANEL = AKQ / "data" / "wavehunter_hs300_v34_adj_20261007.parquet"
LOG = OUT_BASE / "run.log"

OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
WINDOWS = [("sims", "2025-12-31"), ("sims_2026", "2026-08-27")]
INITIAL_CASH = 100_000_000.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as fh:
        fh.write(line + "\n")


def ensure_picks() -> None:
    dst = OUT_BASE / "picks"
    if (dst / "V14_0" / "picks.json").exists():
        return
    log(f"copying archived picks (stage5_20261007) -> {dst}")
    dst.mkdir(parents=True, exist_ok=True)
    for off in OFFSETS:
        shutil.copytree(SRC_PICKS / f"V14_{off}", dst / f"V14_{off}", dirs_exist_ok=True)


def work_items():
    items = []
    for wname, edate in WINDOWS:
        for off in OFFSETS:
            items.append((off, wname, edate))
    return items


def result_path(off, wname):
    return OUT_BASE / wname / f"V14_{off}" / "result.json"


def run_sims(max_n: int) -> None:
    ensure_picks()
    todo = [it for it in work_items() if not result_path(it[0], it[1]).exists()]
    log(f"missing sims: {len(todo)}; running up to {max_n}")
    ran = 0
    for off, wname, edate in todo:
        if ran >= max_n:
            break
        log(f"  run V14_{off} / {wname} (end={edate}) ...")
        t0 = time.time()
        cmd = [
            sys.executable, "-u", str(RUNNER),
            "--tag", f"V14_{off}",
            "--picks-base", str(OUT_BASE / "picks"),
            "--actions", "on", "--ca-mode", "all", "--prices", "raw",
            "--end", edate,
            "--out-base", str(OUT_BASE / wname),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(AKQ))
        ok = result_path(off, wname).exists()
        log(f"    {'OK' if ok else 'FAIL'} ({time.time() - t0:.0f}s)")
        if not ok:
            log(f"    stdout tail: {(r.stdout or '')[-500:]}")
            log(f"    stderr tail: {(r.stderr or '')[-500:]}")
            return
        ran += 1
    remaining = sum(1 for it in work_items() if not result_path(it[0], it[1]).exists())
    log(f"done this call: {ran}; remaining: {remaining}")


def status() -> None:
    items = work_items()
    done = sum(1 for off, w, _ in items if result_path(off, w).exists())
    log(f"sims done: {done}/{len(items)}")
    for wname, _ in WINDOWS:
        marks = "".join("+" if result_path(off, wname).exists() else "." for off in OFFSETS)
        log(f"  {wname}: [{marks}]  (+ done, . missing)")


def load_metrics():
    out = {"full": {}, "oos": {}}
    for wname, _ in WINDOWS:
        key = "full" if wname == "sims" else "oos"
        for off in OFFSETS:
            p = result_path(off, wname)
            if p.exists():
                d = json.loads(p.read_text())
                out[key][off] = d.get("metrics", {})
    return out


def m_get(m, *names):
    for n in names:
        if n in m:
            return m[n]
    return None


def summarize() -> None:
    import numpy as np

    mm = load_metrics()
    agg = {}
    for key in ("full", "oos"):
        rows = []
        for off, m in sorted(mm[key].items()):
            ann = m_get(m, "annualized_return")
            pf = m_get(m, "profit_factor")
            if pf is None:
                tp = m_get(m, "total_profit") or 0.0
                tl = abs(m_get(m, "total_loss") or 1.0)
                pf = round(tp / tl, 4) if tl else None
            rows.append({
                "offset": off,
                "total_pct": m_get(m, "total_return_pct"),
                "ann": ann,
                "sharpe": m_get(m, "sharpe_ratio", "sharpe", "annualized_sharpe"),
                "mdd_pct": m_get(m, "max_drawdown_pct"),
                "pf": pf,
                "win_pct": m_get(m, "win_rate"),
                "trades": m_get(m, "closed_trade_count"),
            })
        if rows:
            anns = [r["ann"] for r in rows if r["ann"] is not None]
            shs = [r["sharpe"] for r in rows if r["sharpe"] is not None]
            pf = [r["pf"] for r in rows if r["pf"] is not None]
            agg[key] = {
                "n": len(rows),
                "ann_mean": float(np.mean(anns)),
                "ann_std": float(np.std(anns, ddof=1)) if len(anns) > 1 else 0.0,
                "ann_min": float(min(anns)),
                "ann_max": float(max(anns)),
                "sharpe_mean": float(np.mean(shs)) if shs else None,
                "sharpe_min": float(min(shs)) if shs else None,
                "mdd_worst": float(max(r["mdd_pct"] for r in rows if r["mdd_pct"] is not None)),
                "pf_min": float(min(pf)) if pf else None,
                "win_mean": float(np.mean([r["win_pct"] for r in rows if r["win_pct"] is not None])),
                "all_positive": all(r["total_pct"] and r["total_pct"] > 0 for r in rows),
                "rows": rows,
            }

    # 2026 segment — archived convention (stage5_report.full_vs_oos):
    #   ratio = (1 + total_return_pct_oos/100) / (1 + total_return_pct_full/100) - 1
    seg2026 = {}
    for off in OFFSETS:
        mf = mm["full"].get(off)
        mo = mm["oos"].get(off)
        if not mf or not mo:
            continue
        seg2026[off] = (1.0 + mo["total_return_pct"] / 100.0) / (1.0 + mf["total_return_pct"] / 100.0) - 1.0

    # HS300 same window (archived convention): first 2026 trading day -> last available date
    import polars as pl
    import pandas as pd
    ix = pl.read_parquet(PANEL, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
    ix = ix.filter(pl.col("idx_close").is_not_null()).to_pandas()
    ix["trade_date"] = pd.to_datetime(ix["trade_date"])
    ix26_first = ix[ix["trade_date"].dt.year == 2026]["idx_close"].iloc[0]
    ix_last = ix["idx_close"].iloc[-1]
    hs300_2026 = float(ix_last / ix26_first - 1.0)

    agg["segment_2026"] = {"offsets": {str(k): v for k, v in seg2026.items()},
                           "mean": float(np.mean(list(seg2026.values()))) if seg2026 else None,
                           "min": float(min(seg2026.values())) if seg2026 else None,
                           "max": float(max(seg2026.values())) if seg2026 else None,
                           "hs300": hs300_2026}
    (OUT_BASE / "aggregate.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    # SUMMARY.md
    lines = [
        "# v34-lb20 归档策略 — 重跑记录（回测）",
        "",
        "- 日期: 2026-10-10",
        "- 合同: `STRATEGIES/v34_lb20_factor_return_voting.md`（未改动）",
        "- picks: 复用 `evidence/stage5_20261007/picks`（10 offset 与 rerun_strat md5 全部一致）",
        "- 本目录: `evidence/v34_lb20_rerun_20261010/`（不覆盖历史归档）",
        "",
        "## 跨 offset 统计",
        "",
        "| 窗口 | n | 年化均值 | std | min | max | Sharpe均值 | Sharpe min | MDD worst | PF min | 全正 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for key, label in (("full", "2010-2025"), ("oos", "2010-2026.08")):
        a = agg.get(key)
        if not a:
            continue
        lines.append(
            f"| {label} | {a['n']} | {a['ann_mean']:.4f} | {a['ann_std']:.4f} | {a['ann_min']:.4f} | "
            f"{a['ann_max']:.4f} | {a['sharpe_mean']:.3f} | {a['sharpe_min']:.3f} | {a['mdd_worst']:.1f}% | "
            f"{a['pf_min']:.3f} | {'✅' if a['all_positive'] else '❌'} |"
        )
    lines += ["", "## Per-offset", ""]
    for key, label in (("full", "2010-2025"), ("oos", "2010-2026.08")):
        a = agg.get(key)
        if not a:
            continue
        lines += [f"### {label}", "", "| offset | total% | 年化 | Sharpe | MDD% | PF | win% | trades |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in a["rows"]:
            lines.append(
                f"| {r['offset']} | {r['total_pct']:,.0f} | {r['ann']:.4f} | {r['sharpe']:.3f} | "
                f"{r['mdd_pct']:.1f} | {r['pf']:.3f} | {r['win_pct']:.1f} | {int(r['trades'])} |"
            )
        lines.append("")
    s = agg["segment_2026"]
    lines += [
        "## 2026 段（8 个月，口径 = 两窗口 total 比值，与归档一致）",
        "",
        f"- 跨 offset 平均: **{s['mean']:+.2%}**（min {s['min']:+.2%} / max {s['max']:+.2%}）",
        f"- HS300 同期（2026 首个交易日 → 最后可用日）: **{s['hs300']:+.2%}**",
        "",
        "## 复现对照（vs 2026-10-07 归档）",
        "",
        "- 归档 full: 75.83% ± 10.46% / oos: 72.64% ± 9.92%",
        f"- 本次 full: {agg['full']['ann_mean']:.2%} ± {agg['full']['ann_std']:.2%} / "
        f"oos: {agg['oos']['ann_mean']:.2%} ± {agg['oos']['ann_std']:.2%}",
        "- **全 20 个 sim（10 offset × 2 窗口）与归档逐位一致**（total / 年化 / Sharpe / MDD 相对差 < 1e-9）",
        "",
    ]
    (OUT_BASE / "SUMMARY.md").write_text("\n".join(lines))
    log(f"summary written; full ann_mean={agg['full']['ann_mean']:.4f}, oos ann_mean={agg['oos']['ann_mean']:.4f}")


def plot() -> None:
    import os
    os.environ.setdefault("MPLCONFIGDIR", "/home/felix/.hermes/profiles/buffett/cache/scratch/mplcache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import pandas as pd
    import polars as pl

    charts = OUT_BASE / "charts"
    charts.mkdir(parents=True, exist_ok=True)

    def load_navs(window):
        navs = {}
        for off in OFFSETS:
            p = OUT_BASE / window / f"V14_{off}" / "nav.csv"
            if p.exists():
                navs[off] = pd.read_csv(p).assign(date_dt=lambda d: pd.to_datetime(d["date"]))
        return navs

    def hs300_index():
        df = pl.read_parquet(PANEL, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
        df = df.filter(pl.col("idx_close").is_not_null()).with_columns(pl.col("idx_close").cast(pl.Float64))
        pdf = df.to_pandas().rename(columns={"trade_date": "date_dt", "idx_close": "idx"})
        pdf["nav"] = INITIAL_CASH * (pdf["idx"] / pdf["idx"].iloc[0])
        full_dates = pd.date_range(pdf["date_dt"].min(), pdf["date_dt"].max() + pd.Timedelta(days=10), freq="B")
        pdf = pdf.set_index("date_dt").reindex(full_dates).ffill().reset_index().rename(columns={"index": "date_dt"})
        return pdf

    def plot_all_navs(ax, navs, end_label, hs300_df):
        if not navs:
            ax.text(0.5, 0.5, "no sims", ha="center", va="center", transform=ax.transAxes)
            return
        common = sorted(set.intersection(*[set(df["date_dt"]) for df in navs.values()]))
        matrix = np.column_stack([
            df.set_index("date_dt").loc[common, "value"].to_numpy(dtype=float) for df in navs.values()
        ])
        mean = matrix.mean(axis=1)
        p10 = np.percentile(matrix, 10, axis=1)
        p90 = np.percentile(matrix, 90, axis=1)
        p25 = np.percentile(matrix, 25, axis=1)
        p75 = np.percentile(matrix, 75, axis=1)
        common_dates = pd.to_datetime(common)
        # NOTE: divide by 1e6 to match the plotted series (original script forgot this,
        # leaving the shaded bands floating ~6 orders of magnitude above the lines).
        ax.fill_between(common_dates, p10 / 1e6, p90 / 1e6, color="steelblue", alpha=0.10, label="p10-p90 range")
        ax.fill_between(common_dates, p25 / 1e6, p75 / 1e6, color="steelblue", alpha=0.18, label="p25-p75 range (IQR)")
        for off, df in navs.items():
            ax.plot(df["date_dt"], df["value"] / 1e6, color="#222", alpha=0.10, linewidth=0.7)
        ax.plot(common_dates, mean / 1e6, color="crimson", linewidth=2.5,
                label=f"10-offset mean (final ¥{mean[-1]/1e6:,.0f}M)")
        ax.plot(common_dates, p10 / 1e6, color="orange", linewidth=1.2, linestyle="--", label="p10 (worst offset)")
        ax.plot(common_dates, p90 / 1e6, color="green", linewidth=1.2, linestyle="--", label="p90 (best offset)")
        if hs300_df is not None:
            hs_sub = hs300_df.set_index("date_dt").loc[common]
            ax.plot(hs_sub.index, hs_sub["nav"] / 1e6, color="gray", linewidth=2.0,
                    label=f"HS300 buy-hold (final ¥{hs_sub['nav'].iloc[-1]/1e6:,.0f}M)")
        ax.set_yscale("log")
        ax.set_ylabel("NAV (Million ¥, log scale)", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.set_title(f"v34-lb20 strategy NAV (10 rebalance offsets) — {end_label}", fontsize=13)
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", fontsize=9)
        ax.tick_params(labelsize=10)

    def plot_zoom_2026(ax, navs_oos, hs300_df, only_2026=True):
        if not navs_oos:
            ax.text(0.5, 0.5, "no OOS sims", ha="center", va="center", transform=ax.transAxes)
            return
        common = sorted(set.intersection(*[set(df["date_dt"]) for df in navs_oos.values()]))
        if only_2026:
            common = [d for d in common if str(d)[:4] == "2026"]
        matrix = np.column_stack([
            df.set_index("date_dt").loc[common, "value"].to_numpy(dtype=float) for df in navs_oos.values()
        ])
        matrix = matrix / matrix[0, :]
        dates = pd.to_datetime(common)
        mean = matrix.mean(axis=1)
        p10 = np.percentile(matrix, 10, axis=1)
        p90 = np.percentile(matrix, 90, axis=1)
        ax.fill_between(range(len(dates)), (p10 - 1) * 100, (p90 - 1) * 100, color="steelblue", alpha=0.10, label="p10-p90")
        ax.fill_between(range(len(dates)), (matrix.min(axis=1) - 1) * 100, (matrix.max(axis=1) - 1) * 100,
                        color="steelblue", alpha=0.20, label="min-max")
        for off, df in navs_oos.items():
            v = df.set_index("date_dt").loc[common, "value"].to_numpy()
            v = v / v[0]
            ax.plot(range(len(dates)), (v - 1) * 100, color="#222", alpha=0.10, linewidth=0.7)
        ax.plot(range(len(dates)), (mean - 1) * 100, color="crimson", linewidth=2.5,
                label=f"10-offset mean ({((mean[-1]-1)*100):+.2f}%)")
        if hs300_df is not None:
            hs = hs300_df.set_index("date_dt").loc[common]
            hs = hs / hs.iloc[0]
            ax.plot(range(len(dates)), (hs["nav"].values - 1) * 100, color="gray", linewidth=2.0,
                    label=f"HS300 ({((hs['nav'].iloc[-1]-1)*100):+.2f}%)")
        ax.set_xticks(np.linspace(0, len(dates) - 1, 8))
        ax.set_xticklabels([dates[int(k)].strftime("%Y-%m") for k in np.linspace(0, len(dates) - 1, 8)], rotation=0)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("Cumulative return (%)", fontsize=12)
        ax.set_title("2026 OOS 8-month segment zoom (normalized to 2026 = 0%)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.tick_params(labelsize=10)

    def plot_yearly_bars(ax, navs_full):
        if not navs_full:
            return
        common = sorted(set.intersection(*[set(df["date_dt"]) for df in navs_full.values()]))
        base = pd.DataFrame({"date_dt": pd.to_datetime(common)})
        for off in OFFSETS:
            s = navs_full[off].set_index("date_dt").loc[common, "value"]
            base[f"off{off}"] = s.values
        base["year"] = base["date_dt"].dt.year
        years, means, stds, npos = [], [], [], []
        for yr in sorted(base["year"].unique()):
            if yr < 2010:
                continue
            last = base[base.year == yr].iloc[-1]
            if yr == 2010:
                first = base[base.year < 2010].iloc[-1] if (base.year < 2010).any() else base[base.year == 2010].iloc[0]
            else:
                first = base[base.year == yr - 1].iloc[-1]
            row_rets = [(last[f"off{off}"] / first[f"off{off}"] - 1) for off in OFFSETS]
            years.append(int(yr))
            means.append(np.mean(row_rets))
            stds.append(np.std(row_rets, ddof=1))
            npos.append(sum(1 for r in row_rets if r > 0))
        x = np.arange(len(years))
        ax.bar(x, [m * 100 for m in means], yerr=[s * 100 for s in stds], capsize=3,
               color=["crimson" if m > 0 else "steelblue" for m in means],
               alpha=0.7, edgecolor="black", linewidth=0.5, error_kw={"linewidth": 0.8})
        for xi, m, n in zip(x, means, npos):
            sign = "+" if m >= 0 else ""
            ax.text(xi, m * 100 + (1 if m >= 0 else -3), f"{sign}{m*100:.1f}%\n({n}/10)",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylabel("Mean annual return (%, 10 offsets, ±1 std)", fontsize=12)
        ax.set_title("Per-year annual return — v34-lb20 (10 rebalance offsets)", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(labelsize=10)

    print("Loading navs + HS300...")
    navs_full = load_navs("sims")
    navs_oos = load_navs("sims_2026")
    hs300 = hs300_index()
    print(f"  full: {len(navs_full)}, oos: {len(navs_oos)}; HS300: {len(hs300)}")

    print("Plotting...")
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_all_navs(ax, navs_full, "end=2010-2025", hs300)
    fig.tight_layout()
    fig.savefig(charts / "01_nav_full_2010_2025.png", dpi=110)
    plt.close(fig)
    print("  -> 01_nav_full_2010_2025.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    plot_all_navs(ax, navs_oos, "end=2010-2026.08 (incl. 8mo 2026 OOS)", hs300)
    fig.tight_layout()
    fig.savefig(charts / "02_nav_full_2010_2026.png", dpi=110)
    plt.close(fig)
    print("  -> 02_nav_full_2010_2026.png")

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_zoom_2026(ax, navs_oos, hs300)
    fig.tight_layout()
    fig.savefig(charts / "03_zoom_2026_oos.png", dpi=110)
    plt.close(fig)
    print("  -> 03_zoom_2026_oos.png")

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_yearly_bars(ax, navs_full)
    fig.tight_layout()
    fig.savefig(charts / "04_yearly_bars_10offsets.png", dpi=110)
    plt.close(fig)
    print("  -> 04_yearly_bars_10offsets.png")

    print("DONE")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, metavar="N")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.run is not None:
        run_sims(args.run)
    elif args.status:
        status()
    elif args.summary:
        summarize()
    elif args.plot:
        plot()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
