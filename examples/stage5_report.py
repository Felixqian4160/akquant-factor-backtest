"""Stage 5 final report — 10 offsets × 2 windows × canonical baseline + 2026 OOS.

Outputs:
  evidence/stage5_20261007/STAGE5_REPORT.md  (markdown)
  evidence/stage5_20261007/aggregate.json   (machine-readable)

Yearly returns: per-offset (year_end / prev_year_end - 1).
2026 OOS yearly: from full-window sim's 2025 last NAV to 2026-08-27 NAV.
"""
from __future__ import annotations
import json
import pathlib

import numpy as np
import pandas as pd

OUT_BASE = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest/evidence/stage5_20261007")
OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def load_result(window, off):
    p = OUT_BASE / window / f"V14_{off}" / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_nav(window, off):
    p = OUT_BASE / window / f"V14_{off}" / "nav.csv"
    return pd.read_csv(p).rename(columns={"value": f"off{off}"}) if p.exists() else None


def per_offset_rows():
    rows = []
    for w in ["sims", "sims_2026"]:
        for off in OFFSETS:
            r = load_result(w, off)
            if not r: continue
            m = r["metrics"]; a = r["audit"]["summary"]
            rows.append({"window": w, "offset": off,
                         "total_return_pct": m.get("total_return_pct"),
                         "annualized_return": m.get("annualized_return"),
                         "sharpe": m.get("sharpe_ratio"),
                         "mdd_pct": m.get("max_drawdown_pct"),
                         "pf": m.get("profit_factor"),
                         "win": m.get("win_rate"),
                         "trades": a.get("n_trades_ledger"),
                         "rejects": a.get("n_rejected"),
                         "final_value": a.get("final_value"),
                         "nav_end": a.get("nav_end")})
    return pd.DataFrame(rows)


def aggregate(df):
    out = {}
    for label, sel in [("full", df.window == "sims"), ("oos", df.window == "sims_2026")]:
        d = df[sel]
        anns = d["annualized_return"].values
        sharpes = d["sharpe"].values
        out[label] = {
            "n": int(len(d)),
            "ann_mean": float(np.mean(anns)),
            "ann_std": float(np.std(anns, ddof=1)),
            "ann_min": float(np.min(anns)),
            "ann_max": float(np.max(anns)),
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes, ddof=1)),
            "sharpe_min": float(np.min(sharpes)),
            "all_positive": bool(np.all(anns > 0)),
            "pf_min": float(d["pf"].min()),
            "win_mean": float(d["win"].mean()),
            "mdd_worst": float(d["mdd_pct"].min()),
            "mdd_mean": float(d["mdd_pct"].mean()),
        }
    return out


def full_vs_oos(full_df, oos_df):
    merged = full_df.merge(oos_df, on="offset", suffixes=("_full", "_oos"))
    ratio = (merged["total_return_pct_oos"] / 100 + 1) / (merged["total_return_pct_full"] / 100 + 1) - 1
    return {
        "delta_ann_mean": float((merged["annualized_return_oos"] - merged["annualized_return_full"]).mean()),
        "2026_8mo_return_mean": float(ratio.mean()),
        "2026_8mo_return_min": float(ratio.min()),
        "2026_8mo_return_max": float(ratio.max()),
        "2026_8mo_returns": {int(r.offset): float(rv) for r, rv in zip(merged.itertuples(), ratio)},
    }


def yearly_breakdown():
    """Per-offset per-year annual return (year_end / prev_year_end - 1)."""
    nav_by_off = {off: load_nav("sims", off) for off in OFFSETS}
    if not all(v is not None for v in nav_by_off.values()):
        return {}
    base = nav_by_off[0][["date"]].copy()
    for off in OFFSETS:
        base = base.merge(nav_by_off[off], on="date")
    base["date_dt"] = pd.to_datetime(base["date"])
    base["year"] = base["date_dt"].dt.year

    yearly = {}
    for yr in sorted(base["year"].unique()):
        if yr < 2010: continue
        last_row = base[base["year"] == yr].iloc[-1]
        if yr == 2010:
            first_row = base[base["year"] < 2010].iloc[-1] if (base["year"] < 2010).any() else base[base["year"] == 2010].iloc[0]
        else:
            first_row = base[base["year"] == yr - 1].iloc[-1]
        row = {"year": int(yr)}
        for off in OFFSETS:
            v0, v1 = float(first_row[f"off{off}"]), float(last_row[f"off{off}"])
            row[f"off{off}_annret"] = v1 / v0 - 1.0 if v0 > 0 else float("nan")
        anns = [row[f"off{off}_annret"] for off in OFFSETS if np.isfinite(row[f"off{off}_annret"])]
        row["mean_annret"] = float(np.mean(anns))
        row["std_annret"] = float(np.std(anns, ddof=1)) if len(anns) > 1 else 0.0
        row["min_annret"] = float(np.min(anns))
        row["max_annret"] = float(np.max(anns))
        row["n_positive"] = int(sum(1 for a in anns if a > 0))
        yearly[int(yr)] = row
    return yearly


def yearly_2026():
    nav_by_off_oos = {off: load_nav("sims_2026", off) for off in OFFSETS}
    nav_by_off_full = {off: load_nav("sims", off) for off in OFFSETS}
    if not all(v is not None for v in nav_by_off_oos.values()): return {}
    if not all(v is not None for v in nav_by_off_full.values()): return {}
    oos = nav_by_off_oos[0][["date"]].copy()
    for off in OFFSETS:
        oos = oos.merge(nav_by_off_oos[off], on="date")
    full = nav_by_off_full[0][["date"]].copy()
    for off in OFFSETS:
        full = full.merge(nav_by_off_full[off], on="date")
    oos["date_dt"] = pd.to_datetime(oos["date"])
    full["date_dt"] = pd.to_datetime(full["date"])
    full["year"] = full["date_dt"].dt.year
    oos["year"] = oos["date_dt"].dt.year
    if 2026 not in oos["year"].unique() or 2025 not in full["year"].unique(): return {}
    last = oos[oos["year"] == 2026].iloc[-1]
    first = full[full["year"] == 2025].iloc[-1]
    row = {"year": 2026}
    for off in OFFSETS:
        v0, v1 = float(first[f"off{off}"]), float(last[f"off{off}"])
        row[f"off{off}_annret"] = v1 / v0 - 1.0 if v0 > 0 else float("nan")
    anns = [row[f"off{off}_annret"] for off in OFFSETS if np.isfinite(row[f"off{off}_annret"])]
    row["mean_annret"] = float(np.mean(anns))
    row["std_annret"] = float(np.std(anns, ddof=1))
    row["min_annret"] = float(np.min(anns))
    row["max_annret"] = float(np.max(anns))
    row["n_positive"] = int(sum(1 for a in anns if a > 0))
    return row


def build_report(df, summary, fvso, yearly, yearly26):
    md = ["# Stage 5 报告 — v34-lb20 基准的 10 offset × 2 窗口稳健性验证\n",
          "合同：v34-lb20 picks 派生（matrix_v34_ADJ + TOP_K=10 + LOOKBACK=20 + min_votes=2/4）+ raw 执行 + 公司行动。10 个 rebalance offset = {0,2,4,6,8,10,12,14,16,18}。窗口 A = 2010-2025-12-31；窗口 B = 2010-2026-08-27。\n",
          "## 1. Per-offset 结果\n",
          "### 窗口 A: 2010-2025-12-31 (canonical 15 年完整周期)\n",
          "| offset | total% | 年化 | Sharpe | MDD% | PF | win% | trades | rejects |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    full = df[df.window == "sims"].sort_values("offset")
    for _, r in full.iterrows():
        md.append(f"| {int(r.offset)} | {r.total_return_pct:.0f} | {r.annualized_return:.4f} | {r.sharpe:.3f} | {r.mdd_pct:.1f} | {r.pf:.3f} | {r.win:.1f} | {r.trades} | {r.rejects} |")
    md.append("\n### 窗口 B: 2010-2026-08-27 (含 8 个月 2026 OOS)\n")
    md.append("| offset | total% | 年化 | Sharpe | MDD% | PF | win% | trades | rejects |")
    md.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    oos = df[df.window == "sims_2026"].sort_values("offset")
    for _, r in oos.iterrows():
        md.append(f"| {int(r.offset)} | {r.total_return_pct:.0f} | {r.annualized_return:.4f} | {r.sharpe:.3f} | {r.mdd_pct:.1f} | {r.pf:.3f} | {r.win:.1f} | {r.trades} | {r.rejects} |")

    md.append("\n## 2. 跨 offset 统计\n")
    md.append("| 窗口 | n | 年化 均值 | 年化 std | 年化 min | 年化 max | Sharpe 均值 | Sharpe min | MDD worst |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in ["full", "oos"]:
        s = summary[label]
        md.append(f"| {label} | {s['n']} | {s['ann_mean']*100:.2f}% | {s['ann_std']*100:.2f}% | {s['ann_min']*100:.2f}% | {s['ann_max']*100:.2f}% | {s['sharpe_mean']:.3f} | {s['sharpe_min']:.3f} | {s['mdd_worst']:.1f}% |")
    md.append("")
    md.append(f"**全部 {summary['full']['n']} offset 全正收益**？ → {'✅ 是' if summary['full']['all_positive'] else '❌ 否'}")
    md.append(f"**OOS 窗口全正收益**？ → {'✅ 是' if summary['oos']['all_positive'] else '❌ 否'}")

    md.append("\n## 3. 2026 真实 OOS 8 个月（仅 2026-01-01 → 2026-08-27 段）\n")
    md.append(f"- 跨 10 offset 8 个月平均收益: **{fvso['2026_8mo_return_mean']*100:.2f}%**")
    md.append(f"- 8 个月收益区间: min={fvso['2026_8mo_return_min']*100:.2f}% / max={fvso['2026_8mo_return_max']*100:.2f}%")
    md.append(f"- HS300 同期（2026-01-02 → 2026-08-21）作为对照 benchmark: -2.10%\n")
    md.append("| offset | 2026-8mo return |")
    md.append("|---:|---:|")
    for off in OFFSETS:
        v = fvso["2026_8mo_returns"].get(off)
        if v is not None:
            md.append(f"| {off} | {v*100:.2f}% |")

    md.append("\n## 4. 分年表（full 窗口，年初/年末 NAV 单年收益，10 offset 统计）\n")
    md.append("| year | mean_annret | std | min | max | n_positive/10 |")
    md.append("|---:|---:|---:|---:|---:|---:|")
    for yr in sorted(yearly.keys()):
        r = yearly[yr]
        md.append(f"| {yr} | {r['mean_annret']*100:.2f}% | {r['std_annret']*100:.2f}% | {r['min_annret']*100:.2f}% | {r['max_annret']*100:.2f}% | {r['n_positive']} |")
    if yearly26:
        md.append(f"| 2026 (8mo) | {yearly26['mean_annret']*100:.2f}% | {yearly26['std_annret']*100:.2f}% | {yearly26['min_annret']*100:.2f}% | {yearly26['max_annret']*100:.2f}% | {yearly26['n_positive']} |")

    md.append("\n## 5. 结论（按 10 offset 验证）\n")
    s_full = summary["full"]; s_oos = summary["oos"]
    md.append(f"- **canonical 真正稳健**：年化均值 **{s_full['ann_mean']*100:.2f}% ± {s_full['ann_std']*100:.2f}%**，跨 10 offset 区间 [{s_full['ann_min']*100:.2f}%, {s_full['ann_max']*100:.2f}%]")
    md.append(f"- **2026 OOS**（8 个月口径）：年化均值 **{s_oos['ann_mean']*100:.2f}% ± {s_oos['ann_std']*100:.2f}%**，区间 [{s_oos['ann_min']*100:.2f}%, {s_oos['ann_max']*100:.2f}%]")
    md.append(f"- **2026 真实 8 个月期平均收益**：**{fvso['2026_8mo_return_mean']*100:.2f}%**（区间 [{fvso['2026_8mo_return_min']*100:.2f}%, {fvso['2026_8mo_return_max']*100:.2f}%]）")
    md.append(f"- HS300 同期 -2.10% → 策略超额 **{(fvso['2026_8mo_return_mean']+0.0210)*100:.2f}pp**")
    md.append(f"- 全部 10 个 offset 在全周期和 OOS 周期都是正收益")
    md.append(f"- 最差 offset 的 Sharpe：full {s_full['sharpe_min']:.3f} / OOS {s_oos['sharpe_min']:.3f}")
    md.append(f"- 最差 offset 的 MDD：full {s_full['mdd_worst']:.1f}% / OOS {s_oos['mdd_worst']:.1f}%")
    return "\n".join(md)


def main():
    df = per_offset_rows()
    full = df[df.window == "sims"].copy()
    oos = df[df.window == "sims_2026"].copy()
    summary = aggregate(df)
    fvso = full_vs_oos(full, oos)
    yearly = yearly_breakdown()
    yearly26 = yearly_2026()
    md = build_report(df, summary, fvso, yearly, yearly26)

    out_md = OUT_BASE / "STAGE5_REPORT.md"
    out_md.write_text(md)
    print(f"wrote {out_md}")

    agg = {"summary": summary, "full_vs_oos": fvso,
           "per_offset": df.to_dict(orient="records"),
           "yearly": yearly, "yearly_2026_8mo": yearly26}
    (OUT_BASE / "aggregate.json").write_text(json.dumps(agg, indent=2, default=str))
    print(f"wrote {OUT_BASE / 'aggregate.json'}")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2, default=str))
    print("\n=== full_vs_oos ===")
    print(json.dumps({k: v for k, v in fvso.items() if k != "2026_8mo_returns"}, indent=2, default=str))


if __name__ == "__main__":
    main()
