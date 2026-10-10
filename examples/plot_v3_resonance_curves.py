"""v3 resonance framework charts:
  01: NAV full 2010-2025 (log scale) — 4 configs + EW + HS300
  02: NAV segment 2018-2025 (base=1 @ 2017-12-29)
  03: drawdown segment 2018-2025 — 4 configs + HS300 index
Output -> evidence/v3_resonance_20261010/charts/
"""
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
OUT = AKQ / "evidence" / "v3_resonance_20261010"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)
BENCH = AKQ / "evidence" / "v2_screen_20261010" / "charts"

CONFIGS = {
    "v1 (sell P90)":        ("V14_0",   "#9e9e9e", 1.3, "--"),
    "hyst (sell P50)":      ("V3_hyst", "#8e24aa", 1.5, "-"),
    "cd3 (cooldown 3)":     ("V3_cd3",  "#2e7d32", 2.4, "-"),
    "g10 (grid 10)":        ("V3_g10",  "#f57c00", 1.5, "-"),
}


def load_nav(p):
    df = pd.read_csv(p)
    dcol = "date" if "date" in df.columns else "d"
    df = df[[dcol, "value"]].rename(columns={dcol: "date"}).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= pd.Timestamp("2025-12-31")]
    return df.sort_values("date").reset_index(drop=True)


def seg_base(df, base_ts):
    seg = df[df["date"] >= base_ts].copy()
    pre = df[df["date"] < base_ts]
    base = float(pre["value"].iloc[-1]) if len(pre) else float(seg["value"].iloc[0])
    seg["norm"] = seg["value"] / base
    return seg


def main():
    navs = {k: load_nav(OUT / "sims" / v[0] / "nav.csv") for k, v in CONFIGS.items()}
    ew = load_nav(BENCH / "ew_nav.csv")
    ix = load_nav(BENCH / "hs300_nav.csv")
    B2018 = pd.Timestamp("2018-01-01")
    B2010 = pd.Timestamp("2010-01-04")

    # ---- 01 full log ----
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    for label, (tag, color, lw, ls) in CONFIGS.items():
        seg = seg_base(navs[label], B2010)
        ax.plot(seg["date"], seg["norm"], color=color, lw=lw, ls=ls, label=label)
    for label, src, color, lw in [("EW portfolio", ew, "#1565c0", 2.0), ("HS300 index", ix, "#bdbdbd", 1.4)]:
        seg = seg_base(src, B2010)
        ax.plot(seg["date"], seg["norm"], color=color, lw=lw, label=label)
        y = seg["norm"].iloc[-1]
        ax.annotate(f"{y:.2f}x", (seg["date"].iloc[-1], y), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    for label, (tag, color, lw, ls) in CONFIGS.items():
        seg = seg_base(navs[label], B2010)
        y = seg["norm"].iloc[-1]
        ax.annotate(f"{y:.2f}x", (seg["date"].iloc[-1], y), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
    ax.set_title("v3 resonance framework - NAV full 2010-2025 (log scale, base = 1.0 @ 2010-01-04)", fontsize=13)
    ax.set_ylabel("normalized NAV (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "01_nav_full_2010_2025_log.png")
    plt.close(fig)

    # ---- 02 segment ----
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    for label, (tag, color, lw, ls) in CONFIGS.items():
        seg = seg_base(navs[label], B2018)
        ax.plot(seg["date"], seg["norm"], color=color, lw=lw, ls=ls, label=label)
        y = seg["norm"].iloc[-1]
        ax.annotate(f"{y:.2f}x", (seg["date"].iloc[-1], y), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    for label, src, color, lw in [("EW portfolio", ew, "#1565c0", 2.0), ("HS300 index", ix, "#bdbdbd", 1.4)]:
        seg = seg_base(src, B2018)
        ax.plot(seg["date"], seg["norm"], color=color, lw=lw, label=label)
        y = seg["norm"].iloc[-1]
        ax.annotate(f"{y:.2f}x", (seg["date"].iloc[-1], y), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    ax.axhline(1.0, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_title("v3 resonance framework - NAV segment 2018-2025 (base = 1.0 @ 2017-12-29)", fontsize=13)
    ax.set_ylabel("normalized NAV")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "02_nav_segment_2018_2025.png")
    plt.close(fig)

    # ---- 03 drawdown segment ----
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    dd_sets = list(CONFIGS.items()) + [("HS300 index", (None, "#bdbdbd", 1.4, "-"))]
    for label, (tag, color, lw, ls) in dd_sets:
        src = ix if label == "HS300 index" else navs[label]
        seg = seg_base(src, B2018)
        v = seg["norm"].to_numpy()
        peak = np.maximum.accumulate(v)
        dd = v / peak - 1.0
        ax.plot(seg["date"], dd * 100, color=color, lw=lw, ls=ls, label=label)
        j = int(np.argmin(dd))
        ax.annotate(f"{dd.min()*100:.1f}%", (seg["date"].iloc[j], dd.min() * 100),
                    xytext=(0, -12), textcoords="offset points", fontsize=9, color=color, ha="center")
    ax.set_title("v3 resonance framework - drawdown segment 2018-2025", fontsize=13)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(CH / "03_drawdown_segment_2018_2025.png")
    plt.close(fig)

    for f in sorted(CH.glob("*.png")):
        print(f)
    print("done")


if __name__ == "__main__":
    main()
