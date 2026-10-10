"""v3 multiphase NAV charts — STANDARD tool (round 2).

每个 config = 一组相位 tag；曲线 = 各相位归一化 NAV 的平均（phase-mean），
cap40/cd3 附带 min-max 相位带。输出 -> evidence/v3_resonance_20261010/charts/
用法: 直接运行（config 列表在 CONFIGS 中维护）。
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
S3 = OUT / "sims"
S2 = AKQ / "evidence" / "v2_screen_20261010" / "sims"
BENCH = AKQ / "evidence" / "v2_screen_20261010" / "charts"
END_TS = pd.Timestamp("2025-12-31")

# label -> (phase tags, color, lw, ls, draw_band)
CONFIGS = {
    "cap40 (optimized)":   ([f"V3_cd3_cap40_o{o}" for o in range(5)], "#1e88e5", 2.6, "-", True),
    "cap60":               ([f"V3_cd3_cap60_o{o}" for o in range(5)], "#64b5f6", 1.5, "--", False),
    "cd3 (cap21)":         (["V3_cd3"] + [f"V3_cd3_o{o}" for o in range(1, 5)], "#2e7d32", 2.0, "-", True),
    "static_2017":         (["static_2017", "static_2017_o4", "static_2017_o8",
                             "static_2017_o12", "static_2017_o16"], "#00897b", 1.8, "-", False),
}


def nav_path(tag):
    return (S2 / tag / "V14_0" / "nav.csv") if tag.startswith("static_2017") else (S3 / tag / "nav.csv")


def load_series(path):
    df = pd.read_csv(path)
    dcol = "date" if "date" in df.columns else "d"
    df = df[[dcol, "value"]].rename(columns={dcol: "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END_TS].sort_values("date")
    return df.set_index("date")["value"]


def mean_curve(tags, base_ts):
    frames = []
    for t in tags:
        s = load_series(nav_path(t))
        pre = s[s.index < base_ts]
        b = float(pre.iloc[-1]) if len(pre) else float(s.iloc[0])
        frames.append((s[s.index >= base_ts] / b).rename(t))
    joined = pd.concat(frames, axis=1).dropna()
    return joined.mean(axis=1), joined.min(axis=1), joined.max(axis=1)


def main():
    B2018 = pd.Timestamp("2018-01-01")
    B2010 = pd.Timestamp("2010-01-04")
    ew = load_series(BENCH / "ew_nav.csv")
    ix = load_series(BENCH / "hs300_nav.csv")

    # ---------- 01 segment ----------
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    for label, (tags, color, lw, ls, band) in CONFIGS.items():
        m, lo, hi = mean_curve(tags, B2018)
        ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label)
        if band:
            ax.fill_between(m.index, lo.values, hi.values, color=color, alpha=0.10)
        dy = -9 if label == "cap60" else 0
        ax.annotate(f"{m.iloc[-1]:.2f}x", (m.index[-1], m.iloc[-1]), xytext=(4, dy),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    for label, src, color, lw in [("EW portfolio", ew, "#5c6bc0", 1.6), ("HS300 index", ix, "#bdbdbd", 1.3)]:
        seg = src[src.index >= B2018]
        seg = seg / float(src[src.index < B2018].iloc[-1])
        ax.plot(seg.index, seg.values, color=color, lw=lw, label=label)
        ax.annotate(f"{seg.iloc[-1]:.2f}x", (seg.index[-1], seg.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    ax.axhline(1.0, color="#eeeeee", lw=0.8, zorder=0)
    ax.set_xlim(pd.Timestamp("2017-12-20"), pd.Timestamp("2026-02-28"))
    ax.set_title("v3 resonance - NAV segment 2018-2025 (phase-mean; base = 1.0 @ 2017-12-29)", fontsize=13)
    ax.set_ylabel("normalized NAV")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "04_round2_segment_2018_2025.png")
    plt.close(fig)

    # ---------- 02 full log ----------
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    for label, (tags, color, lw, ls, band) in CONFIGS.items():
        m, lo, hi = mean_curve(tags, B2010)
        ax.plot(m.index, m.values, color=color, lw=lw, ls=ls, label=label)
        ax.annotate(f"{m.iloc[-1]:.2f}x", (m.index[-1], m.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=color, va="center")
    for label, src, color, lw in [("EW portfolio", ew, "#5c6bc0", 1.6), ("HS300 index", ix, "#bdbdbd", 1.3)]:
        pre = src[src.index < B2010]
        base = float(pre.iloc[-1]) if len(pre) else float(src.iloc[0])
        seg = src[src.index >= B2010] / base
        ax.plot(seg.index, seg.values, color=color, lw=lw, label=label)
    ax.set_yscale("log")
    ax.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2026-02-28"))
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x"])
    ax.set_title("v3 resonance - NAV full 2010-2025 (phase-mean, log scale, base = 1.0 @ 2010-01-04)", fontsize=13)
    ax.set_ylabel("normalized NAV (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "05_round2_full_2010_2025_log.png")
    plt.close(fig)

    # ---------- 03 drawdown ----------
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    dd_sets = [(k, v) for k, v in CONFIGS.items() if k in ("cap40 (optimized)", "cap60", "cd3 (cap21)", "static_2017")]
    for label, (tags, color, lw, ls, band) in dd_sets:
        m, _, _ = mean_curve(tags, B2018)
        v = m.to_numpy()
        peak = np.maximum.accumulate(np.concatenate(([1.0], v)))[1:]
        dd = v / peak - 1.0
        ax.plot(m.index, dd * 100, color=color, lw=lw, label=label)
        j = int(np.argmin(dd))
        ax.annotate(f"{dd.min()*100:.1f}%", (m.index[j], dd.min() * 100), xytext=(0, -12),
                    textcoords="offset points", fontsize=9, color=color, ha="center")
    seg = ix[ix.index >= B2018]; seg = seg / float(ix[ix.index < B2018].iloc[-1])
    v = seg.to_numpy(); peak = np.maximum.accumulate(np.concatenate(([1.0], v)))[1:]
    dd = v / peak - 1.0
    ax.plot(seg.index, dd * 100, color="#bdbdbd", lw=1.3, label="HS300 index")
    ax.annotate(f"{dd.min()*100:.1f}%", (seg.index[int(np.argmin(dd))], dd.min() * 100), xytext=(0, -12),
                textcoords="offset points", fontsize=9, color="#9e9e9e", ha="center")
    ax.set_xlim(pd.Timestamp("2018-01-01"), pd.Timestamp("2026-02-28"))
    ax.set_title("v3 resonance - drawdown segment 2018-2025 (phase-mean)", fontsize=13)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(CH / "06_round2_drawdown_segment.png")
    plt.close(fig)

    # ---------- save phase-mean CSVs ----------
    for label, (tags, *_rest) in CONFIGS.items():
        m, lo, hi = mean_curve(tags, B2010)
        out = pd.DataFrame({"date": m.index, "mean": m.values, "min": lo.values, "max": hi.values})
        slug = label.split(" ")[0]
        out.to_csv(CH / f"phase_mean_{slug}.csv", index=False)
    for f in sorted(CH.glob("0[456]_round2*.png")):
        print(f)
    print("done")


if __name__ == "__main__":
    main()
