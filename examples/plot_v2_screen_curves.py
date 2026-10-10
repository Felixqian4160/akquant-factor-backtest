"""v2 screen 收益曲线图:
  01: NAV segment 2018-2025 (base=1 @ 2017-12-29) — static_2017 vs static vs roll250 vs k120 vs EW vs HS300
  02: drawdown segment 2018-2025 — static_2017 / static / EW / k120
  03: NAV full 2010-2025 (log scale)
输出到 evidence/v2_screen_20261010/charts/
"""
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AKQ = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
OUT = AKQ / "evidence" / "v2_screen_20261010"
CH = OUT / "charts"
CH.mkdir(parents=True, exist_ok=True)

ARMS = {
    "static_2017 (honest set)": ("static_2017", "#2e7d32", 2.2, "-"),
    "static (in-sample set)": ("static", "#9e9e9e", 1.4, "--"),
    "static_roll250": ("static_roll250", "#f57c00", 1.6, "-"),
    "k120 (dynamic best)": ("k120", "#6a1b9a", 1.6, "-"),
}


def load_nav(arm):
    df = pd.read_csv(OUT / "sims" / arm / "V14_0" / "nav.csv")
    cols = list(df.columns)
    vcol = "value" if "value" in cols else cols[-1]
    df["d"] = pd.to_datetime(df["date"])
    return df[["d", vcol]].rename(columns={vcol: "value"}).sort_values("d").reset_index(drop=True)


def load_benchmarks():
    pan = pl.read_parquet(AKQ / "data" / "wavehunter_hs300_v34_dedup_20261010.parquet",
                          columns=["trade_date", "ts_code", "adj_close", "idx_close"])
    pan = pan.filter(pl.col("trade_date") >= pl.datetime(2010, 1, 1))
    pan = pan.with_columns(pl.col("trade_date").dt.date().alias("d")).drop("trade_date")
    df = pan.sort(["ts_code", "d"]).to_pandas()
    df["r"] = df.groupby("ts_code")["adj_close"].pct_change()
    ewd = df.dropna(subset=["r"]).groupby("d")["r"].mean().sort_index()
    ew = pd.DataFrame({"d": pd.to_datetime(ewd.index), "value": np.cumprod(1.0 + ewd.values)})
    ix = pan.select(["d", "idx_close"]).unique().drop_nulls().sort("d").to_pandas()
    ix["d"] = pd.to_datetime(ix["d"])
    ix = ix[["d", "idx_close"]].rename(columns={"idx_close": "value"}).reset_index(drop=True)
    ew.to_csv(CH / "ew_nav.csv", index=False)
    ix.to_csv(CH / "hs300_nav.csv", index=False)
    ew = ew[ew["d"] <= pd.Timestamp("2025-12-31")]
    ix = ix[ix["d"] <= pd.Timestamp("2025-12-31")]
    return ew, ix


def seg_base(df, base_ts):
    seg = df[df["d"] >= base_ts].copy()
    pre = df[df["d"] < base_ts]
    base = float(pre["value"].iloc[-1]) if len(pre) else float(seg["value"].iloc[0])
    seg["norm"] = seg["value"] / base
    return seg


def main():
    navs = {k: load_nav(v[0]) for k, v in ARMS.items()}
    ew, ix = load_benchmarks()
    B2018 = pd.Timestamp("2018-01-01")
    B2010 = pd.Timestamp("2010-01-04")

    # ---- chart 01: segment NAV ----
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    ends = {}
    for label, (arm, color, lw, ls) in ARMS.items():
        seg = seg_base(navs[label], B2018)
        ax.plot(seg["d"], seg["norm"], color=color, lw=lw, ls=ls, label=label)
        ends[label] = (seg["d"].iloc[-1], seg["norm"].iloc[-1])
    for label, color, lw in [("EW portfolio (hold all)", "#1565c0", 2.2), ("HS300 index", "#bdbdbd", 1.4)]:
        src = ew if "EW" in label else ix
        seg = seg_base(src, B2018)
        ax.plot(seg["d"], seg["norm"], color=color, lw=lw, label=label)
        ends[label] = (seg["d"].iloc[-1], seg["norm"].iloc[-1])
    for label, (x, y) in ends.items():
        c = ARMS[label][1] if label in ARMS else ("#1565c0" if "EW" in label else "#bdbdbd")
        ax.annotate(f"{y:.2f}x", (x, y), xytext=(4, 0), textcoords="offset points",
                    fontsize=9, color=c, va="center")
    ax.set_title("v2 screen - NAV segment 2018-2025 (base = 1.0 @ 2017-12-29)", fontsize=13)
    ax.set_ylabel("normalized NAV")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "01_nav_segment_2018_2025.png")
    plt.close(fig)

    # ---- chart 02: drawdown segment ----
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    dd_sets = [("static_2017 (honest set)", navs["static_2017 (honest set)"], "#2e7d32", 2.0),
               ("static (in-sample set)", navs["static (in-sample set)"], "#9e9e9e", 1.2),
               ("k120 (dynamic best)", navs["k120 (dynamic best)"], "#6a1b9a", 1.4),
               ("EW portfolio (hold all)", ew, "#1565c0", 2.0)]
    for label, src, color, lw in dd_sets:
        seg = seg_base(src, B2018)
        peak = np.maximum.accumulate(seg["norm"].to_numpy())
        dd = seg["norm"].to_numpy() / peak - 1.0
        ax.plot(seg["d"], dd * 100, color=color, lw=lw, label=label)
        ax.annotate(f"{dd.min()*100:.1f}%", (seg["d"].iloc[int(np.argmin(dd))], dd.min() * 100),
                    xytext=(0, -12), textcoords="offset points", fontsize=9, color=color, ha="center")
    ax.set_title("v2 screen - drawdown segment 2018-2025", fontsize=13)
    ax.set_ylabel("drawdown %")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(CH / "02_drawdown_segment_2018_2025.png")
    plt.close(fig)

    # ---- chart 03: full period log ----
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=110)
    for label, (arm, color, lw, ls) in ARMS.items():
        seg = seg_base(navs[label], B2010)
        ax.plot(seg["d"], seg["norm"], color=color, lw=lw, ls=ls, label=label)
    for label, color, lw in [("EW portfolio (hold all)", "#1565c0", 2.2), ("HS300 index", "#bdbdbd", 1.4)]:
        src = ew if "EW" in label else ix
        seg = seg_base(src, B2010)
        ax.plot(seg["d"], seg["norm"], color=color, lw=lw, label=label)
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["0.5x", "1x", "2x", "5x", "10x", "20x", "50x"])
    ax.set_title("v2 screen - NAV full 2010-2025 (log scale, base = 1.0 @ 2010-01-04)", fontsize=13)
    ax.set_ylabel("normalized NAV (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(CH / "03_nav_full_2010_2025_log.png")
    plt.close(fig)

    for f in sorted(CH.glob("*.png")):
        print(f)
    print("done")


if __name__ == "__main__":
    main()
