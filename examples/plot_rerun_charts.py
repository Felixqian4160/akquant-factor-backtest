"""Plot NAV curves + 2026 OOS zoom + yearly bars for v34-lb20 strategy.

Inputs: evidence/rerun_strat_20261007/{picks,sims,sims_2026}/V14_*/nav.csv
        HS300 idx_close from v34 panel

Outputs:
  evidence/rerun_strat_20261007/charts/01_nav_full_2010_2025.png
  evidence/rerun_strat_20261007/charts/02_nav_full_2010_2026.png
  evidence/rerun_strat_20261007/charts/03_zoom_2026_oos.png
  evidence/rerun_strat_20261007/charts/04_yearly_bars_10offsets.png
"""
from __future__ import annotations
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import polars as pl

OUT_BASE = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest/evidence/rerun_strat_20261007")
CHARTS = OUT_BASE / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)
PANEL = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest/data/wavehunter_hs300_v34_adj_20261007.parquet")

OFFSETS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
INITIAL_CASH = 100_000_000.0


def load_navs(window: str) -> dict[int, pd.DataFrame]:
    navs = {}
    for off in OFFSETS:
        p = OUT_BASE / window / f"V14_{off}" / "nav.csv"
        if p.exists():
            navs[off] = pd.read_csv(p).assign(date_dt=lambda d: pd.to_datetime(d["date"]))
    return navs


def hs300_index() -> pd.DataFrame:
    df = pl.read_parquet(PANEL, columns=["trade_date", "idx_close"]).unique().sort("trade_date")
    df = df.filter(pl.col("idx_close").is_not_null()).with_columns(pl.col("idx_close").cast(pl.Float64))
    pdf = df.to_pandas().rename(columns={"trade_date": "date_dt", "idx_close": "idx"})
    pdf["nav"] = INITIAL_CASH * (pdf["idx"] / pdf["idx"].iloc[0])
    # ffill in case sim runs past panel last date
    full_dates = pd.date_range(pdf["date_dt"].min(), pdf["date_dt"].max() + pd.Timedelta(days=10), freq="B")
    pdf = pdf.set_index("date_dt").reindex(full_dates).ffill().reset_index().rename(columns={"index": "date_dt"})
    return pdf


def nav_to_returns(nav_df: pd.DataFrame) -> pd.DataFrame:
    pdf = nav_df.copy()
    pdf["ret"] = pdf["value"].pct_change().fillna(0.0)
    return pdf


def plot_all_navs(ax, navs, end_label, hs300_df):
    """Plot all 10 offsets (transparent) + mean (bold) + HS300 benchmark."""
    if not navs:
        ax.text(0.5, 0.5, "no sims", ha="center", va="center", transform=ax.transAxes)
        return
    # aligned values matrix
    common = sorted(set.intersection(*[set(df["date_dt"]) for df in navs.values()]))
    matrix = np.column_stack([
        df.set_index("date_dt").loc[common, "value"].to_numpy(dtype=float) for df in navs.values()
    ])
    mean = matrix.mean(axis=1)
    p10 = np.percentile(matrix, 10, axis=1)
    p90 = np.percentile(matrix, 90, axis=1)
    p25 = np.percentile(matrix, 25, axis=1)
    p75 = np.percentile(matrix, 75, axis=1)

    # shade IQR and p10-p90
    common_dates = pd.to_datetime(common)
    ax.fill_between(common_dates, p10, p90, color="steelblue", alpha=0.10, label="p10-p90 range")
    ax.fill_between(common_dates, p25, p75, color="steelblue", alpha=0.18, label="p25-p75 range (IQR)")
    for off, df in navs.items():
        ax.plot(df["date_dt"], df["value"] / 1e6, color="#222", alpha=0.10, linewidth=0.7)
    ax.plot(common_dates, mean / 1e6, color="crimson", linewidth=2.5, label=f"10-offset mean (final ¥{mean[-1]/1e6:,.0f}M)")
    ax.plot(common_dates, p10 / 1e6, color="orange", linewidth=1.2, linestyle="--", label=f"p10 (worst offset)")
    ax.plot(common_dates, p90 / 1e6, color="green", linewidth=1.2, linestyle="--", label=f"p90 (best offset)")

    # HS300 buy-hold
    if hs300_df is not None:
        hs_sub = hs300_df.set_index("date_dt").loc[common]
        ax.plot(hs_sub.index, hs_sub["nav"] / 1e6, color="gray", linewidth=2.0, label=f"HS300 buy-hold (final ¥{hs_sub['nav'].iloc[-1]/1e6:,.0f}M)")

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
    matrix = matrix / matrix[0, :]  # normalize to 1.0 at first day
    dates = pd.to_datetime(common)
    mean = matrix.mean(axis=1)
    p10 = np.percentile(matrix, 10, axis=1)
    p90 = np.percentile(matrix, 90, axis=1)
    ax.fill_between(range(len(dates)), (p10 - 1) * 100, (p90 - 1) * 100, color="steelblue", alpha=0.10, label="p10-p90")
    ax.fill_between(range(len(dates)), (matrix.min(axis=1) - 1) * 100, (matrix.max(axis=1) - 1) * 100, color="steelblue", alpha=0.20, label="min-max")
    for off, df in navs_oos.items():
        v = df.set_index("date_dt").loc[common, "value"].to_numpy()
        v = v / v[0]
        ax.plot(range(len(dates)), (v - 1) * 100, color="#222", alpha=0.10, linewidth=0.7)
    ax.plot(range(len(dates)), (mean - 1) * 100, color="crimson", linewidth=2.5, label=f"10-offset mean ({((mean[-1]-1)*100):+.2f}%)")

    if hs300_df is not None:
        hs = hs300_df.set_index("date_dt").loc[common]
        hs = hs / hs.iloc[0]
        ax.plot(range(len(dates)), (hs["nav"].values - 1) * 100, color="gray", linewidth=2.0, label=f"HS300 ({((hs["nav"].iloc[-1]-1)*100):+.2f}%)")

    ax.set_xticks(np.linspace(0, len(dates) - 1, 8))
    ax.set_xticklabels([dates[int(k)].strftime("%Y-%m") for k in np.linspace(0, len(dates) - 1, 8)], rotation=0)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_ylabel("Cumulative return (%)", fontsize=12)
    ax.set_title("2026 OOS 8-month segment zoom (normalized to 2026-01-05 = 0%)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.tick_params(labelsize=10)


def plot_yearly_bars(ax, navs_full):
    """Per-year mean ± std annual return across 10 offsets."""
    if not navs_full:
        return
    common = sorted(set.intersection(*[set(df["date_dt"]) for df in navs_full.values()]))
    base = pd.DataFrame({"date_dt": pd.to_datetime(common)})
    for off in OFFSETS:
        s = navs_full[off].set_index("date_dt").loc[common, "value"]
        base[f"off{off}"] = s.values

    base["year"] = base["date_dt"].dt.year
    years = []
    means, stds, mins, maxs, npos = [], [], [], [], []
    for yr in sorted(base["year"].unique()):
        if yr < 2010: continue
        last = base[base.year == yr].iloc[-1]
        if yr == 2010:
            first = base[base.year < 2010].iloc[-1] if (base.year < 2010).any() else base[base.year == 2010].iloc[0]
        else:
            first = base[base.year == yr - 1].iloc[-1]
        row_rets = [(last[f"off{off}"] / first[f"off{off}"] - 1) for off in OFFSETS]
        years.append(int(yr))
        means.append(np.mean(row_rets))
        stds.append(np.std(row_rets, ddof=1))
        mins.append(min(row_rets))
        maxs.append(max(row_rets))
        npos.append(sum(1 for r in row_rets if r > 0))

    x = np.arange(len(years))
    bars = ax.bar(x, [m * 100 for m in means], yerr=[s * 100 for s in stds],
                 capsize=3, color=["crimson" if m > 0 else "steelblue" for m in means],
                 alpha=0.7, edgecolor="black", linewidth=0.5, error_kw={"linewidth": 0.8})
    # annotations
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


def main():
    print("Loading navs + HS300...")
    navs_full = load_navs("sims")
    navs_oos = load_navs("sims_2026")
    hs300 = hs300_index()
    print(f"  full: {len(navs_full)}, oos: {len(navs_oos)}; HS300: {len(hs300)}")

    print("Plotting...")
    plt.rcParams["font.family"] = ["DejaVu Sans"]

    # 1. full 2010-2025
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_all_navs(ax, navs_full, "end=2010-2025", hs300)
    fig.tight_layout()
    fig.savefig(CHARTS / "01_nav_full_2010_2025.png", dpi=110)
    plt.close(fig)
    print(f"  -> 01_nav_full_2010_2025.png")

    # 2. full 2010-2026
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_all_navs(ax, navs_oos, "end=2010-2026.08 (含 8 个月 2026 OOS)", hs300)
    fig.tight_layout()
    fig.savefig(CHARTS / "02_nav_full_2010_2026.png", dpi=110)
    plt.close(fig)
    print(f"  -> 02_nav_full_2010_2026.png")

    # 3. 2026 OOS zoom
    fig, ax = plt.subplots(figsize=(13, 7))
    plot_zoom_2026(ax, navs_oos, hs300)
    fig.tight_layout()
    fig.savefig(CHARTS / "03_zoom_2026_oos.png", dpi=110)
    plt.close(fig)
    print(f"  -> 03_zoom_2026_oos.png")

    # 4. yearly bars
    fig, ax = plt.subplots(figsize=(15, 7))
    plot_yearly_bars(ax, navs_full)
    fig.tight_layout()
    fig.savefig(CHARTS / "04_yearly_bars_10offsets.png", dpi=110)
    plt.close(fig)
    print(f"  -> 04_yearly_bars_10offsets.png")

    print("DONE")


if __name__ == "__main__":
    main()