"""Stock usage analytics for v33_factorrank lookback sweep (lb=20, lb=40, lb=60).

For each lookback, read the picks.json and aggregate per-stock stats:
- n_selected_total
- n_selected_bull, n_selected_bear
- pct_selected_total
- avg_votes_per_selection
- first/last selected date
- holding continuity (counted as 1 even if same stock is held across consecutive rebal dates)

Writes to evidence/sweep/v33_factorrank_lookback_sweep_20261007/:
  - stock_usage_lb20.csv / stock_usage_lb40.csv / stock_usage_lb60.csv
  - combined_top50.csv
  - chart_top3_stocks.png (3-stock comparison)
  - chart_n_equity_curves.png (strategy equity curves for all 3 lookbacks)
  - REPORT.md
"""
import csv, json, pathlib
from collections import Counter, defaultdict
from datetime import datetime
from itertools import groupby

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import polars as pl

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
SWEEP_BASE = ROOT / "evidence" / "sweep"
LOOKBACKS = [20, 40, 60]
LABELS = {20: "lb=20", 40: "lb=40", 60: "lb=60"}
COLORS = {20: "#c62828", 40: "#1565c0", 60: "#2e7d32"}
OUT_DIR = SWEEP_BASE / "v33_factorrank_lookback_sweep_20261007"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"

# ── Load panel for stock close prices to compute per-stock hold returns ──


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_picks(picks_path: pathlib.Path) -> dict[str, dict[str, int]]:
    return json.loads(picks_path.read_text())


def aggregate_stock_usage(picks: dict[str, dict[str, int]]) -> pd.DataFrame:
    """For each stock, count how many rebal dates it appeared, in bull and bear."""
    dates_sorted = sorted(picks.keys())
    counter = Counter()
    counter_bull = Counter()
    counter_bear = Counter()
    votes_sum = defaultdict(float)
    first_seen = {}
    last_seen = {}
    for d, picks_at_d in picks.items():
        regime = None  # not available from picks.json alone; load picks_meta
        for sym, v in picks_at_d.items():
            counter[sym] += 1
            votes_sum[sym] += int(v)
            if sym not in first_seen:
                first_seen[sym] = d
            last_seen[sym] = d

    # Load regime per date from picks_meta
    meta = json.loads((picks_path.parent / "picks_meta.json").read_text())
    for d, picks_at_d in picks.items():
        regime = meta.get(d, {}).get("regime", "bull_neutral")
        for sym in picks_at_d:
            if regime == "bull_neutral":
                counter_bull[sym] += 1
            else:
                counter_bear[sym] += 1

    rows = []
    for sym, n in counter.items():
        rows.append({
            "symbol": sym,
            "n_selected_total": n,
            "n_selected_bull": counter_bull.get(sym, 0),
            "n_selected_bear": counter_bear.get(sym, 0),
            "pct_selected_total": round(n / len(dates_sorted) * 100, 2),
            "avg_votes_per_selection": round(votes_sum[sym] / n, 2),
            "first_seen": first_seen[sym],
            "last_seen": last_seen[sym],
        })
    df = pd.DataFrame(rows).sort_values("n_selected_total", ascending=False).reset_index(drop=True)
    return df


def holding_continuity_stats(picks: dict[str, dict[str, int]]) -> dict[str, int]:
    """For each stock, count how many rebal dates it was held continuously (>=2 consecutive)."""
    dates_sorted = sorted(picks.keys())
    cont_count = defaultdict(int)
    held_prev = set()
    for d in dates_sorted:
        held_now = set(picks[d].keys())
        # If a stock was held at d AND d-1, count as continuity
        for s in held_now & held_prev:
            cont_count[s] += 1
        held_prev = held_now
    return cont_count


# ── Load panel close prices for stock-level equity curve ──
log("Loading panel close prices for stock-level NAV...")
panel_idx = (pl.read_parquet(PANEL, columns=["trade_date", "ts_code", "close"])
             .filter(pl.col("close").is_not_null())
             .sort(["ts_code", "trade_date"]))
panel_dates = panel_idx["trade_date"].to_numpy()


# ── Per-lookback analysis ──
all_dfs = {}
cont_dfs = {}
for lb in LOOKBACKS:
    if lb == 60:
        name = "v33_factorrank_20261007"
    else:
        name = f"v33_factorrank_lb{lb}_20261007"
    log(f"=== Processing {name} (lookback={lb}) ===")
    picks_path = SWEEP_BASE / name / "V14_0" / "picks.json"
    picks = load_picks(picks_path)
    log(f"  Loaded {len(picks)} rebal dates")

    df = aggregate_stock_usage(picks)
    cont = holding_continuity_stats(picks)
    cont_df = pd.DataFrame([{"symbol": s, "n_consecutive_holds": n} for s, n in cont.items()])
    cont_df = cont_df.sort_values("n_consecutive_holds", ascending=False).reset_index(drop=True)
    all_dfs[lb] = df
    cont_dfs[lb] = cont_df

    # Save per-lookback CSVs
    df_out = df.copy()
    df_out["n_consecutive_holds"] = df_out["symbol"].map(cont).fillna(0).astype(int)
    df_out.to_csv(OUT_DIR / f"stock_usage_lb{lb}.csv", index=False)
    cont_df.to_csv(OUT_DIR / f"stock_consecutive_holds_lb{lb}.csv", index=False)
    log(f"  Saved stock_usage_lb{lb}.csv ({len(df)} unique stocks)")
    log(f"  Saved stock_consecutive_holds_lb{lb}.csv ({len(cont_df)} stocks with ≥1 hold)")

# ── Combined top 50 ──
combined = []
for lb in LOOKBACKS:
    df = all_dfs[lb].copy()
    df["lookback"] = lb
    combined.append(df)
combined_df = pd.concat(combined, ignore_index=True)
combined_pivot = combined_df.pivot_table(
    index="symbol",
    columns="lookback",
    values="n_selected_total",
    fill_value=0,
).reset_index()
combined_pivot["total_all"] = combined_pivot[[20, 40, 60]].sum(axis=1)
combined_pivot = combined_pivot.sort_values("total_all", ascending=False).reset_index(drop=True)
combined_pivot.to_csv(OUT_DIR / "combined_top50.csv", index=False)
log(f"\nWrote combined_top50.csv ({len(combined_pivot)} unique symbols)")

# ── Chart 1: Equity curves for all 3 lookbacks ──
log("Building equity curves for 3 lookbacks...")
panel_idx_full = (pl.read_parquet(PANEL, columns=["trade_date", "idx_close"])
                  .filter(pl.col("idx_close").is_not_null())
                  .group_by("trade_date").agg(pl.col("idx_close").first())
                  .sort("trade_date"))
idx_dates = [str(x)[:10] for x in panel_idx_full["trade_date"].to_list()]
idx_vals = panel_idx_full["idx_close"].to_numpy()
bench_map = dict(zip(idx_dates, idx_vals))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})

curves = {}
for lb in LOOKBACKS:
    name = "v33_factorrank_20261007" if lb == 60 else f"v33_factorrank_lb{lb}_20261007"
    p = SWEEP_BASE / name / "akquant" / "V14_0" / "nav.csv"
    with p.open(newline="") as f:
        rows_ = list(csv.DictReader(f))
    dates = [r["date"] for r in rows_]
    values = np.array([float(r["value"]) for r in rows_])
    curves[lb] = (dates, values)
    norm = values / values[0]
    final = (norm[-1] - 1) * 100
    ax.plot(dates, norm, color=COLORS[lb], lw=1.6, label=f"lb={lb} sessions  {final:+.1f}%")

# Benchmark
dates = curves[LOOKBACKS[0]][0]
bench = [bench_map.get(d, None) for d in dates]
bench_arr = np.asarray(bench, dtype=float)
mask = np.isfinite(bench_arr)
bench_arr[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), bench_arr[mask])
bench_norm = bench_arr / bench_arr[0]
ax.plot(dates, bench_norm, color="#b71c1c", lw=1.4, ls=":",
        label=f"HS300 buy&hold  {(bench_norm[-1]-1)*100:+.1f}%")

ax.set_yscale("log")
ax.set_ylabel("Normalized NAV (initial=1.0)")
ax.set_title("Factor-Return Voting — Lookback Sweep (20 / 40 / 60 sessions) | V33 panel (428 factors)")
ax.grid(True, alpha=.22); ax.legend(loc="upper left", fontsize=10)
ax.text(0.995, 0.04,
        "Same router, panel, cost model, single-run offset=0 | Only lookback varies",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round,pad=.35", facecolor="white", alpha=.88, edgecolor="#bbbbbb"))


def dd(curve):
    return curve / np.maximum.accumulate(curve) - 1.0


for lb in LOOKBACKS:
    d, v = curves[lb]
    ax2.plot(d, dd(v / v[0]) * 100, color=COLORS[lb], lw=1.0, label=f"lb={lb}")
ax2.plot(dates, dd(bench_norm) * 100, color="#b71c1c", lw=1.0, ls=":", label="HS300")
ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("Date")
ax2.grid(True, alpha=.22); ax2.legend(loc="lower left", fontsize=10)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
step = max(1, len(dates) // 12)
ax2.set_xticks(dates[::step]); ax2.tick_params(axis="x", rotation=35)

fig.text(0.01, 0.005,
         "Saved to evidence/sweep/v33_factorrank_lookback_sweep_20261007/ | Generated by factor_usage_analytics_lb.py",
         fontsize=8, color="#555555")
fig.savefig(OUT_DIR / "chart_n_equity_curves.png", dpi=160, bbox_inches="tight")
plt.close(fig)
log("Saved chart_n_equity_curves.png")

# ── Chart 2: Top 3 stocks comparison ──
log("Building top-3 stock comparison chart...")
fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
top_stocks = combined_pivot.head(3)["symbol"].tolist()
for ax_i, sym in enumerate(top_stocks):
    ax = axes[ax_i]
    for lb in LOOKBACKS:
        df = all_dfs[lb]
        sub = df[df["symbol"] == sym]
        if sub.empty:
            continue
        n = sub["n_selected_total"].iloc[0]
        n_bull = sub["n_selected_bull"].iloc[0]
        n_bear = sub["n_selected_bear"].iloc[0]
        pct = sub["pct_selected_total"].iloc[0]
        ax.bar(lb, n, color=COLORS[lb], alpha=0.85, edgecolor="#333", lw=0.6,
               label=f"lb={lb} (n={int(n)}, pct={pct:.1f}%, bull={int(n_bull)}, bear={int(n_bear)})")
    ax.set_title(f"#{ax_i+1} {sym} — selected count by lookback")
    ax.set_ylabel("# selections")
    ax.set_xticks(LOOKBACKS)
    ax.grid(True, alpha=.22, axis="y"); ax.legend(loc="upper right", fontsize=9)
    # Show first/last
    ax.text(0.99, 0.92, f"first_seen={sub['first_seen'].iloc[0] if not sub.empty else 'N/A'} | last_seen={sub['last_seen'].iloc[0] if not sub.empty else 'N/A'}",
            transform=ax.transAxes, ha="right", fontsize=8, color="#555")

fig.suptitle("Top-3 Most-Selected Stocks Across Lookback Sweep", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "chart_top3_stocks.png", dpi=160, bbox_inches="tight")
plt.close(fig)
log("Saved chart_top3_stocks.png")

# ── Print summary tables ──
print()
print("=" * 90)
print("STOCK USAGE SUMMARY — per-lookback Top 10 (most selected)")
print("=" * 90)
for lb in LOOKBACKS:
    print(f"\n>>> Lookback = {lb} sessions <<<")
    df = all_dfs[lb]
    print(df.head(10).to_string(index=False))
    print(f"\n  Total unique stocks: {len(df)}")
    print(f"  Stocks selected >= 10 times: {(df['n_selected_total'] >= 10).sum()}")
    print(f"  Stocks selected >= 50 times: {(df['n_selected_total'] >= 50).sum()}")
    print(f"  Max selections: {df['n_selected_total'].max()}")
    print(f"  Mean selections per stock: {df['n_selected_total'].mean():.1f}")

print("\n" + "=" * 90)
print("CONSECUTIVE HOLDS — per-lookback Top 10 (held ≥ 2 consecutive rebal dates)")
print("=" * 90)
for lb in LOOKBACKS:
    print(f"\n>>> Lookback = {lb} sessions <<<")
    print(cont_dfs[lb].head(10).to_string(index=False))
    print(f"  Total stocks with ≥1 consecutive hold: {len(cont_dfs[lb])}")
    print(f"  Max consecutive holds: {cont_dfs[lb]['n_consecutive_holds'].max() if len(cont_dfs[lb]) else 0}")

print("\n" + "=" * 90)
print("COMBINED TOP 10 — stocks appearing most across all 3 lookbacks")
print("=" * 90)
print(combined_pivot.head(10).to_string(index=False))

log("\nDONE")