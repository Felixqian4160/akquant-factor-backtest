"""V33 picks: factor-return ranking voting (BULL + BEAR) using Causal ZigZag router.

合同:
- Panel: V33 with_new_factors (458 cols)
- Voting factors: 428 (= 458 - 19 base cols - 11 zigzag labels)
- Router: Causal ZigZag (leg.start, no look-ahead)
- Selector: factor-return ranking voting (top10 - bot10 net20 mean) in BOTH regimes
- Rebalance: every 20 trading days, single-run (offset=0)
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PANEL = Path("data/wavehunter_hs300_v33_with_new_factors_20261003.parquet")
ROUTER_MAP = Path("evidence/causal_zigzag_router_20261006/router_map.json")

START = "2010-01-01"
END = "2025-12-31"
REBAL_STEP = 20
ENTRY_TO_EXIT_SHIFT = 21
LOOKBACK_SESSIONS = 60
TOP_K = 10
K_NOM = 10
MIN_VOTES_BULL = 2
MIN_STOCKS_BULL = 5
MAX_STOCKS_BULL = 10
MIN_VOTES_BEAR = 4
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10
ROUND_TRIP_COST = 0.005


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def dt_expr(d) -> pl.Expr:
    if isinstance(d, str):
        ts = pd.Timestamp(d).to_pydatetime()
    else:
        ts = pd.Timestamp(d).to_pydatetime()
    return pl.lit(ts, dtype=pl.Datetime("ms"))


def trading_dates() -> list[str]:
    df = pl.scan_parquet(PANEL).select(["trade_date"]).unique().sort("trade_date").collect()
    return [str(x)[:10] for x in df["trade_date"].to_list()]


def load_router() -> dict[str, str]:
    return json.loads(ROUTER_MAP.read_text())


def compute_factor_returns_polars(panel_full: pl.DataFrame, factors: list[str],
                                   all_dates_iso: list[str], lookback: int
                                   ) -> dict[str, dict[str, float]]:
    """For each factor, compute per-trade_date top10-bot10 net20 portfolio diff.
    Then for each rebal date, rolling mean over past `lookback` sessions.
    """
    log(f"  Polars factor returns, {len(factors)} factors × {len(all_dates_iso)} dates")
    t0 = time.time()

    # Pre-extract sorted trading dates as Python list of Timestamps
    n_dates = len(all_dates_iso)
    rebal_dates = all_dates_iso[lookback:]

    out: dict[str, dict[str, float]] = {d: {} for d in rebal_dates}

    n_factors = len(factors)
    for fi, f in enumerate(factors, start=1):
        # Filter null and rank
        df = panel_full.select(["trade_date", "_fwd_net", f])
        df = df.filter(pl.col(f).is_not_null() & pl.col("_fwd_net").is_not_null())
        df = df.with_columns(
            pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias("_rk")
        )
        # Compute per-date diff
        per_date = df.group_by("trade_date").agg([
            (pl.col("_fwd_net").filter(pl.col("_rk") <= 10).sum() / 10.0).alias("_top_mean"),
            (pl.col("_fwd_net").filter(pl.col("_rk") > (pl.col("_rk").max() - 10)).sum() / 10.0).alias("_bot_mean"),
        ]).sort("trade_date").select(["trade_date", (pl.col("_top_mean") - pl.col("_bot_mean")).alias("_diff")])

        # Convert to dict[date_pos] = diff
        date_pos = []
        diffs = []
        for row in per_date.iter_rows(named=True):
            d_str = str(row["trade_date"])[:10]
            if d_str in date_pos and False:
                pass
            pos = all_dates_iso.index(d_str) if d_str in all_dates_iso else -1
            if pos >= 0 and np.isfinite(row["_diff"]):
                date_pos.append(pos)
                diffs.append(row["_diff"])

        if not date_pos:
            continue

        date_pos_arr = np.asarray(date_pos, dtype=np.int64)
        diffs_arr = np.asarray(diffs, dtype=np.float64)

        # For each rebal date position i, take last `lookback` entries where pos < i
        for i in range(lookback, n_dates):
            d_iso = all_dates_iso[i]
            mask = date_pos_arr < i
            if mask.sum() < 10:
                continue
            window = diffs_arr[mask][-lookback:]
            window = window[np.isfinite(window)]
            if len(window) >= 10:
                out[d_iso][f] = float(np.mean(window))

        if fi % 25 == 0:
            log(f"    factor {fi}/{n_factors}: {f}, elapsed {time.time()-t0:.0f}s")

    log(f"  factor_returns done in {time.time()-t0:.0f}s")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebal-offset", type=int, default=0)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--lookback", type=int, default=LOOKBACK_SESSIONS,
                        help="rolling lookback sessions (default 60)")
    args = parser.parse_args()

    offset = args.rebal_offset
    OUT_ROOT = Path(args.out_root)
    log(f"=== build_picks_v33_factorrank offset={offset} ===")

    v33_schema = pl.read_parquet_schema(PANEL)
    v33_cols = list(v33_schema.keys())
    base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
                 "vol", "amount", "pct_chg", "adj_factor", "adj_close",
                 "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
                 "turnover_rate", "circ_cap", "cap", "symbol", "volume",
                 "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}
    zigzag_labels = {"v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
                     "v10_1_b1_start", "v10_1_b1_interval",
                     "v10_1_down_start", "v10_1_down_interval",
                     "v10_1_peak_zone", "v10_1_valley_zone",
                     "v10_1_zig_peak", "v10_1_zig_valley"}
    excluded = base_cols | zigzag_labels
    factors = [c for c in v33_cols if c not in excluded]
    log(f"v33 panel {len(v33_cols)} cols; voting factors={len(factors)}; "
        f"excluded {len(excluded)} base/zigzag")

    log(f"Building _fwd_net on full panel...")
    t0 = time.time()
    panel_full = (
        pl.scan_parquet(PANEL)
        .select(["trade_date", "ts_code", "open", "close"] + factors)
        .sort(["ts_code", "trade_date"])
        .with_columns([
            pl.col("open").shift(-1).over("ts_code").alias("_open_t1"),
            pl.col("close").shift(-ENTRY_TO_EXIT_SHIFT).over("ts_code").alias("_close_t21"),
        ])
        .with_columns(
            ((pl.col("_close_t21") / pl.col("_open_t1") - 1.0 - ROUND_TRIP_COST).alias("_fwd_net"))
        )
        .select(["trade_date", "ts_code", "_fwd_net"] + factors)
        .collect()
    )
    log(f"_fwd_net done in {time.time()-t0:.0f}s, shape {panel_full.shape}")

    all_dates = trading_dates()
    start_dt = pd.Timestamp(START).date()
    end_dt = pd.Timestamp(END).date()
    rebal_dates = all_dates[offset::REBAL_STEP]
    rebal_dates = [d for d in rebal_dates if start_dt <= pd.Timestamp(d).date() <= end_dt]
    log(f"rebal_dates: {len(rebal_dates)} (offset={offset})")

    factor_returns = compute_factor_returns_polars(panel_full, factors, all_dates, args.lookback)
    fr_rebal = {d: factor_returns.get(d, {}) for d in rebal_dates}
    log(f"factor_returns coverage: {sum(1 for v in fr_rebal.values() if v)}/{len(fr_rebal)} rebal dates have scores")

    router = load_router()
    log(f"router: {sum(1 for v in router.values() if v == 'bear')} bear / "
        f"{sum(1 for v in router.values() if v == 'bull_neutral')} bull_neutral")

    picks: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict] = {}
    bull_count = bear_count = 0
    t_picks = time.time()

    for idx, rd in enumerate(rebal_dates, start=1):
        d = pd.Timestamp(rd).date().isoformat()
        regime = router.get(d, "bull_neutral")

        if regime == "bull_neutral":
            bull_count += 1
            min_votes = MIN_VOTES_BULL
            min_stocks = MIN_STOCKS_BULL
            max_stocks = MAX_STOCKS_BULL
        else:
            bear_count += 1
            min_votes = MIN_VOTES_BEAR
            min_stocks = MIN_STOCKS_BEAR
            max_stocks = MAX_STOCKS_BEAR

        fr = fr_rebal.get(d, {})
        if not fr:
            continue
        ranked = sorted(fr.items(), key=lambda kv: -kv[1])
        active = [f for f, _ in ranked[:TOP_K]]

        day = panel_full.filter(dt_expr(d) == pl.col("trade_date"))
        if day.height == 0:
            continue
        codes = day.get_column("ts_code").to_list()
        values = day.select(active).to_numpy()
        votes = np.zeros(day.height, dtype=np.int32)

        for j, factor in enumerate(active):
            column = values[:, j]
            valid = np.flatnonzero(np.isfinite(column))
            if len(valid) < K_NOM:
                continue
            chosen = valid[np.argpartition(column[valid], -K_NOM)[-K_NOM:]]
            votes[chosen] += 1

        order = sorted(range(len(codes)), key=lambda i: (-int(votes[i]), str(codes[i])))
        selected = [i for i in order if int(votes[i]) >= min_votes]
        if len(selected) < min_stocks:
            selected = order[:min_stocks]
        selected = selected[:max_stocks]
        if not selected:
            continue
        picks[d] = {str(codes[i]): int(votes[i]) for i in selected}
        metadata[d] = {
            "regime": regime,
            "selector": "factor_return_voting",
            "n_active_factors": len(active),
            "n_picks": len(selected),
            "max_votes": int(votes.max()),
            "n_at_or_above_threshold": int((votes >= min_votes).sum()),
        }
        if idx % 25 == 0:
            log(f"  picks {idx}/{len(rebal_dates)}, date={rd}, regime={regime}, "
                f"selected={len(selected)}, max_votes={int(votes.max())}")

    out = OUT_ROOT / f"V14_{offset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "picks.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1))
    (out / "picks_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=1))

    summary = {
        "tag": f"V14_{offset}",
        "panel": str(PANEL),
        "panel_version": "v33_with_new_factors_20261003",
        "panel_total_cols": len(v33_cols),
        "window": f"{START} ~ {END}",
        "router": "causal ZigZag leg.start (≥100d legs, lookback-zero)",
        "selector": "factor_return_voting (BULL + BEAR)",
        "factor_count_voting": len(factors),
        "lookback_sessions": args.lookback,
        "excluded_zigzag_labels": sorted(zigzag_labels),
        "excluded_base_cols": sorted(base_cols),
        "rebalance_step": REBAL_STEP,
        "top_k": TOP_K,
        "k_nom": K_NOM,
        "min_votes_bull": MIN_VOTES_BULL,
        "min_votes_bear": MIN_VOTES_BEAR,
        "min_stocks_bull": MIN_STOCKS_BULL,
        "min_stocks_bear": MIN_STOCKS_BEAR,
        "max_stocks_bull": MAX_STOCKS_BULL,
        "max_stocks_bear": MAX_STOCKS_BEAR,
        "fwd_exit_shift": ENTRY_TO_EXIT_SHIFT,
        "round_trip_cost": ROUND_TRIP_COST,
        "requested_rebalances": len(rebal_dates),
        "rebalance_dates_with_picks": len(picks),
        "n_bull_neutral_rebalances": bull_count,
        "n_bear_rebalances": bear_count,
        "avg_n_stocks": float(np.mean([len(p) for p in picks.values()])) if picks else 0.0,
        "runtime_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log(f"saved {out}: {len(picks)} dates bull_neutral={bull_count} bear={bear_count} "
        f"elapsed {time.time()-t_picks:.0f}s")


if __name__ == "__main__":
    main()