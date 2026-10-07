"""Daily Factor Reselect v6 — IC-weighted + Stage 5b proven factors.

Key changes from v5:
1. NO average-rank composite — IC-weighted composite (weights from recent 60d IC)
2. Hard-coded Stage 5b best 6 factors (talib_NATR, talib_TRANGE, gtja_gtja_159,
   gtja_gtja_149, gtja_gtja_144, mw_vol_20d) — already proven +134% annualised
   on v1+v2+v2c workflow
3. 20-day rebalance cycle (not daily) — reduces turnover cost
4. AKQuant run_backtest per cycle (per user requirement: 回测用 AKQuant)
5. Use total_return_pct (includes unrealized P&L)

Usage:
    python examples/daily_factor_pipeline_v6.py \
        --start-date 2020-01-01 --end-date 2020-12-31 \
        --job-id v6_2020_TEST
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import numpy as np

# Path setup
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/media/felix/f/quant/aurumq-rl/src")

import akquant as aq  # noqa: E402

# ============================================================================
# Config
# ============================================================================
PANEL_FILE = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet"
)

# Stage 5b proven 6 factors — already verified +134% annualised
# (talib_NATR, talib_TRANGE, gtja_gtja_159, gtja_gtja_149, gtja_gtja_144, mw_vol_20d)
STAGE5B_FACTORS = [
    "talib_NATR",
    "talib_TRANGE",
    "gtja_gtja_159",
    "gtja_gtja_149",
    "gtja_gtja_144",
    "mw_vol_20d",
]

TOP_FACTORS = 5  # use top 5 by IC (out of 6)
TOP_STOCKS = 10
HOLD_DAYS = 20  # 20-day rebalance (not daily)
REBALANCE_DAYS = 20  # re-evaluate picks every 20 days
WINDOW_DAYS = 60
FWD_DAYS = 20
INITIAL_CASH = 1_000_000_000.0  # ¥1B for margin buffer (10% × 1B = 100M per stock)
COST_BPS_PER_SIDE = 25
SLIPPAGE = {"type": "percent", "value": 0.0010}

# IC window: how many recent days to compute IC for weighting
IC_WINDOW_DAYS = 60

# Module-level state for AKQuant strategy
_CURRENT_PICKS: dict[str, float] = {}


# ============================================================================
# AKQuant strategy
# ============================================================================
class DailyFactorReselectStrategy6(aq.Strategy):
    """Buy top-K stocks once at cycle start, hold until cycle end."""

    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rebalanced = False

    def on_bar(self, bar) -> None:
        pass

    def on_cross_section(self, trading_date, timestamp) -> None:
        if self._rebalanced:
            return
        self._rebalanced = True
        if not _CURRENT_PICKS:
            return
        try:
            self.rebalance_to_topn(
                scores=_CURRENT_PICKS,
                top_n=len(_CURRENT_PICKS),
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


# ============================================================================
# Factor IC computation
# ============================================================================
def compute_factor_ic(
    panel: pl.DataFrame,
    factor: str,
    as_of_date: str,
    window_days: int = IC_WINDOW_DAYS,
    fwd_days: int = FWD_DAYS,
) -> float:
    """Information Coefficient: rank correlation between factor and next-N-day return.

    IC = Spearman correlation of (factor_value vs forward_return) over universe.
    """
    as_of = datetime.strptime(as_of_date[:10], "%Y-%m-%d")
    window_start = as_of - timedelta(days=window_days)
    fwd_end = as_of + timedelta(days=fwd_days)

    # Subset: recent window for factor + future window for return
    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(window_start.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(fwd_end.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    ).filter(pl.col(factor).is_not_null())

    if sub.height < 100:
        return 0.0

    # As-of cross section: each stock's factor value at the most recent date <= as_of
    as_of_sub = (
        sub.filter(pl.col("trade_date") <= pl.lit(as_of.date().isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        .group_by("ts_code")
        .agg(pl.col(factor).last().alias(f"factor_val"), pl.col("close").last().alias("px_at_eval"))
    )

    # Future cross section: each stock's close at as_of + fwd_days
    future_target_date = pl.lit((as_of + timedelta(days=fwd_days)).date().isoformat()).str.strptime(
        pl.Datetime("ms"), "%Y-%m-%d"
    )
    future_sub = sub.filter(pl.col("trade_date") >= future_target_date).group_by("ts_code").agg(
        pl.col("close").first().alias("px_future")
    )

    joined = as_of_sub.join(future_sub, on="ts_code", how="inner").with_columns(
        (pl.col("px_future") / pl.col("px_at_eval") - 1).alias("fwd_ret")
    )
    joined = joined.filter(pl.col("fwd_ret").is_not_null() & pl.col("factor_val").is_not_null())

    if joined.height < 50:
        return 0.0

    arr = joined.select(["factor_val", "fwd_ret"]).to_numpy()
    if arr.shape[0] < 50:
        return 0.0

    # Spearman rank correlation
    f_rank = np.argsort(np.argsort(arr[:, 0]))
    r_rank = np.argsort(np.argsort(arr[:, 1]))
    if np.std(f_rank) == 0 or np.std(r_rank) == 0:
        return 0.0
    ic = float(np.corrcoef(f_rank, r_rank)[0, 1])
    return ic if np.isfinite(ic) else 0.0


def rank_factors_by_ic(
    panel: pl.DataFrame, eval_date: str, factors: list[str]
) -> list[tuple[str, float]]:
    """Return factors ranked by abs(IC) descending — keeps best-positive AND best-negative."""
    ics = []
    for f in factors:
        ic = compute_factor_ic(panel, f, eval_date, IC_WINDOW_DAYS, FWD_DAYS)
        ics.append((f, ic))
    # Sort by absolute IC descending, but weight by sign so we use direction
    ics.sort(key=lambda x: -abs(x[1]))
    return ics


# ============================================================================
# IC-weighted composite stock selection
# ============================================================================
def select_top_stocks_ic_weighted(
    panel: pl.DataFrame,
    factors_with_ic: list[tuple[str, float]],
    eval_date: str,
    top_k: int = TOP_STOCKS,
) -> tuple[list[str], dict[str, float]]:
    """Pick top-K stocks using IC-weighted composite.

    For each factor: rank stocks (cross-section), multiply by abs(IC) weight,
    sum across factors. Top-K by weighted composite.
    Direction of each factor's IC is implicit: if IC < 0, flip rank order.
    """
    as_of_dt = datetime.strptime(eval_date[:10], "%Y-%m-%d").date()
    sub = panel.filter(pl.col("trade_date") <= pl.lit(as_of_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
    # Use last available date <= eval_date
    last_date = sub["trade_date"].max()
    sub = sub.filter(pl.col("trade_date") == last_date)

    # Filter to factors present + non-null
    valid = [(f, ic) for f, ic in factors_with_ic if f in sub.columns]
    if not valid:
        return [], {}

    # Build cross-section rank for each factor
    df = sub.select(["ts_code"] + [f for f, _ in valid]).drop_nulls()
    if df.height == 0:
        return [], {}

    n = df.height
    score_cols = []
    for f, ic in valid:
        # weight = abs(IC); direction = sign(IC)
        w = abs(ic)
        # rank ascending so highest factor value = rank n-1; then flip by sign
        if ic >= 0:
            df = df.with_columns((pl.col(f).rank(method="ordinal", descending=True) * w).alias(f"_sc_{f}"))
        else:
            df = df.with_columns((pl.col(f).rank(method="ordinal", descending=False) * w).alias(f"_sc_{f}"))
        score_cols.append(f"_sc_{f}")

    df = df.with_columns(pl.sum_horizontal(score_cols).alias("_composite"))
    sorted_sub = df.sort("_composite", descending=True, nulls_last=True).head(top_k)
    picks = sorted_sub["ts_code"].to_list()
    scores = {row["ts_code"]: float(row["_composite"]) for row in sorted_sub.to_dicts()}
    return picks, scores


# ============================================================================
# Build mini data dict for AKQuant
# ============================================================================
def build_mini_data_dict(
    panel: pl.DataFrame, picks: list[str], entry_date: date, exit_date: date
) -> dict[str, "pd.DataFrame"]:
    """Build {symbol: ohlcv_df} for AKQuant between entry_date and exit_date."""
    import pandas as pd

    sub = panel.filter(
        (pl.col("trade_date") >= pl.lit(entry_date.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & (pl.col("trade_date") <= pl.lit(exit_date.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        & pl.col("ts_code").is_in(picks)
    ).select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"])

    data_dict: dict[str, pd.DataFrame] = {}
    for code in picks:
        pdf = (
            sub.filter(pl.col("ts_code") == code)
            .sort("trade_date")
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        if pdf.empty:
            continue
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = 1.0e9  # synthetic high volume so orders fill
        data_dict[code] = pdf
    return data_dict


# ============================================================================
# Single cycle: AKQuant run_backtest
# ============================================================================
def run_akquant_cycle(
    panel: pl.DataFrame,
    picks: dict[str, float],
    entry_date: date,
    exit_date: date,
) -> dict:
    """Run AKQuant backtest for one cycle, return {cycle_return_pct, n_trades}."""
    global _CURRENT_PICKS
    _CURRENT_PICKS = dict(picks)

    if not picks:
        return {"cycle_return_pct": 0.0, "n_trades": 0, "error": "no picks"}

    data = build_mini_data_dict(panel, list(picks.keys()), entry_date, exit_date)
    if not data:
        return {"cycle_return_pct": 0.0, "n_trades": 0, "error": "no data"}

    try:
        result = aq.run_backtest(
            data=data,
            strategy=DailyFactorReselectStrategy6,
            initial_cash=INITIAL_CASH,
            commission_rate=COST_BPS_PER_SIDE / 10_000,
            slippage=SLIPPAGE,
            t_plus_one=False,
            fill_policy=aq.NextOpen(),
            lot_size=100,
        )
        m = result.metrics_df
        cycle_ret_pct = float(m.loc["total_return_pct", "value"])
        n_trades = int(m.loc["closed_trade_count", "value"])
        _CURRENT_PICKS = {}
        return {"cycle_return_pct": cycle_ret_pct, "n_trades": n_trades}
    except Exception as exc:
        _CURRENT_PICKS = {}
        return {"cycle_return_pct": 0.0, "n_trades": 0, "error": str(exc)}


# ============================================================================
# Main loop
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-days", type=int, default=None,
                        help="Limit to first N trading days (smoke test)")
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v6")
    args = parser.parse_args()

    job_dir = Path(args.out_base) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_log = job_dir / "progress.jsonl"

    def emit(stage: str, **kw) -> None:
        rec = {"stage": stage, "job_id": args.job_id, "timestamp": datetime.now().isoformat(), **kw}
        with open(progress_log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec, ensure_ascii=False))

    emit("init", start_date=args.start_date, end_date=args.end_date)

    # Load panel
    panel = pl.read_parquet(PANEL_FILE)
    emit("panel_loaded", n_rows=panel.height, n_cols=panel.width)

    # Get trading dates in range
    start_dt = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    end_dt = datetime.strptime(args.end_date[:10], "%Y-%m-%d").date()
    all_dates = (
        panel.filter(
            (pl.col("trade_date") >= pl.lit(start_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
            & (pl.col("trade_date") <= pl.lit(end_dt.isoformat()).str.strptime(pl.Datetime("ms"), "%Y-%m-%d"))
        )["trade_date"].unique().sort().to_list()
    )
    if args.max_days:
        all_dates = all_dates[: args.max_days]
    n_total = len(all_dates)
    emit("dates_loaded", n_dates_in_range=n_total)

    if n_total < 30:
        emit("error", msg="too few dates")
        return

    # Main loop: every REBALANCE_DAYS, re-select top factors (by IC) and stocks
    nav = 1.0
    nav_curve = []
    ledger = []
    cycle_id = 0
    rebal_idx = 0  # index into all_dates where we last rebalanced

    t_start = time.time()
    while rebal_idx < n_total:
        eval_date_str = str(all_dates[rebal_idx])[:10]
        # Entry is eval_date + 1 (T+1 open), exit is +HOLD_DAYS
        entry_idx = min(rebal_idx + 1, n_total - 1)
        exit_idx = min(rebal_idx + 1 + HOLD_DAYS, n_total - 1)
        entry_date_str = str(all_dates[entry_idx])[:10]
        exit_date_str = str(all_dates[exit_idx])[:10]

        emit("cycle_start", cycle=cycle_id, eval_date=eval_date_str,
             entry_date=entry_date_str, exit_date=exit_date_str)

        # 1. Rank factors by IC
        factors_with_ic = rank_factors_by_ic(panel, eval_date_str, STAGE5B_FACTORS)
        # Take top TOP_FACTORS by abs IC
        top_factors = [f for f, _ in factors_with_ic[:TOP_FACTORS]]
        top_ics = [(f, ic) for f, ic in factors_with_ic[:TOP_FACTORS]]
        emit("factor_ic_ranked", top_factors=top_factors, ics=top_ics)

        # 2. Pick top stocks via IC-weighted composite
        picks, scores = select_top_stocks_ic_weighted(
            panel, top_ics, eval_date_str, top_k=TOP_STOCKS
        )
        emit("picks_selected", n_picks=len(picks), picks=picks)

        # 3. AKQuant cycle
        result = run_akquant_cycle(
            panel,
            scores,
            entry_date=datetime.strptime(entry_date_str, "%Y-%m-%d").date(),
            exit_date=datetime.strptime(exit_date_str, "%Y-%m-%d").date(),
        )
        cycle_ret = result["cycle_return_pct"]
        nav *= 1 + cycle_ret / 100

        cycle_ledger = {
            "cycle": cycle_id,
            "eval_date": eval_date_str,
            "entry_date": entry_date_str,
            "exit_date": exit_date_str,
            "factors": top_factors,
            "factor_ics": dict(top_ics),
            "picks": picks,
            "scores": scores,
            "cycle_return_pct": cycle_ret,
            "nav": nav,
            "n_trades": result.get("n_trades", 0),
            "error": result.get("error"),
        }
        nav_curve.append([eval_date_str, nav])
        ledger.append(cycle_ledger)
        emit("loop", date=eval_date_str, cycle=cycle_id,
             cycle_ret_pct=cycle_ret, nav=nav, n_picks=len(picks),
             n_factors=len(top_factors), elapsed_sec=time.time() - t_start)

        cycle_id += 1
        rebal_idx += REBALANCE_DAYS

    # Write result
    final_return_pct = (nav - 1) * 100
    result_obj = {
        "job_id": args.job_id,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "final_nav": nav,
        "final_return_pct": final_return_pct,
        "n_rebal_days": cycle_id,
        "n_days": n_total,
        "hold_days": HOLD_DAYS,
        "rebalance_days": REBALANCE_DAYS,
        "top_factors": STAGE5B_FACTORS[:TOP_FACTORS],
        "top_stocks": TOP_STOCKS,
        "initial_cash": INITIAL_CASH,
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "slippage_bps": SLIPPAGE["value"] * 10_000,
        "nav_curve": nav_curve,
        "ledger": ledger,
    }
    with open(job_dir / "result.json", "w") as f:
        json.dump(result_obj, f, indent=2, ensure_ascii=False, default=str)
    emit("done", final_nav=nav, n_cycles=cycle_id,
         final_return_pct=final_return_pct, duration_sec=time.time() - t_start)
    emit("result_written", result_file=str(job_dir / "result.json"))


if __name__ == "__main__":
    main()