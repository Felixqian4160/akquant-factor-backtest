"""V23-B: Bear Specialist — Rolling IC factor ranking (替代 V19 factor-return).

Strategy:
- For each bear signal date and each factor, compute the Spearman rank IC
  (correlation between per-stock factor value at date t and 20-day forward
  return t+1..t+21) across all stocks that day.
- Use only completed bear sessions whose exit_date is strictly before the
  current rebalance date, and only the last BEAR_LOOKBACK_SESSIONS of them.
- Per factor, mean IC across lookback bear sessions = rolling IC score.
- Top-N factors by mean IC vote for current stocks (top N by factor value).
- MIN_VOTES filter applies identically to V19.

Single variable vs V19-opt: ranking method (factor-return → rolling IC).
Lookback, MIN_VOTES, MIN_FACTOR_MEAN_RETURN, factor universe identical.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/media/felix/f/quant/aurumq-rl/src")
import akquant as aq  # noqa: E402

PANEL_FILE = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_021633/wavehunter_mainwave_features_v2.parquet"
)

ROUND_TRIP_COST = 0.005
BEAR_LOOKBACK_SESSIONS = 30
BEAR_IC_WINDOW_DAYS = 30  # rolling IC window (different from rank IC)
TOP_FACTORS = 10
TOP_STOCKS_PER_FACTOR = 10
MIN_VOTES = 3
MIN_STOCKS = 5
MAX_STOCKS = 10
BEAR_REBALANCE_SESSIONS = 20
MIN_FACTOR_OBS = 10
MIN_FACTOR_MEAN_RETURN = 0.005
FWD_DAYS = 20

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100

EXCLUDE_PREFIXES = (
    "v10_1_",
    "idx_",
    "alpha_alpha_custom_argmax_recent",
    "alpha_alpha_custom_argmin_recent",
)
BASE_COLUMNS = {
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    "adj_factor", "adj_close", "adj_factor_inferred", "pct_chg",
}

_DAILY_PICKS: dict[str, dict[str, float]] = {}
_DAILY_REGIME: dict[str, str] = {}
_BEAR_REBALANCE_DATES: set[str] = set()
FACTORS_RUNTIME: list[str] = []
BEAR_LOOKBACK_SESSIONS_RUNTIME: int = BEAR_LOOKBACK_SESSIONS


class BearFactorVoteStrategy(aq.Strategy):
    """AKQuant execution sleeve for the bear-factor experiment."""

    warmup = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()

    def on_bar(self, bar) -> None:
        pass

    def _close_all(self, date_str: str) -> None:
        for symbol in list(self._current_picks):
            try:
                self.close_position(symbol)
            except Exception as exc:
                self.log(f"close {symbol} failed: {exc}")
        self._current_picks = set()
        self.log(f"bear-factor sleeve -> cash on {date_str}")

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]
        regime = _DAILY_REGIME.get(date_str, "neutral")

        # This is a bear specialist, not a bull strategy. Keep the sleeve in
        # cash outside the strict bear router state.
        if regime != "bear":
            if self._current_picks:
                self._close_all(date_str)
            return

        # Hold between pre-registered bear rebalance dates.
        if date_str not in _BEAR_REBALANCE_DATES:
            return

        picks = _DAILY_PICKS.get(date_str, {})
        new_set = set(picks)
        if new_set == self._current_picks:
            return
        if not new_set:
            if self._current_picks:
                self._close_all(date_str)
            return

        try:
            self.rebalance_to_topn(
                scores=picks,
                top_n=len(new_set),
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
            self._current_picks = new_set
            self.log(f"bear-factor rebalance {date_str}: {len(new_set)} stocks")
        except Exception as exc:
            self.log(f"rebalance failed on {date_str}: {exc}")


def discover_factors(columns: list[str]) -> list[str]:
    return [
        c for c in columns
        if c not in BASE_COLUMNS and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
    ]


def parse_date(value) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.Timestamp(value).date()


def add_regime(date_panel: pl.DataFrame) -> pl.DataFrame:
    """Observable router: strict bear = 60d index return <= -5% and 20d <= 0."""
    date_state = (
        date_panel.select(["trade_date", "idx_ret_20d", "idx_ret_60d"])
        .group_by("trade_date")
        .agg([
            pl.col("idx_ret_20d").first(),
            pl.col("idx_ret_60d").first(),
        ])
        .sort("trade_date")
        .with_columns(
            pl.when(
                (pl.col("idx_ret_60d") <= -0.05)
                & (pl.col("idx_ret_20d") <= 0.0)
            )
            .then(pl.lit("bear"))
            .when(
                (pl.col("idx_ret_60d") > 0.05)
                & (pl.col("idx_ret_20d") > 0.0)
            )
            .then(pl.lit("bull"))
            .otherwise(pl.lit("neutral"))
            .alias("regime")
        )
    )
    return date_state


def build_bear_work(
    date_panel: pl.DataFrame,
    factors: list[str],
    history_start: date,
    end_date: date,
    emit,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[object, object]]:
    """Prepare per-stock bear-history work table.

    Returns:
        work: per-stock per-date rows for bear regime with _fwd_net pre-computed.
        date_state: daily regime + idx ret columns.
        exit_dates_map: {signal_date: exit_date} mapping for IC panel use.
    """
    date_state = add_regime(date_panel)
    work = date_panel.filter(
        (pl.col("trade_date") >= pl.lit(history_start))
        & (pl.col("trade_date") <= pl.lit(end_date + timedelta(days=FWD_DAYS + 30)))
    ).select(["trade_date", "ts_code", "open", "close", "idx_ret_20d", "idx_ret_60d"] + factors)

    all_dates = (
        work.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    )
    exit_dates_map: dict[object, object] = {}
    exit_records: list[dict] = []
    for i, signal_date in enumerate(all_dates):
        j = i + FWD_DAYS + 1
        exit_d = all_dates[j] if j < len(all_dates) else None
        exit_dates_map[signal_date] = exit_d
        exit_records.append({"trade_date": signal_date, "exit_date": exit_d})
    exit_df = pl.DataFrame(exit_records)

    work = (
        work.sort(["ts_code", "trade_date"])
        .with_columns(
            (
                pl.col("open").shift(-(FWD_DAYS + 1)).over("ts_code")
                / pl.col("open").shift(-1).over("ts_code")
                - 1.0
                - ROUND_TRIP_COST
            ).alias("_fwd_net")
        )
        .join(date_state.select(["trade_date", "regime"]), on="trade_date", how="left")
        .join(exit_df, on="trade_date", how="left")
        .filter(pl.col("regime") == "bear")
    )
    emit("bear_history_rows", rows=work.height, bear_dates=work["trade_date"].n_unique())
    return work, date_state, exit_dates_map


def _spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (handles ties via average ranks)."""
    n = len(x)
    if n < 5:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    rxm = rx - rx.mean()
    rym = ry - ry.mean()
    denom = np.sqrt((rxm ** 2).sum() * (rym ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((rxm * rym).sum() / denom)


def build_bear_ic_panel(
    work: pl.DataFrame,
    factors: list[str],
    emit,
) -> dict[date, dict[str, float]]:
    """Pre-compute Spearman IC per (bear_date, factor) for the entire history.

    Returns {date: {factor: ic}} — only dates with >=10 valid stocks per factor
    are recorded (others skipped). This avoids building a giant per-stock frame.
    Keys are normalized to datetime.date for matching with exit_dates_map.
    """
    pdf = work.select(["trade_date", "_fwd_net", "ts_code"] + factors).to_pandas()
    bear_dates = sorted(pdf["trade_date"].unique())
    t0 = time.time()
    out: dict[date, dict[str, float]] = {}
    for di, d in enumerate(bear_dates, 1):
        day = pdf[pdf["trade_date"] == d]
        if day.empty:
            continue
        y = day["_fwd_net"].to_numpy(dtype=float)
        per_factor_ic: dict[str, float] = {}
        for fac in factors:
            x = day[fac].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue
            ic = _spearman_ic(x[mask], y[mask])
            if np.isfinite(ic):
                per_factor_ic[fac] = ic
        if per_factor_ic:
            d_norm = d.date() if hasattr(d, "date") else d
            out[d_norm] = per_factor_ic
        if di % 20 == 0:
            emit("bear_ic_panel_progress", dates=di, total=len(bear_dates), elapsed_sec=time.time() - t0)
    emit("bear_ic_panel_ready", n_dates=len(out), duration_sec=time.time() - t0)
    return out


def rank_bear_factors_ic(
    ic_panel: dict[date, dict[str, float]],
    exit_dates: dict[date, date],
    current_date: date,
) -> list[dict]:
    """Rank factors by mean Spearman IC across the last BEAR_LOOKBACK_SESSIONS
    bear dates whose exit_date is strictly before current_date.
    """
    eligible: list[tuple[date, dict[str, float]]] = []
    for d, ic_map in ic_panel.items():
        d_norm = d.date() if hasattr(d, "date") else d
        if d_norm >= current_date:
            continue
        exit_d = exit_dates.get(d_norm)
        if exit_d is None:
            continue
        exit_d_norm = exit_d.date() if hasattr(exit_d, "date") else exit_d
        if exit_d_norm >= current_date:
            continue
        eligible.append((d_norm, ic_map))
    if len(eligible) < 5:
        return []
    eligible.sort(key=lambda x: x[0])
    eligible = eligible[-BEAR_LOOKBACK_SESSIONS_RUNTIME:]
    factor_ics: dict[str, list[float]] = {}
    for _, ic_map in eligible:
        for fac, ic in ic_map.items():
            factor_ics.setdefault(fac, []).append(ic)
    ranked: list[tuple[str, float, int]] = []
    for fac, ics in factor_ics.items():
        if len(ics) < 5:
            continue
        ranked.append((fac, float(np.nanmean(ics)), len(ics)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    out: list[dict] = []
    for fac, ic_mean, n_obs in ranked[:TOP_FACTORS]:
        out.append({
            "factor": fac,
            "direction": "high" if ic_mean >= 0 else "low",
            "score_ic_mean": ic_mean,
            "n_obs": n_obs,
        })
    return out


def bear_rebalance_dates(date_state: pl.DataFrame, start_date: date, end_date: date) -> list[date]:
    rows = date_state.filter(
        (pl.col("trade_date") >= pl.lit(start_date))
        & (pl.col("trade_date") <= pl.lit(end_date))
        & (pl.col("regime") == "bear")
    ).sort("trade_date")["trade_date"].to_list()
    result: list[date] = []
    previous = None
    count = BEAR_REBALANCE_SESSIONS
    for d in rows:
        # Reset the 20-session counter after a non-bear gap.
        if previous is None or d != previous:
            if previous is None:
                count = BEAR_REBALANCE_SESSIONS
        if count >= BEAR_REBALANCE_SESSIONS:
            result.append(d)
            count = 0
        count += 1
        previous = d
    return result


def rank_bear_factors(
    ic_panel: dict[date, dict[str, float]],
    exit_dates: dict[date, date],
    current_date: date,
) -> list[dict]:
    """V23-B: IC ranking instead of factor-return ranking."""
    return rank_bear_factors_ic(ic_panel, exit_dates, current_date)


def make_bear_picks(
    date_panel: pl.DataFrame,
    ic_panel: dict[date, dict[str, float]],
    exit_dates: dict[date, date],
    date_state: pl.DataFrame,
    rebalance_dates: list[date],
    emit,
) -> tuple[dict[str, dict[str, float]], list[dict], list[dict]]:
    picks: dict[str, dict[str, float]] = {}
    ranking_log: list[dict] = []
    vote_log: list[dict] = []
    current_rows = date_panel.select(["trade_date", "ts_code"] + FACTORS_RUNTIME)

    for d in rebalance_dates:
        chosen = rank_bear_factors(ic_panel, exit_dates, d)
        ranking_log.append({
            "date": str(d),
            "selected_factor_directions": [
                {
                    "factor": x["factor"],
                    "direction": x["direction"],
                    "score_ic_mean": x.get("score_ic_mean"),
                    "n_obs": x["n_obs"],
                }
                for x in chosen
            ],
        })
        if len(chosen) < 2:
            vote_log.append({"date": str(d), "status": "no_factor_consensus", "n_factors": len(chosen)})
            continue

        day = current_rows.filter(pl.col("trade_date") == pl.lit(d))
        votes: Counter[str] = Counter()
        reasons: defaultdict[str, list[str]] = defaultdict(list)
        for factor_row in chosen:
            f = factor_row["factor"]
            direction = factor_row["direction"]
            ranked = (
                day.filter(pl.col(f).is_not_null())
                .sort(f, descending=(direction == "high"))
                .head(TOP_STOCKS_PER_FACTOR)
            )
            for symbol in ranked["ts_code"].to_list():
                votes[symbol] += 1
                reasons[symbol].append(f"{f}:{direction}")

        qualified = [(symbol, count) for symbol, count in votes.items() if count >= MIN_VOTES]
        qualified.sort(key=lambda item: (item[1], item[0]), reverse=True)
        selected = qualified[:MAX_STOCKS]
        if len(selected) < MIN_STOCKS:
            vote_log.append({
                "date": str(d), "status": "no_stock_consensus",
                "n_factors": len(chosen), "n_qualified": len(qualified),
                "votes": dict(votes),
            })
            continue
        picks[str(d)] = {symbol: float(count) for symbol, count in selected}
        vote_log.append({
            "date": str(d), "status": "selected", "n_factors": len(chosen),
            "n_qualified": len(qualified), "selected": [s for s, _ in selected],
            "votes": {s: c for s, c in selected},
            "reasons": {s: reasons[s] for s, _ in selected},
        })
    emit("bear_picks_ready", rebalance_dates=len(rebalance_dates), picked_dates=len(picks))
    return picks, ranking_log, vote_log


def build_data_dict(raw_panel: pl.DataFrame, universe: list[str], start_date: date, end_date: date) -> dict[str, pd.DataFrame]:
    sub = raw_panel.filter(
        (pl.col("trade_date") >= pl.lit(datetime.combine(start_date, datetime.min.time())))
        & (pl.col("trade_date") <= pl.lit(datetime.combine(end_date, datetime.min.time())))
        & pl.col("ts_code").is_in(universe)
    ).select(["trade_date", "ts_code", "open", "high", "low", "close", "vol"])
    out: dict[str, pd.DataFrame] = {}
    for symbol in universe:
        pdf = (
            sub.filter(pl.col("ts_code") == symbol)
            .sort("trade_date")
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        if pdf.empty:
            continue
        pdf["symbol"] = symbol
        pdf = pdf.drop(columns=["ts_code"])
        # AKQuant requires non-zero volume for tradability.
        pdf["volume"] = 1.0e9
        out[symbol] = pdf
    return out


def extract_records(obj) -> list[dict]:
    if not isinstance(obj, list):
        return []
    output = []
    enum_types = {"OrderSide", "OrderStatus", "OrderType", "OrderRole", "PositionEffect", "TimeInForce"}
    for item in obj:
        if isinstance(item, dict):
            output.append(item)
            continue
        record = {}
        for attr in dir(item):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(item, attr)
            except Exception:
                continue
            if callable(value):
                continue
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            elif hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    value = str(value)
            elif type(value).__name__ in enum_types:
                value = str(value)
            record[attr] = value
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v23b")
    parser.add_argument("--bear-lookback-sessions", type=int, default=BEAR_LOOKBACK_SESSIONS)
    parser.add_argument("--min-votes", type=int, default=2)
    parser.add_argument("--min-factor-mean-return", type=float, default=0.0)
    args = parser.parse_args()

    # Declare all module globals up front so the parameter overrides are visible
    # to every helper closure (including emit and build_factor_return_history)
    # that reads them.
    global FACTORS_RUNTIME, _DAILY_REGIME, _BEAR_REBALANCE_DATES, _DAILY_PICKS
    global BEAR_LOOKBACK_SESSIONS_RUNTIME, MIN_VOTES, MIN_FACTOR_MEAN_RETURN
    MIN_VOTES = args.min_votes
    MIN_FACTOR_MEAN_RETURN = args.min_factor_mean_return
    BEAR_LOOKBACK_SESSIONS_RUNTIME = args.bear_lookback_sessions

    raw = pl.read_parquet(PANEL_FILE)
    date_panel = raw.with_columns(pl.col("trade_date").dt.date())
    FACTORS_RUNTIME = discover_factors(date_panel.columns)
    if not FACTORS_RUNTIME:
        raise RuntimeError("no factor columns after leakage exclusions")

    start_date = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date[:10], "%Y-%m-%d").date()
    history_start = start_date - timedelta(days=800)
    job_dir = Path(args.out_base) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"

    def emit(stage: str, **kwargs) -> None:
        row = {"stage": stage, "job_id": args.job_id, "timestamp": datetime.now().isoformat(), **kwargs}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    emit(
        "init", start_date=args.start_date, end_date=args.end_date,
        n_factors=len(FACTORS_RUNTIME), fwd_days=FWD_DAYS,
        bear_lookback_sessions=args.bear_lookback_sessions,
        top_factors=TOP_FACTORS, top_stocks_per_factor=TOP_STOCKS_PER_FACTOR,
        min_votes=args.min_votes, min_stocks=MIN_STOCKS, max_stocks=MAX_STOCKS,
        bear_rebalance_sessions=BEAR_REBALANCE_SESSIONS,
        min_factor_mean_return=args.min_factor_mean_return, lot_size=LOT_SIZE,
    )

    work, date_state, exit_dates_map = build_bear_work(
        date_panel, FACTORS_RUNTIME, history_start, end_date, emit
    )
    ic_panel = build_bear_ic_panel(work, FACTORS_RUNTIME, emit)
    rebalance_dates = bear_rebalance_dates(date_state, start_date, end_date)

    _DAILY_REGIME = {
        str(row["trade_date"]): row["regime"]
        for row in date_state.filter(
            (pl.col("trade_date") >= pl.lit(start_date))
            & (pl.col("trade_date") <= pl.lit(end_date))
        ).to_dicts()
    }
    _BEAR_REBALANCE_DATES = {str(x) for x in rebalance_dates}
    _DAILY_PICKS, ranking_log, vote_log = make_bear_picks(
        date_panel.filter(
            (pl.col("trade_date") >= pl.lit(start_date))
            & (pl.col("trade_date") <= pl.lit(end_date))
        ),
        ic_panel,
        exit_dates_map,
        date_state,
        rebalance_dates,
        emit,
    )

    universe = sorted({s for p in _DAILY_PICKS.values() for s in p})
    emit("universe_ready", n_symbols=len(universe))
    data = build_data_dict(raw, universe, start_date - timedelta(days=30), end_date)
    emit("data_ready", n_symbols=len(data))
    if not data:
        raise RuntimeError("no AKQuant data for bear picks")

    emit("akquant_start")
    result = aq.run_backtest(
        data=data,
        strategy=BearFactorVoteStrategy,
        initial_cash=INITIAL_CASH,
        commission_rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        t_plus_one=False,
        fill_policy=aq.NextOpen(),
        lot_size=LOT_SIZE,
    )
    metrics = result.metrics_df
    metrics_dict = {}
    for index in metrics.index:
        value = metrics.loc[index, "value"]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        try:
            metrics_dict[index] = float(value)
        except (TypeError, ValueError):
            metrics_dict[index] = str(value)
    emit("akquant_done")

    nav_curve = []
    if hasattr(result.equity_curve, "items"):
        nav_curve = [[str(ts)[:10], value] for ts, value in result.equity_curve.items()]
    final_value = float(metrics_dict.get("end_market_value", INITIAL_CASH))
    trades = extract_records(result.trades)
    orders = extract_records(result.orders)
    regime_counts = Counter(_DAILY_REGIME.values())
    result_obj = {
        "job_id": args.job_id,
        "contract": {
            "type": "v23b_bear_specialist_rolling_ic_rank_vote",
            "ranking_method": "rolling spearman IC across last BEAR_LOOKBACK_SESSIONS bear signal dates",
            "factor_score": "mean per-day Spearman IC across lookback bear sessions",
            "direction": "positive IC=high, negative IC=low",
            "entry": "T+1 raw open",
            "exit": "T+21 raw open",
            "round_trip_cost": ROUND_TRIP_COST,
            "bear_router": "idx_ret_60d <= -5% AND idx_ret_20d <= 0",
            "outside_bear": "cash",
            "rebalance_sessions": BEAR_REBALANCE_SESSIONS,
            "min_votes": MIN_VOTES,
            "lot_size": LOT_SIZE,
        },
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_factors": len(FACTORS_RUNTIME),
        "factors": FACTORS_RUNTIME,
        "regime_counts": dict(regime_counts),
        "n_bear_rebalance_dates": len(rebalance_dates),
        "n_pick_dates": len(_DAILY_PICKS),
        "avg_stocks_per_pick_date": float(np.mean([len(x) for x in _DAILY_PICKS.values()])) if _DAILY_PICKS else 0.0,
        "final_value": final_value,
        "initial_cash": INITIAL_CASH,
        "final_return_pct": (final_value / INITIAL_CASH - 1.0) * 100.0,
        "metrics": metrics_dict,
        "n_trades": len(trades),
        "n_orders": len(orders),
        "nav_curve": nav_curve,
        "daily_regime": [
            {"date": str(row["trade_date"]), "regime": row["regime"],
             "idx_ret_20d": row["idx_ret_20d"], "idx_ret_60d": row["idx_ret_60d"]}
            for row in date_state.filter(
                (pl.col("trade_date") >= pl.lit(start_date))
                & (pl.col("trade_date") <= pl.lit(end_date))
            ).to_dicts()
        ],
        "factor_ranking_log": ranking_log,
        "vote_log": vote_log,
        "trades_ledger": trades,
        "orders_ledger": orders,
    }
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    emit(
        "done", final_value=final_value, final_return_pct=result_obj["final_return_pct"],
        n_trades=len(trades), n_orders=len(orders), result_file=str(result_path),
    )


if __name__ == "__main__":
    main()
