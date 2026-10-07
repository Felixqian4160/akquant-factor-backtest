"""Bear-regime factor-return ranking + consensus voting (v19).

Purpose
-------
The IC selector is not sufficient during a falling market: a cross-sectional
IC can remain positive while every long position loses money. This experiment
uses a different, executable selector only in the bear regime:

1. For every factor and both directions (high/low), build a top/bottom-10
   portfolio on each historical bear signal date.
2. The portfolio return is T+1 raw open -> T+21 raw open minus 0.5% round-trip
   cost. This is the same economic timing used by the AKQuant simulation.
3. At a current bear date, use only *completed* historical bear outcomes from
   the last BEAR_LOOKBACK_SESSIONS bear signal dates. Rank each factor by its
   mean historical net portfolio return.
4. Keep the better direction per factor. ``high`` is a strength direction;
   ``low`` is a potential oversold/reversal direction, determined by the
   historical return rather than hard-coded semantics.
5. The top factor directions vote for current stocks. Stocks with fewer than
   MIN_VOTES votes are rejected; at least MIN_STOCKS and at most MAX_STOCKS are
   held. Rebalance every BEAR_REBALANCE_SESSIONS bear sessions.
6. Outside bear regime this diagnostic sleeve is in cash. The output is not a
   production approval; it is a bear-specialist experiment to be compared
   against the existing cash/IC baseline.

Leakage guard
-------------
A factor-return observation generated on signal date S is eligible for current
selection date T only when its exit date is strictly before T. Current stock
selection uses factor values on T only. No current/future forward return is
used to choose factors or stocks.
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

# Immutable experiment contract.
FWD_DAYS = 20
ROUND_TRIP_COST = 0.005  # 25 bps each side; slippage is passed separately to AKQuant.
BEAR_LOOKBACK_SESSIONS = 60
TOP_FACTORS = 10
TOP_STOCKS_PER_FACTOR = 10
MIN_VOTES = 2
MIN_STOCKS = 5
MAX_STOCKS = 10
BEAR_REBALANCE_SESSIONS = 20
MIN_FACTOR_OBS = 10
MIN_FACTOR_MEAN_RETURN = 0.0
INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100

EXCLUDE_PREFIXES = (
    "v10_1_",  # retrospective labels / zigzag fields
    "idx_",  # index-level values are not stock selectors
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


def build_factor_return_history(
    date_panel: pl.DataFrame,
    factors: list[str],
    history_start: date,
    end_date: date,
    emit,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build historical executable top/bottom factor portfolio returns.

    Returns:
      factor_returns: signal_date, exit_date, factor, direction, mean_net_return,
                      win_rate, n_stocks.
      date_state: date-level regime and index-return router values.
    """
    date_state = add_regime(date_panel)
    work = date_panel.filter(
        (pl.col("trade_date") >= pl.lit(history_start))
        & (pl.col("trade_date") <= pl.lit(end_date + timedelta(days=FWD_DAYS + 30)))
    ).select(["trade_date", "ts_code", "open", "close", "idx_ret_20d", "idx_ret_60d"] + factors)

    # The signal-to-exit mapping is session based, not calendar based.
    all_dates = (
        work.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    )
    exit_map = []
    for i, signal_date in enumerate(all_dates):
        j = i + FWD_DAYS + 1  # T+1 open -> T+21 open for a 20-session hold
        exit_map.append({
            "trade_date": signal_date,
            "exit_date": all_dates[j] if j < len(all_dates) else None,
        })
    exit_df = pl.DataFrame(exit_map, schema_overrides={"trade_date": date_panel.schema["trade_date"]})

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

    records: list[pl.DataFrame] = []
    t0 = time.time()
    for index, factor in enumerate(factors, 1):
        # Keep each factor's ranking calculation isolated to limit memory.
        selected = work.select(["trade_date", "exit_date", "ts_code", "_fwd_net", factor]).drop_nulls(
            [factor, "_fwd_net", "exit_date"]
        )
        if selected.is_empty():
            continue
        ranked = selected.with_columns([
            pl.col(factor).rank(method="ordinal", descending=True).over("trade_date").alias("_rank_high"),
            pl.col(factor).rank(method="ordinal", descending=False).over("trade_date").alias("_rank_low"),
            pl.col(factor).count().over("trade_date").alias("_n_day"),
        ])
        for direction, rank_column in (("high", "_rank_high"), ("low", "_rank_low")):
            daily = (
                ranked.filter(pl.col(rank_column) <= TOP_STOCKS_PER_FACTOR)
                .group_by("trade_date")
                .agg([
                    pl.col("_fwd_net").mean().alias("mean_net_return"),
                    (pl.col("_fwd_net") > 0).mean().alias("win_rate"),
                    pl.len().alias("n_stocks"),
                    pl.col("exit_date").first().alias("exit_date"),
                ])
                .with_columns([
                    pl.lit(factor).alias("factor"),
                    pl.lit(direction).alias("direction"),
                ])
                .select([
                    "trade_date", "exit_date", "factor", "direction",
                    "mean_net_return", "win_rate", "n_stocks",
                ])
            )
            records.append(daily)
        if index % 50 == 0:
            emit("factor_returns_progress", completed=index, total=len(factors), elapsed_sec=time.time() - t0)

    if not records:
        raise RuntimeError("no historical bear factor-return records")
    factor_returns = pl.concat(records).sort(["trade_date", "factor", "direction"])
    emit("factor_returns_ready", rows=factor_returns.height, duration_sec=time.time() - t0)
    return factor_returns, date_state


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
    factor_returns: pl.DataFrame,
    current_date: date,
) -> list[dict]:
    """Rank directions by completed historical bear factor-portfolio returns."""
    eligible = factor_returns.filter(
        (pl.col("trade_date") < pl.lit(current_date))
        & (pl.col("exit_date") < pl.lit(current_date))
    )
    if eligible.is_empty():
        return []
    hist_dates = eligible["trade_date"].unique().sort().to_list()
    hist_dates = hist_dates[-BEAR_LOOKBACK_SESSIONS_RUNTIME:]
    eligible = eligible.filter(pl.col("trade_date").is_in(hist_dates))
    scored = (
        eligible.group_by(["factor", "direction"])
        .agg([
            pl.col("mean_net_return").mean().alias("score_mean_return"),
            pl.col("mean_net_return").median().alias("score_median_return"),
            pl.col("win_rate").mean().alias("score_win_rate"),
            pl.len().alias("n_obs"),
        ])
        .filter(pl.col("n_obs") >= MIN_FACTOR_OBS)
    )
    if scored.is_empty():
        return []
    # Pick the better high/low direction once per factor; then rank factors.
    rows = scored.sort(
        ["factor", "score_mean_return", "score_win_rate"],
        descending=[False, True, True],
    ).to_dicts()
    best_by_factor: dict[str, dict] = {}
    for row in rows:
        if row["factor"] not in best_by_factor:
            best_by_factor[row["factor"]] = row
    chosen = sorted(
        best_by_factor.values(),
        key=lambda r: (r["score_mean_return"], r["score_win_rate"], r["n_obs"]),
        reverse=True,
    )
    return [r for r in chosen if r["score_mean_return"] > MIN_FACTOR_MEAN_RETURN][:TOP_FACTORS]


def make_bear_picks(
    date_panel: pl.DataFrame,
    factor_returns: pl.DataFrame,
    date_state: pl.DataFrame,
    rebalance_dates: list[date],
    emit,
) -> tuple[dict[str, dict[str, float]], list[dict], list[dict]]:
    picks: dict[str, dict[str, float]] = {}
    ranking_log: list[dict] = []
    vote_log: list[dict] = []
    current_rows = date_panel.select(["trade_date", "ts_code"] + FACTORS_RUNTIME)

    for d in rebalance_dates:
        chosen = rank_bear_factors(factor_returns, d)
        ranking_log.append({
            "date": str(d),
            "selected_factor_directions": [
                {
                    "factor": x["factor"],
                    "direction": x["direction"],
                    "score_mean_return": x["score_mean_return"],
                    "score_median_return": x["score_median_return"],
                    "score_win_rate": x["score_win_rate"],
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
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v19")
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

    factor_returns, date_state = build_factor_return_history(
        date_panel, FACTORS_RUNTIME, history_start, end_date, emit
    )
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
        factor_returns,
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
            "type": "bear_specialist_factor_return_rank_vote",
            "factor_return": "top/bottom-10 stock portfolio mean net return",
            "entry": "T+1 raw open",
            "exit": "T+21 raw open",
            "round_trip_cost": ROUND_TRIP_COST,
            "factor_score": "mean completed historical bear portfolio return over last 60 bear signal sessions",
            "direction": "better high/low direction per factor; high=strength, low=oversold/reversal candidate",
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
    (job_dir / "factor_returns.parquet").write_bytes(factor_returns.write_parquet(None)) if False else factor_returns.write_parquet(job_dir / "factor_returns.parquet")
    emit(
        "done", final_value=final_value, final_return_pct=result_obj["final_return_pct"],
        n_trades=len(trades), n_orders=len(orders), result_file=str(result_path),
    )


if __name__ == "__main__":
    main()
