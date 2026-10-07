"""V20 Hybrid: Bull/Neutral use V14-style rolling-IC voting, Bear uses V19-style factor-return voting.

Strategy:
- Bull/Neutral (idx_ret_60d > -5% OR idx_ret_20d > 0):
  - Every daily rebalance:
      1. Compute per-factor rolling Spearman IC against 20-day forward return
         over the last IC_WINDOW_DAYS sessions (default 20).
      2. Rank factors by rolling IC and keep the top TOP_FACTORS_BULL.
      3. Each kept factor votes for its top TOP_STOCKS_PER_FACTOR stocks
         (cross-section rank, descending factor value).
      4. Stocks with >= MIN_VOTES_BULL votes enter the daily pool.
- Bear (idx_ret_60d <= -5% AND idx_ret_20d <= 0):
  - Re-balance only every BEAR_REBALANCE_SESSIONS bear sessions.
      1. For every factor and both directions (high/low), build a top/bottom
         portfolio on every historical bear signal date; mean T+1 open -> T+21
         open minus 0.5% round-trip cost.
      2. At the current bear rebalance date, use only completed bear
         outcomes whose exit_date is strictly before the current date and that
         lie in the last BEAR_LOOKBACK_SESSIONS bear signal sessions.
      3. Per factor, keep the better high/low direction. Rank factors by mean
         completed bear portfolio return; require mean return >
         MIN_FACTOR_MEAN_RETURN.
      4. Each chosen factor votes for its top TOP_STOCKS_PER_FACTOR stocks
         (cross-section rank, descending if direction == high else ascending).
      5. Stocks with >= MIN_VOTES_BEAR votes enter the daily pool.
- Outside the bull/neutral pool and bear pool: keep current positions,
  otherwise go to cash on a regime switch.

Leakage guard: forward returns are only consulted for fully completed past
bear portfolios; current-day stock ranking uses only current-day factor
values.
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

# Bull side (V14 style)
FWD_DAYS = 20
IC_WINDOW_DAYS = 20
TOP_FACTORS_BULL = 10
TOP_STOCKS_PER_FACTOR_BULL = 10
MIN_VOTES_BULL = 2
MAX_STOCKS_BULL = 10

# Bear side (V19 style)
ROUND_TRIP_COST = 0.005
BEAR_LOOKBACK_SESSIONS = 30
TOP_FACTORS_BEAR = 10
TOP_STOCKS_PER_FACTOR_BEAR = 10
MIN_VOTES_BEAR = 3
MIN_STOCKS_BEAR = 5
MAX_STOCKS_BEAR = 10
BEAR_REBALANCE_SESSIONS = 10
MIN_FACTOR_OBS = 10
MIN_FACTOR_MEAN_RETURN = 0.005

# Execution
INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
MIN_HOLD_DAYS = 10

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

# Module-level state populated by main(). The strategy reads only these dicts.
_DAILY_PICKS_BULL: dict[str, dict[str, float]] = {}
_DAILY_PICKS_BEAR: dict[str, dict[str, float]] = {}
_BEAR_REBALANCE_DATES: set[str] = set()
_DAILY_REGIME: dict[str, str] = {}


class V20HybridStrategy(aq.Strategy):
    """Single AKQuant execution sleeve. Daily rebalance for bull/neutral,
    20 bear-session rebalance for bear."""

    warmup = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._current_picks: set[str] = set()
        self._entry_dates: dict[str, object] = {}

    def on_bar(self, bar) -> None:
        pass

    def _close_all(self, date_str: str, reason: str) -> None:
        for symbol in list(self._current_picks):
            try:
                self.close_position(symbol)
            except Exception as exc:
                self.log(f"close {symbol} failed: {exc}")
        self._current_picks = set()
        self._entry_dates = {}
        if reason:
            self.log(f"{reason} on {date_str}")

    def _entry_age(self, symbol: str, current_date_str: str) -> int:
        """Trading days elapsed since entry. Approximate by date-string diff."""
        try:
            entry = self._entry_dates[symbol]
            return self._session_diff(entry, current_date_str)
        except Exception:
            return MIN_HOLD_DAYS + 1

    def _session_diff(self, start_str: str, end_str: str) -> int:
        try:
            from datetime import datetime as _dt
            s = _dt.strptime(start_str[:10], "%Y-%m-%d")
            e = _dt.strptime(end_str[:10], "%Y-%m-%d")
            return max(0, (e - s).days)
        except Exception:
            return 0

    def on_cross_section(self, trading_date, timestamp) -> None:
        date_str = str(trading_date)[:10]
        regime = _DAILY_REGIME.get(date_str, "bull")

        if regime == "bear":
            # Only rebalance on registered bear sessions.
            if date_str not in _BEAR_REBALANCE_DATES:
                return
            picks = _DAILY_PICKS_BEAR.get(date_str, {})
        else:
            picks = _DAILY_PICKS_BULL.get(date_str, {})

        new_set = set(picks)
        if not new_set:
            if self._current_picks:
                self._close_all(date_str, f"empty picks in {regime}")
            return

        # MIN_HOLD_DAYS guard: symbols currently held but with insufficient age
        # are sticky. We do NOT add new symbols if the candidate would require
        # selling a sticky one.
        sticky = {
            sym for sym in self._current_picks
            if self._entry_age(sym, date_str) < MIN_HOLD_DAYS
        }
        final_set = (new_set | sticky) - (self._current_picks - new_set - sticky)
        # Remove only those that (a) not in new_set and (b) not sticky
        removable = self._current_picks - new_set - sticky
        if removable:
            final_set = final_set - removable

        if final_set == self._current_picks:
            return

        # Ensure we have a top_n that covers everything we want
        scores: dict[str, float] = {}
        for sym in final_set:
            if sym in picks:
                scores[sym] = float(picks[sym])
            else:
                scores[sym] = 1.0  # sticky symbol, weight placeholder

        if not scores:
            return

        try:
            self.rebalance_to_topn(
                scores=scores,
                top_n=len(scores),
                weight_mode="equal",
                long_only=True,
                liquidate_unmentioned=True,
            )
            # Track entry dates for newly added symbols
            for sym in final_set - self._current_picks:
                self._entry_dates[sym] = date_str
            for sym in self._current_picks - final_set:
                self._entry_dates.pop(sym, None)
            self._current_picks = final_set
            self.log(
                f"v22 rebalance {date_str} regime={regime} "
                f"stocks={len(final_set)} sticky={len(sticky)}"
            )
        except Exception as exc:
            self.log(f"rebalance failed on {date_str}: {exc}")


def discover_factors(columns: list[str]) -> list[str]:
    return [
        c for c in columns
        if c not in BASE_COLUMNS and not any(c.startswith(p) for p in EXCLUDE_PREFIXES)
    ]


# ----- Regime / bear router -----
def add_regime(date_panel: pl.DataFrame) -> pl.DataFrame:
    return (
        date_panel.select(["trade_date", "idx_ret_20d", "idx_ret_60d"])
        .group_by("trade_date")
        .agg([pl.col("idx_ret_20d").first(), pl.col("idx_ret_60d").first()])
        .sort("trade_date")
        .with_columns(
            pl.when(
                (pl.col("idx_ret_60d") <= -0.05)
                & (pl.col("idx_ret_20d") <= 0.0)
            )
            .then(pl.lit("bear"))
            .otherwise(pl.lit("bull_neutral"))
            .alias("regime")
        )
    )


def bear_rebalance_dates(date_state: pl.DataFrame, start_date: date, end_date: date) -> list[date]:
    rows_filtered = date_state.filter(
        (pl.col("trade_date") >= pl.lit(start_date))
        & (pl.col("trade_date") <= pl.lit(end_date))
        & (pl.col("regime") == "bear")
    ).sort("trade_date")["trade_date"].to_list()
    result: list[date] = []
    count = 0
    for d in rows_filtered:
        if count >= BEAR_REBALANCE_SESSIONS:
            result.append(d)
            count = 0
        count += 1
    return result


# ----- Bear side (V19 style) -----
def build_factor_return_history(
    date_panel: pl.DataFrame,
    factors: list[str],
    history_start: date,
    end_date: date,
    emit,
) -> pl.DataFrame:
    date_state = add_regime(date_panel)
    work = date_panel.filter(
        (pl.col("trade_date") >= pl.lit(history_start))
        & (pl.col("trade_date") <= pl.lit(end_date + timedelta(days=FWD_DAYS + 30)))
    ).select(["trade_date", "ts_code", "open", "close", "idx_ret_20d", "idx_ret_60d"] + factors)

    all_dates = work.select("trade_date").unique().sort("trade_date")["trade_date"].to_list()
    exit_map = []
    for i, signal_date in enumerate(all_dates):
        j = i + FWD_DAYS + 1
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
        selected = work.select(["trade_date", "exit_date", "ts_code", "_fwd_net", factor]).drop_nulls(
            [factor, "_fwd_net", "exit_date"]
        )
        if selected.is_empty():
            continue
        ranked = selected.with_columns([
            pl.col(factor).rank(method="ordinal", descending=True).over("trade_date").alias("_rank_high"),
            pl.col(factor).rank(method="ordinal", descending=False).over("trade_date").alias("_rank_low"),
        ])
        for direction, rank_column in (("high", "_rank_high"), ("low", "_rank_low")):
            daily = (
                ranked.filter(pl.col(rank_column) <= TOP_STOCKS_PER_FACTOR_BEAR)
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
        if index % 100 == 0:
            emit("factor_returns_progress", completed=index, total=len(factors), elapsed_sec=time.time() - t0)
    if not records:
        raise RuntimeError("no historical bear factor-return records")
    factor_returns = pl.concat(records).sort(["trade_date", "factor", "direction"])
    emit("factor_returns_ready", rows=factor_returns.height, duration_sec=time.time() - t0)
    return factor_returns


def rank_bear_factors(
    factor_returns: pl.DataFrame,
    current_date: date,
) -> list[dict]:
    eligible = factor_returns.filter(
        (pl.col("trade_date") < pl.lit(current_date))
        & (pl.col("exit_date") < pl.lit(current_date))
    )
    if eligible.is_empty():
        return []
    hist_dates = eligible["trade_date"].unique().sort().to_list()
    hist_dates = hist_dates[-BEAR_LOOKBACK_SESSIONS:]
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
    return [r for r in chosen if r["score_mean_return"] > MIN_FACTOR_MEAN_RETURN][:TOP_FACTORS_BEAR]


def make_bear_picks(
    date_panel: pl.DataFrame,
    factor_returns: pl.DataFrame,
    rebalance_dates: list[date],
    emit,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    picks: dict[str, dict[str, float]] = {}
    vote_log: list[dict] = []
    current_rows = date_panel.select(["trade_date", "ts_code"] + FACTORS_RUNTIME)

    for d in rebalance_dates:
        chosen = rank_bear_factors(factor_returns, d)
        if len(chosen) < 2:
            vote_log.append({"date": str(d), "status": "no_factor_consensus", "n_factors": len(chosen)})
            continue

        day = current_rows.filter(pl.col("trade_date") == pl.lit(d))
        votes: Counter[str] = Counter()
        for factor_row in chosen:
            f = factor_row["factor"]
            direction = factor_row["direction"]
            ranked = (
                day.filter(pl.col(f).is_not_null())
                .sort(f, descending=(direction == "high"))
                .head(TOP_STOCKS_PER_FACTOR_BEAR)
            )
            for symbol in ranked["ts_code"].to_list():
                votes[symbol] += 1

        qualified = [(symbol, count) for symbol, count in votes.items() if count >= MIN_VOTES_BEAR]
        qualified.sort(key=lambda item: (item[1], item[0]), reverse=True)
        selected = qualified[:MAX_STOCKS_BEAR]
        if len(selected) < MIN_STOCKS_BEAR:
            vote_log.append({
                "date": str(d), "status": "no_stock_consensus",
                "n_factors": len(chosen), "n_qualified": len(qualified),
            })
            continue
        picks[str(d)] = {symbol: float(count) for symbol, count in selected}
        vote_log.append({
            "date": str(d), "status": "selected", "n_factors": len(chosen),
            "n_qualified": len(qualified), "selected": [s for s, _ in selected],
        })
    emit("bear_picks_ready", rebalance_dates=len(rebalance_dates), picked_dates=len(picks))
    return picks, vote_log


# ----- Bull side (V14 style: daily rolling IC voting) -----
def compute_rolling_ic(sub_panel: pl.DataFrame, factors: list[str], fwd_days: int, window: int) -> pl.DataFrame:
    sub_panel = sub_panel.with_columns(
        (pl.col("close").shift(-fwd_days).over("ts_code") / pl.col("close") - 1).alias("_fwd_ret")
    )
    sub_panel = sub_panel.sort(["trade_date", "ts_code"])
    rank_exprs = [
        pl.col(f).rank(method="average").over("trade_date").alias(f"_f_rank_{f}")
        for f in factors
    ]
    rank_exprs.append(
        pl.col("_fwd_ret").rank(method="average").over("trade_date").alias("_y_rank")
    )
    sub_panel = sub_panel.with_columns(rank_exprs)

    ic_records = []
    for f in factors:
        agg = (
            sub_panel.group_by("trade_date")
            .agg(pl.corr(pl.col(f"_f_rank_{f}"), pl.col("_y_rank")).alias("_ic"))
            .with_columns(pl.lit(f).alias("factor"))
            .select(["trade_date", "factor", "_ic"])
        )
        ic_records.append(agg)
    ic_df = pl.concat(ic_records).sort(["factor", "trade_date"])
    ic_df = ic_df.with_columns(
        pl.col("_ic").rolling_mean(window).over("factor").alias("_rolling_ic")
    )
    return ic_df


def make_bull_picks(
    date_panel: pl.DataFrame,
    factors: list[str],
    fwd_days: int,
    ic_window: int,
    start_date: date,
    end_date: date,
    history_start: date,
    emit,
) -> dict[str, dict[str, float]]:
    sub_all = date_panel.filter(
        (pl.col("trade_date") >= pl.lit(history_start))
        & (pl.col("trade_date") <= pl.lit(end_date + timedelta(days=fwd_days + 30)))
    ).select(["trade_date", "ts_code", "close"] + factors)

    sub_all = sub_all.sort(["ts_code", "trade_date"])
    sub_all = sub_all.with_columns(
        (pl.col("close").shift(-fwd_days).over("ts_code") / pl.col("close") - 1).alias("_fwd_ret")
    )

    emit("computing_rolling_ic", factors=len(factors), window=ic_window, fwd_days=fwd_days)
    t0 = time.time()
    ic_df = compute_rolling_ic(sub_all, factors, fwd_days, ic_window)
    emit("rolling_ic_done", duration_sec=time.time() - t0)

    # Daily top factors
    warmup_threshold = start_date + timedelta(days=10)
    sub_picks_window = (
        ic_df.filter(pl.col("trade_date") >= pl.lit(warmup_threshold))
        .filter(pl.col("trade_date") <= pl.lit(end_date))
        .with_columns(
            pl.col("_rolling_ic").rank(method="ordinal", descending=True).over("trade_date").alias("_factor_rank")
        )
    )
    top_factors_per_day = sub_picks_window.filter(pl.col("_factor_rank") <= TOP_FACTORS_BULL)
    emit("top_bull_factors_per_day", n_rows=top_factors_per_day.height)

    sub_data_window = sub_all.filter(
        (pl.col("trade_date") >= pl.lit(start_date))
        & (pl.col("trade_date") <= pl.lit(end_date))
    )

    rank_exprs = [
        pl.col(f).rank(method="ordinal", descending=True).over("trade_date").alias(f"_rank_{f}")
        for f in factors
    ]
    sub_data_window = sub_data_window.with_columns(rank_exprs)

    per_factor_top_dfs = []
    for f in factors:
        rank_col = f"_rank_{f}"
        top_df = (
            sub_data_window.filter(pl.col(rank_col) <= TOP_STOCKS_PER_FACTOR_BULL)
            .select(pl.col("trade_date").alias("date"), pl.col("ts_code").alias("symbol"))
            .with_columns(pl.lit(f).alias("factor"))
        )
        per_factor_top_dfs.append(top_df)
    votes_long = pl.concat(per_factor_top_dfs)

    if votes_long.schema["date"] == pl.Utf8:
        votes_long = votes_long.with_columns(
            pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        )
    top_factors_per_day_for_join = (
        top_factors_per_day.rename({"trade_date": "date"}).select(["date", "factor"])
    )
    if top_factors_per_day_for_join.schema["date"] == pl.Utf8:
        top_factors_per_day_for_join = top_factors_per_day_for_join.with_columns(
            pl.col("date").str.strptime(pl.Datetime("ms"), "%Y-%m-%d")
        )
    filtered_votes = votes_long.join(
        top_factors_per_day_for_join, on=["date", "factor"], how="inner"
    )
    qualified = (
        filtered_votes.group_by(["date", "symbol"])
        .agg(pl.len().alias("votes"))
        .filter(pl.col("votes") >= MIN_VOTES_BULL)
    )
    qualified = qualified.with_columns(
        pl.col("votes").rank(method="ordinal", descending=True).over("date").alias("_date_rank")
    )
    qualified = qualified.filter(pl.col("_date_rank") <= MAX_STOCKS_BULL)

    picks: dict[str, dict[str, float]] = {}
    for row in qualified.sort(["date", "votes"], descending=[False, True]).to_dicts():
        d_raw = row["date"]
        d = d_raw.strftime("%Y-%m-%d") if hasattr(d_raw, "strftime") else str(d_raw)[:10]
        sym = row["symbol"]
        v = int(row["votes"])
        if d not in picks:
            picks[d] = {}
        picks[d][sym] = float(v)
    emit("bull_picks_ready", picked_dates=len(picks))
    return picks


# ----- Universe + data plumbing -----
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
    parser.add_argument("--out-base", default="evidence/daily_factor_reselect/pipeline_v25a")
    parser.add_argument("--ic-window-days", type=int, default=IC_WINDOW_DAYS)
    args = parser.parse_args()

    global FACTORS_RUNTIME
    raw = pl.read_parquet(PANEL_FILE)
    date_panel = raw.with_columns(pl.col("trade_date").dt.date())
    FACTORS_RUNTIME = discover_factors(date_panel.columns)
    if not FACTORS_RUNTIME:
        raise RuntimeError("no factor columns after leakage exclusions")

    start_date = datetime.strptime(args.start_date[:10], "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date[:10], "%Y-%m-%d").date()
    history_start = start_date - timedelta(days=max(args.ic_window_days, BEAR_LOOKBACK_SESSIONS) * 4 + 60)
    job_dir = Path(args.out_base) / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    progress_path = job_dir / "progress.jsonl"

    def emit(stage: str, **kwargs) -> None:
        row = {"stage": stage, "job_id": args.job_id, "timestamp": datetime.now().isoformat(), **kwargs}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)

    emit(
        "init",
        start_date=args.start_date, end_date=args.end_date,
        n_factors=len(FACTORS_RUNTIME),
        bull_top_factors=TOP_FACTORS_BULL,
        bull_top_stocks=TOP_STOCKS_PER_FACTOR_BULL,
        bull_min_votes=MIN_VOTES_BULL,
        bull_ic_window=args.ic_window_days,
        bear_lookback_sessions=BEAR_LOOKBACK_SESSIONS,
        bear_top_factors=TOP_FACTORS_BEAR,
        bear_top_stocks=TOP_STOCKS_PER_FACTOR_BEAR,
        bear_min_votes=MIN_VOTES_BEAR,
        bear_min_factor_mean_return=MIN_FACTOR_MEAN_RETURN,
        bear_rebalance_sessions=BEAR_REBALANCE_SESSIONS,
        lot_size=LOT_SIZE,
    )

    date_state = add_regime(date_panel)
    regime_records_in_window = date_state.filter(
        (pl.col("trade_date") >= pl.lit(start_date))
        & (pl.col("trade_date") <= pl.lit(end_date))
    ).to_dicts()
    global _DAILY_REGIME, _BEAR_REBALANCE_DATES
    _DAILY_REGIME = {str(row["trade_date"]): row["regime"] for row in regime_records_in_window}
    _BEAR_REBALANCE_DATES = {
        str(d) for d in bear_rebalance_dates(date_state, start_date, end_date)
    }
    regime_counter = Counter(_DAILY_REGIME.values())
    emit("regime_summary", regime_counts=dict(regime_counter), n_bear_rebalance_dates=len(_BEAR_REBALANCE_DATES))

    # Bull picks (V14 style)
    bull_picks = make_bull_picks(
        date_panel, FACTORS_RUNTIME, FWD_DAYS, args.ic_window_days,
        start_date, end_date, history_start, emit,
    )

    # Bear picks (V19-opt style)
    factor_returns = build_factor_return_history(
        date_panel, FACTORS_RUNTIME, history_start, end_date, emit,
    )
    bear_rebal = sorted(_BEAR_REBALANCE_DATES)
    bear_picks, bear_vote_log = make_bear_picks(
        date_panel, factor_returns, [datetime.strptime(d, "%Y-%m-%d").date() for d in bear_rebal], emit,
    )

    global _DAILY_PICKS_BULL, _DAILY_PICKS_BEAR
    _DAILY_PICKS_BULL = bull_picks
    _DAILY_PICKS_BEAR = bear_picks

    universe = sorted({s for p in bull_picks.values() for s in p} | {s for p in bear_picks.values() for s in p})
    emit("universe_ready", n_symbols=len(universe))
    data = build_data_dict(raw, universe, start_date - timedelta(days=30), end_date)
    emit("data_ready", n_symbols=len(data))
    if not data:
        raise RuntimeError("no AKQuant data for hybrid picks")

    emit("akquant_start")
    result = aq.run_backtest(
        data=data,
        strategy=V20HybridStrategy,
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
    result_obj = {
        "job_id": args.job_id,
        "contract": {
            "type": "v25a_hold10_rebal10",
            "bull": {
                "selector": "rolling_IC_voting",
                "ic_window_days": args.ic_window_days,
                "fwd_days": FWD_DAYS,
                "top_factors": TOP_FACTORS_BULL,
                "top_stocks_per_factor": TOP_STOCKS_PER_FACTOR_BULL,
                "min_votes": MIN_VOTES_BULL,
                "max_stocks": MAX_STOCKS_BULL,
                "rebalance": "daily",
            },
            "bear": {
                "selector": "factor_return_voting",
                "lookback_sessions": BEAR_LOOKBACK_SESSIONS,
                "fwd_days": FWD_DAYS,
                "round_trip_cost": ROUND_TRIP_COST,
                "top_factors": TOP_FACTORS_BEAR,
                "top_stocks_per_factor": TOP_STOCKS_PER_FACTOR_BEAR,
                "min_votes": MIN_VOTES_BEAR,
                "min_factor_mean_return": MIN_FACTOR_MEAN_RETURN,
                "min_factor_obs": MIN_FACTOR_OBS,
                "rebalance_sessions": BEAR_REBALANCE_SESSIONS,
                "min_stocks": MIN_STOCKS_BEAR,
                "max_stocks": MAX_STOCKS_BEAR,
            },
            "router": {
                "bear": "idx_ret_60d <= -5% AND idx_ret_20d <= 0",
                "bull_neutral": "otherwise",
            },
            "min_hold_days": MIN_HOLD_DAYS,
        },
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_factors": len(FACTORS_RUNTIME),
        "factors": FACTORS_RUNTIME,
        "regime_counts": dict(regime_counter),
        "n_bear_rebalance_dates": len(_BEAR_REBALANCE_DATES),
        "n_bull_pick_dates": len(bull_picks),
        "n_bear_pick_dates": len(bear_picks),
        "avg_stocks_per_bull_pick": float(np.mean([len(x) for x in bull_picks.values()])) if bull_picks else 0.0,
        "avg_stocks_per_bear_pick": float(np.mean([len(x) for x in bear_picks.values()])) if bear_picks else 0.0,
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
            for row in regime_records_in_window
        ],
        "bear_vote_log": bear_vote_log,
        "trades_ledger": trades,
        "orders_ledger": orders,
    }
    result_path = job_dir / "result.json"
    result_path.write_text(json.dumps(result_obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    factor_returns.write_parquet(job_dir / "factor_returns.parquet")
    emit(
        "done", final_value=final_value, final_return_pct=result_obj["final_return_pct"],
        n_trades=len(trades), n_orders=len(orders), result_file=str(result_path),
    )


if __name__ == "__main__":
    main()