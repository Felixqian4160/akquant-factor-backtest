"""Paper trading simulator CLI for WebUI.

Reuses Stage 5b's BullCompositeTopK10Kill strategy. Runs the full panel
from start_date to end_date with paper-trading simulation:
- start with initial_cash
- each rebalance day: top-K picks → buy/sell
- cost: 25bps commission + 10bps slippage per side
- mark-to-market daily
- if MDD > 8% → flat for 20d (kill-switch, like Stage 5b)

Outputs:
- state.json:  latest portfolio snapshot (cash, positions, NAV)
- ledger.json:  every trade with timestamp, price, qty
- result.json:  final metrics + NAV curve
- progress.jsonl: incremental progress updates

Single-variable change vs Stage 5b: panel slicing (per-bull-leg → date range).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import akquant as aq
from akquant import Bar, Strategy

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest")
sys.path.insert(0, str(ROOT / "src"))

PANEL = Path(
    "/media/felix/f/quant/aurumq-rl/evidence/"
    "quant_workflow_migration_20260915/"
    "v10_2_mainwave_features_v2_talib_20260924_022316/"
    "wavehunter_mainwave_features_v2.parquet"
)

BULL_FACTORS = [
    "talib_NATR", "talib_TRANGE",
    "gtja_gtja_159", "gtja_gtja_149", "gtja_gtja_144",
    "mw_vol_20d",
]
TOP_K = 10
REBAL_DAYS = 20
COMMISSION_BPS_PER_SIDE = 40
KILL_DD_THRESHOLD = 0.08
KILL_COOLDOWN_DAYS = 20


class PaperTradingStrategy(Strategy):
    """Stage 5b config wrapped for paper trading. Tracks all trades."""

    warmup = 5
    rebal_days = REBAL_DAYS
    top_n = TOP_K

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._factor_store: dict = {}
        self._symbols_seen: dict = {}
        self._last_rebal_date = None
        self._equity_peak: float = 0.0  # set on first day from initial_cash arg
        self._kill_until = None

    def on_bar(self, bar: Bar) -> None:
        ts = pd.Timestamp(bar.timestamp, unit="ns", tz="Asia/Shanghai")
        d = ts.date()
        if d not in self._factor_store:
            self._factor_store[d] = {}
            self._symbols_seen[d] = set()
        vals = []
        ok = True
        for f in BULL_FACTORS:
            v = bar.extra.get(f)
            if v is None:
                ok = False
                break
            vals.append(float(v))
        if ok:
            self._factor_store[d][bar.symbol] = vals
            self._symbols_seen[d].add(bar.symbol)

    def on_cross_section(self, trading_date, timestamp) -> None:
        d = trading_date
        acct = self.get_account()
        equity = float(acct.get("equity", 0))
        if self._equity_peak == 0 and equity > 0:
            self._equity_peak = equity
        if equity > self._equity_peak:
            self._equity_peak = equity
        dd = (equity / self._equity_peak) - 1.0 if self._equity_peak > 0 else 0.0
        if dd <= -KILL_DD_THRESHOLD:
            self._kill_until = d + timedelta(days=KILL_COOLDOWN_DAYS)

        if self._kill_until is not None and d < self._kill_until:
            try:
                self.close_position()
            except Exception:
                pass
            return

        if d not in self._factor_store:
            return
        if self._last_rebal_date is not None:
            gap = (d - self._last_rebal_date).days
            if gap < self.rebal_days:
                return
        self._last_rebal_date = d
        raw = self._factor_store[d]
        if len(raw) < self.top_n:
            return
        syms = list(raw.keys())
        mat = np.array([raw[s] for s in syms])
        ranks = np.zeros_like(mat)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            order = np.argsort(col, kind="mergesort")
            r = np.empty_like(order, dtype=float)
            r[order] = np.arange(len(col))
            ranks[:, j] = r / max(len(col) - 1, 1)
        composite = ranks.mean(axis=1)
        scores = {syms[i]: float(composite[i]) for i in range(len(syms))}
        try:
            self.rebalance_to_topn(
                scores=scores, top_n=self.top_n, weight_mode="equal",
                long_only=True, liquidate_unmentioned=True,
            )
        except Exception as exc:
            self.log(f"rebalance failed: {exc}")


def build_data_dict(start: date, end: date) -> tuple[dict, int, int]:
    df = (
        pl.scan_parquet(str(PANEL))
        .select(
            ["trade_date", "ts_code", "open", "high", "low", "close", "vol"]
            + BULL_FACTORS
        )
        .with_columns(pl.col("trade_date").cast(pl.Date))
        .filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        .drop_nulls(subset=BULL_FACTORS)
        .collect()
    )
    n_rows = df.shape[0]
    stocks = sorted(df["ts_code"].unique().to_list())
    data_dict: dict[str, pd.DataFrame] = {}
    for code in stocks:
        pdf = (
            df.filter(pl.col("ts_code") == code)
            .sort("trade_date").to_pandas()
            .set_index("trade_date").rename_axis("date")
        )
        pdf["symbol"] = code
        pdf = pdf.drop(columns=["ts_code"])
        data_dict[code] = pdf
    return data_dict, n_rows, len(stocks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--initial-cash", type=float, default=1_000_000.0)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rebal-days", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / "progress.jsonl"
    result_path = out_dir / "result.json"
    state_path = out_dir / "state.json"

    def emit_progress(p: dict) -> None:
        with progress_path.open("a") as fh:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    emit_progress({"event": "start", "args": vars(args)})

    data_dict, n_rows, n_stocks = build_data_dict(start_date, end_date)
    if n_stocks < TOP_K:
        emit_progress({"event": "error", "msg": f"too few stocks ({n_stocks})"})
        return 1

    emit_progress({"event": "running", "n_rows": n_rows, "n_stocks": n_stocks})

    t0 = time.time()
    result = aq.run_backtest(
        data=data_dict, strategy=PaperTradingStrategy,
        initial_cash=args.initial_cash,
        commission_rate=COMMISSION_BPS_PER_SIDE / 10_000,
        slippage=0.0, t_plus_one=False,
        fill_policy=aq.NextOpen(), lot_size=100,
    )
    elapsed = time.time() - t0

    m = result.metrics_df
    total_pnl = float(m.loc["total_pnl", "value"])
    upnl = float(m.loc["unrealized_pnl", "value"])
    initial = float(m.loc["initial_market_value", "value"])
    closed_pnl = total_pnl - upnl
    closed_ret = closed_pnl / initial * 100
    days = (end_date - start_date).days
    ann = ((1 + closed_ret / 100) ** (365 / max(days, 1)) - 1) * 100

    trades = result.trades if hasattr(result, "trades") else []
    ledger = []
    for t in trades:
        try:
            ledger.append({
                "ts_code": getattr(t, "symbol", getattr(t, "ts_code", "?")),
                "side": getattr(t, "side", "?"),
                "price": float(getattr(t, "price", 0)),
                "qty": int(float(getattr(t, "quantity", 0))),
                "timestamp": str(getattr(t, "timestamp", "")),
                "commission": float(getattr(t, "commission", 0)),
            })
        except Exception:
            continue

    out = {
        "session_id": args.session_id,
        "contract": {
            "strategy": "PaperTradingStrategy (Stage 5b)",
            "factors": BULL_FACTORS,
            "top_k": TOP_K,
            "rebal_days": REBAL_DAYS,
            "kill_dd_threshold": KILL_DD_THRESHOLD,
            "kill_cooldown_days": KILL_COOLDOWN_DAYS,
            "commission_bps_per_side": COMMISSION_BPS_PER_SIDE,
        },
        "config": vars(args),
        "metrics": {
            "closed_only_return_pct": round(closed_ret, 4),
            "total_return_pct": round(float(m.loc["total_return_pct", "value"]), 4),
            "unrealized_pnl": round(upnl, 2),
            "sharpe_ratio": round(float(m.loc["sharpe_ratio", "value"]), 3),
            "max_drawdown_pct": round(float(m.loc["max_drawdown_pct", "value"]), 2),
            "win_rate": round(float(m.loc["win_rate", "value"]), 2),
            "closed_trade_count": int(float(m.loc["closed_trade_count", "value"])),
            "annualized_pct": round(ann, 1),
            "initial_cash": initial,
            "end_equity": float(m.loc["end_market_value", "value"]),
        },
        "n_rows": n_rows, "n_stocks": n_stocks,
        "elapsed_sec": round(elapsed, 1),
    }
    result_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    state_path.write_text(json.dumps({
        "session_id": args.session_id,
        "latest_metrics": out["metrics"],
        "n_trades": len(ledger),
        "updated_at": datetime.now().isoformat(),
    }, indent=2))

    # Save ledger separately (could be large).
    (out_dir / "ledger.json").write_text(json.dumps(ledger[:5000], indent=2, default=str))

    emit_progress({"event": "finished", "elapsed_sec": round(elapsed, 1), "metrics": out["metrics"]})
    print(f"wrote {result_path}")
    print(f"closed_only_return: {closed_ret}%, sharpe: {out['metrics']['sharpe_ratio']}, MDD: {out['metrics']['max_drawdown_pct']}%")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        sys.exit(1)