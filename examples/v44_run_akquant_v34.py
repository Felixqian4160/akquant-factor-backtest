"""V44 runner = V43 + basket weighting (等权 → 反波动率 / HRP 分层风险平价).

Weight modes (调仓时对 basket 的个股权重):
  equal : 1/N 等权 (与 v43/v42 一致, 对照)
  invvol: w_i ∝ 1/σ_i(weight_win 日)    (简单的风险平价近似)
  hrp   : Lopez de Prado 分层风险平价 (相关距离单链聚类 + 递归逆方差二分)
可选 --w-cap 权重上限 (等权扩容迭代).
其余合同与 v43 相同 (exposure overlay 可叠加).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, "src")
import akquant as aq
import akquant.backtest.engine as bte

PANEL = Path("data/wavehunter_hs300_v34_adj_20261007.parquet")
IDX_PANEL = Path("data/wavehunter_hs300_v34_dedup_20261010.parquet")
PICKS_BASE = Path("evidence/sweep/v34_factorrank_lb20_20261007")
OUT_BASE = Path("evidence/sweep/v34_akquant_20261007")
CA_PATH = Path("evidence/v34_adj_20261007/corporate_actions_derived.parquet")
START = pd.Timestamp("2010-01-01")
DATA_START = pd.Timestamp("2010-01-01")
DEFAULT_END = "2025-12-31"
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
TARGET_TOTAL = 0.90
INITIAL_CASH = 100_000_000.0
HOLDING_BARS = 21

_CA_ACTIONS: list = []


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def extract_records(obj):
    if not isinstance(obj, list):
        return []
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append(item)
            continue
        rec = {}
        for attr in dir(item):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(item, attr)
            except Exception:
                continue
            if callable(v):
                continue
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            elif hasattr(v, "item"):
                try:
                    v = v.item()
                except Exception:
                    v = str(v)
            rec[attr] = v
        out.append(rec)
    return out


def metrics_dict(result):
    raw = result.metrics_df
    out = {}
    for name in raw.index:
        value = raw.loc[name, "value"]
        if hasattr(value, "isoformat"):
            value = value.isoformat()
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            out[name] = str(value)
    return out


def build_data(universe, end_ts, prices="raw"):
    px = ["open", "high", "low", "close"]
    if prices == "hfq":
        px = ["adj_open", "adj_high", "adj_low", "adj_close"]
    cols = ["trade_date", "ts_code"] + px + ["vol"]
    source = pl.scan_parquet(PANEL).select(cols).filter(
        (pl.col("trade_date") >= pl.lit(DATA_START.to_pydatetime()))
        & (pl.col("trade_date") <= pl.lit(end_ts.to_pydatetime()))
    ).collect()
    if prices == "hfq":
        source = source.rename({"adj_open": "open", "adj_high": "high",
                                "adj_low": "low", "adj_close": "close"})
    all_dates = source.get_column("trade_date").unique().sort().to_list()
    date_grid = pl.DataFrame({"trade_date": all_dates}).with_columns(
        pl.col("trade_date").cast(pl.Datetime("ms"))
    )
    data = {}
    for symbol in universe:
        bars = source.filter(pl.col("ts_code") == symbol).sort("trade_date")
        if bars.is_empty():
            continue
        pdf = (
            date_grid.join(bars, on="trade_date", how="left")
            .sort("trade_date")
            .with_columns([
                pl.col("ts_code").fill_null(symbol),
                pl.col("close").forward_fill(),
                pl.col("open").fill_null(pl.col("close").forward_fill()),
                pl.col("high").fill_null(pl.col("close").forward_fill()),
                pl.col("low").fill_null(pl.col("close").forward_fill()),
                pl.col("vol").fill_null(0.0),
            ])
            .to_pandas()
            .set_index("trade_date")
            .rename_axis("date")
        )
        pdf["symbol"] = symbol
        pdf = pdf.drop(columns=["ts_code"])
        pdf["volume"] = np.where(pdf["vol"].to_numpy() > 0, 1.0e9, 0.0)
        data[symbol] = pdf
    return data


def load_ca_actions(universe, end_ts):
    if not CA_PATH.exists():
        return []
    evd = pl.read_parquet(CA_PATH)
    evd = evd.filter(pl.col("ex_date") >= dt.date(2010, 1, 1))
    evd = evd.filter(pl.col("ex_date") <= end_ts.date())
    evd = evd.filter(pl.col("ts_code").is_in(universe))
    evd = evd.sort(["ex_date", "ts_code", "action"])
    actions = []
    for r in evd.iter_rows(named=True):
        if r["action"] == "dividend" and r["value"] and r["value"] > 0:
            actions.append(aq.CorporateAction(
                symbol=r["ts_code"], date=r["ex_date"],
                action_type=aq.CorporateActionType.Dividend, value=float(r["value"])))
        elif r["action"] == "split" and r["value"] and r["value"] > 1.0005:
            actions.append(aq.CorporateAction(
                symbol=r["ts_code"], date=r["ex_date"],
                action_type=aq.CorporateActionType.Split, value=float(r["value"])))
    return actions


class _EngineProxy:
    """Transparent proxy that injects corporate actions right after add_data()."""
    def __init__(self):
        object.__setattr__(self, "_e", aq.Engine())

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_e"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_e"), name, value)

    def add_data(self, feed):
        self._e.add_data(feed)
        for a in _CA_ACTIONS:
            self._e.add_corporate_action(a)


def _duration_stats(trades):
    if not trades or "duration_bars" not in trades[0]:
        return {}
    d = pd.to_numeric(pd.Series([t.get("duration_bars") for t in trades]), errors="coerce").dropna()
    if d.empty:
        return {}
    return {
        "n": int(len(d)),
        "n_le_1": int((d <= 1).sum()),
        "pct_le_1": float((d <= 1).mean()),
        "n_20_22": int(((d >= 20) & (d <= 22)).sum()),
        "pct_20_22": float(((d >= 20) & (d <= 22)).mean()),
        "median": float(d.median()),
        "max": float(d.max()),
    }


def audit(metrics, trades, orders, nav, picks, end_ts):
    odf = pd.DataFrame(orders)
    tdf = pd.DataFrame(trades)
    if not odf.empty:
        odf["status"] = odf["status"].astype(str)
        odf["side"] = odf["side"].astype(str)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("trades_exist", len(trades) > 0, len(trades))
    check("orders_exist", len(orders) > 0, len(orders))
    n_susp = n_margin = 0
    if odf.empty:
        filled = rejected = 0
        check("orders_both_sides", False, "empty orders")
    else:
        filled = int((odf["status"] == "OrderStatus.Filled").sum())
        rejected = int((odf["status"] == "OrderStatus.Rejected").sum())
        if "reject_reason" in odf:
            rr = odf["reject_reason"].astype(str)
            n_susp = int(rr.str.contains("not tradable", na=False).sum())
            n_margin = int(rr.str.contains("Insufficient margin", na=False).sum())
        check("orders_both_sides", odf["side"].nunique() >= 2, odf["side"].value_counts().to_dict())
    # suspension-day rejects are realistic (A-share halts block trades); margin rejects should be rare
    check("orders_rejected_classified", n_margin <= 5,
          f"filled={filled}, rejected={rejected}, suspension={n_susp}, margin={n_margin}")

    if not odf.empty and "side" in odf:
        buys = odf[odf["side"] == "OrderSide.Buy"]
        bq = pd.to_numeric(buys["quantity"], errors="coerce").dropna()
        check("buy_qty_lot_multiple", bool((bq % LOT_SIZE == 0).all()), f"n_buys={len(bq)}")
        sells = odf[odf["side"] == "OrderSide.Sell"]
        sq = pd.to_numeric(sells["quantity"], errors="coerce").dropna()
        n_odd_sells = int((sq % LOT_SIZE != 0).sum())
        check("sell_odd_lots_info", True, f"odd_sells={n_odd_sells}/{len(sq)} (split residues)")
    check("nav_bounded", bool(nav) and nav[-1][0] <= end_ts.strftime("%Y-%m-%d"),
          f"last={nav[-1][0] if nav else None}")
    check("rebalance_dates_in_window",
          all(START.strftime("%Y-%m-%d") <= d <= end_ts.strftime("%Y-%m-%d") for d in picks),
          len(picks))

    if not tdf.empty and "pnl" in tdf and "commission" in tdf and "net_pnl" in tdf:
        gross = float(pd.to_numeric(tdf["pnl"], errors="coerce").sum())
        comm = float(pd.to_numeric(tdf["commission"], errors="coerce").sum())
        net = float(pd.to_numeric(tdf["net_pnl"], errors="coerce").sum())
        check("trade_pnl_identity", abs(gross - comm - net) < 1.0, f"diff={gross-comm-net:.6f}")

    nav_values = np.array([float(x[1]) for x in nav], dtype=float)
    mdd = None
    if nav_values.size:
        normalized = nav_values / INITIAL_CASH
        curve = np.concatenate(([1.0], normalized))
        peak = np.maximum.accumulate(curve)
        mdd = float(np.min(curve / peak - 1.0))
        check("mdd_recomputed", True, f"{mdd:.6f}")

    dstat = _duration_stats(trades)
    if dstat:
        check(f"duration_info_cap{HOLDING_BARS}", True,
              f"pct_20_22={dstat.get('pct_20_22'):.3f} median={dstat.get('median')} n_le_1={dstat.get('n_le_1')}")

    return {"checks": checks, "summary": {
        "final_value": metrics.get("end_market_value"),
        "total_return_pct": metrics.get("total_return_pct"),
        "annualized_return": metrics.get("annualized_return"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "max_drawdown_pct_engine": metrics.get("max_drawdown_pct"),
        "max_drawdown_pct_recomputed": mdd * 100 if mdd is not None else None,
        "win_rate": metrics.get("win_rate"),
        "profit_factor": metrics.get("profit_factor"),
        "closed_trade_count": metrics.get("closed_trade_count"),
        "n_trades_ledger": len(trades),
        "n_orders_ledger": len(orders),
        "n_rejected": rejected,
        "n_rejected_suspension": n_susp,
        "n_rejected_margin": n_margin,
        "total_commission": metrics.get("total_commission"),
        "nav_start": nav[0][0] if nav else None,
        "nav_end": nav[-1][0] if nav else None,
        "n_rebalances": len(picks),
        "duration_stats": dstat,
        "n_corporate_actions": len(_CA_ACTIONS),
    }}


def _quasi_diag(link, n):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    while sort_ix.max() >= n:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n]
        i, j = df0.index, df0.values - n
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def hrp_weights(R: pd.DataFrame) -> pd.Series:
    """Lopez de Prado HRP: 相关距离单链聚类 -> quasi-diag -> 递归逆方差分配."""
    from scipy.cluster.hierarchy import linkage
    syms = list(R.columns)
    if len(syms) == 1:
        return pd.Series([1.0], index=syms)
    corr = R.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    link = linkage(dist.to_numpy()[np.triu_indices(len(syms), 1)], method="single")
    order = _quasi_diag(link, len(syms))
    sorted_syms = [syms[i] for i in order]
    var = R.var().replace(0, np.nan)
    var = var.fillna(var.mean())
    w = pd.Series(1.0, index=sorted_syms)
    clusters = [list(sorted_syms)]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            if i + 1 < len(clusters):
                c0, c1 = clusters[i], clusters[i + 1]
                v0 = float(var[c0].sum())
                v1 = float(var[c1].sum())
                a = 1.0 - v0 / (v0 + v1) if (v0 + v1) > 0 else 0.5
                w[c0] *= a
                w[c1] *= 1.0 - a
    return w / w.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--picks-base", default=str(PICKS_BASE))
    parser.add_argument("--out-base", default=str(OUT_BASE))
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--actions", choices=["on", "off"], default="on")
    parser.add_argument("--ca-mode", choices=["all", "div", "spl"], default="all")
    parser.add_argument("--prices", choices=["raw", "hfq"], default="raw")
    parser.add_argument("--holding-bars", type=int, default=21)
    parser.add_argument("--exposure", choices=["none", "idx", "idxhard", "idxl", "dd", "volt", "scale", "combo"], default="none")
    parser.add_argument("--scale-f", type=float, default=0.75)
    parser.add_argument("--weight-mode", choices=["equal", "invvol", "hrp"], default="equal")
    parser.add_argument("--weight-win", type=int, default=60)
    parser.add_argument("--w-cap", type=float, default=1.0)
    parser.add_argument("--idx-ma", type=int, default=60)
    parser.add_argument("--vol-target", type=float, default=0.22)
    parser.add_argument("--vol-floor", type=float, default=0.30)
    parser.add_argument("--dd-soft", type=float, default=-0.08)
    parser.add_argument("--dd-hard", type=float, default=-0.14)
    parser.add_argument("--dd-restore", type=float, default=-0.04)
    args = parser.parse_args()

    global _CA_ACTIONS, HOLDING_BARS
    HOLDING_BARS = int(args.holding_bars)
    end_ts = pd.Timestamp(args.end)
    tag = args.tag
    picks_path = pathlib.Path(args.picks_base) / tag / "picks.json"
    meta_path = pathlib.Path(args.picks_base) / tag / "picks_meta.json"
    out_dir = pathlib.Path(args.out_base) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = json.loads(picks_path.read_text())
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    log(f"V41 v34 [{tag}]: picks={len(picks)}; window_end={args.end}; actions={args.actions}")
    universe = sorted({s for basket in picks.values() for s in basket})
    log(f"universe={len(universe)} symbols")

    # ── exposure: 指数状态序列 (因果: MA 仅用 <= d 的收盘) ──
    idx_ok = {}
    idx_ok120 = {}
    if args.exposure in ("idx", "idxhard", "idxl", "combo"):
        idx_df = (pl.read_parquet(IDX_PANEL, columns=["trade_date", "idx_close"])
                  .unique(subset=["trade_date"]).sort("trade_date"))
        idxs = pd.Series(idx_df["idx_close"].to_numpy(),
                         index=pd.to_datetime(idx_df["trade_date"].to_numpy()))
        ma60 = idxs.rolling(int(args.idx_ma)).mean()
        ma120 = idxs.rolling(120).mean()
        s60 = (idxs > ma60) & ma60.notna()
        s120 = (idxs > ma120) & ma120.notna()
        idx_ok = {str(d.date()): bool(v) for d, v in s60.items()}
        idx_ok120 = {str(d.date()): bool(v) for d, v in s120.items()}
        log(f"exposure=idx: ok{args.idx_ma}={sum(idx_ok.values())}/{len(idx_ok)} "
            f"ok120={sum(idx_ok120.values())}")

    data = build_data(universe, end_ts, args.prices)
    if not data:
        raise RuntimeError("empty AKQuant data")
    log(f"data={len(data)} symbols, bounds={min(x.index.min() for x in data.values())}..{max(x.index.max() for x in data.values())}")

    # ── FIXED panel_days: full trading calendar (was: df.index[0] per symbol) ──
    _all_days = set()
    for _sym, _df in data.items():
        _all_days.update(_df.index)
    import bisect
    panel_days = sorted(_all_days)
    panel_days_dt = [d.date() if hasattr(d, "date") else d for d in panel_days]
    log(f"panel_days: {len(panel_days_dt)} trading days ({panel_days_dt[0]}..{panel_days_dt[-1]})")

    def add_trading_days(start_date, k):
        sd = start_date.date() if hasattr(start_date, "date") else start_date
        idx = bisect.bisect_right(panel_days_dt, sd)
        if idx + k - 1 < len(panel_days_dt):
            return panel_days_dt[idx + k - 1]
        return panel_days_dt[-1]

    # ── corporate actions ──
    if args.prices == "hfq":
        args.actions = "off"
        log("prices=hfq -> corporate actions auto-disabled (adjusted prices already total-return)")
    if args.actions == "on":
        _CA_ACTIONS = load_ca_actions(universe, end_ts)
        if args.ca_mode == "div":
            _CA_ACTIONS = [a for a in _CA_ACTIONS if "Dividend" in str(a.action_type)]
        elif args.ca_mode == "spl":
            _CA_ACTIONS = [a for a in _CA_ACTIONS if "Split" in str(a.action_type)]
        n_div = sum(1 for a in _CA_ACTIONS if "Dividend" in str(a.action_type))
        n_spl = sum(1 for a in _CA_ACTIONS if "Split" in str(a.action_type))
        log(f"corporate actions[{args.ca_mode}]: {len(_CA_ACTIONS)} (dividend={n_div}, split={n_spl})")
    else:
        _CA_ACTIONS = []
        log("corporate actions: DISABLED (control run)")

    HOLD_DAYS = HOLDING_BARS

    class V43Strategy(aq.Strategy):
        warmup = 5
        _nav_count = 0
        _f_stats = None

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._entry_date = {}
            self._nav_hist = []      # (date_obj, equity) — on_after_trading 逐日累积
            self._dd_f = 1.0         # DD 棘轮状态
            self._f_used = []

        def on_bar(self, bar):
            pass

        def _force_exit_holding_cap(self, current_date_obj):
            pos = self.positions or {}
            stale = []
            for sym, ed in list(self._entry_date.items()):
                if ed is None:
                    continue
                if add_trading_days(ed, HOLD_DAYS) <= current_date_obj:
                    q = float(pos.get(sym, 0.0) or 0.0)
                    if q > 0:
                        stale.append(sym)
                    else:
                        self._entry_date.pop(sym, None)
            for sym in stale:
                try:
                    self.close_position(sym)
                except Exception:
                    pass
                self._entry_date.pop(sym, None)

        def _basket_weights(self, basket, trading_date):
            """返回 basket 内个股权重 (和=1). 缺数据个股回退等权份额, 最后归一."""
            n = len(basket)
            if args.weight_mode == "equal":
                return {sym: 1.0 / n for sym in basket}
            win = int(args.weight_win)
            pmap = {}
            for sym in basket:
                df = data.get(sym)
                if df is None:
                    continue
                try:
                    s = df["close"].loc[:pd.Timestamp(trading_date)]
                except Exception:
                    continue
                if len(s) >= max(20, win // 2) + 1:
                    r = s.pct_change().dropna().iloc[-win:]
                    if len(r) >= max(20, win // 2):
                        pmap[sym] = r.reset_index(drop=True)
            if len(pmap) < 3:
                return {sym: 1.0 / n for sym in basket}
            R = pd.DataFrame({s: r for s, r in pmap.items()})
            if args.weight_mode == "invvol":
                vol = R.std().replace(0, np.nan)
                vol = vol.fillna(vol.mean())
                iv = 1.0 / vol
                w = iv / iv.sum()
            else:  # hrp
                w = hrp_weights(R)
            k = len(pmap)
            out = {}
            for sym in basket:
                if sym in w.index:
                    out[sym] = float(w[sym]) * (k / n)
                else:
                    out[sym] = 1.0 / n
            tot = sum(out.values())
            out = {s: v / tot for s, v in out.items()}
            cap = float(args.w_cap)
            if cap < 1.0:
                for _ in range(20):
                    over = {s: v for s, v in out.items() if v > cap}
                    if not over:
                        break
                    excess = sum(v - cap for v in over.values())
                    for s in over:
                        out[s] = cap
                    rest = [s for s, v in out.items() if v < cap - 1e-12]
                    if not rest:
                        break
                    tot = sum(out[s] for s in rest)
                    for s in rest:
                        out[s] += excess * out[s] / tot
            return out

        def on_cross_section(self, trading_date, timestamp):
            current_date_obj = trading_date.date() if hasattr(trading_date, "date") else trading_date
            self._force_exit_holding_cap(current_date_obj)
            d = str(trading_date)[:10]
            basket = picks.get(d, {})
            if not basket:
                return
            # ── exposure factor ──
            f = 1.0
            if args.exposure != "none":
                if args.exposure in ("dd", "combo") and len(self._nav_hist) >= 5:
                    navs = [x for _, x in self._nav_hist]
                    peak = max(navs)
                    eq = navs[-1]
                    dd = eq / peak - 1.0 if peak > 0 else 0.0
                    if dd <= args.dd_hard:
                        self._dd_f = 0.25
                    elif dd <= args.dd_soft:
                        self._dd_f = 0.5
                    elif dd >= args.dd_restore:
                        self._dd_f = 1.0
                idx_f = 1.0
                if args.exposure in ("idx", "idxhard", "idxl", "combo"):
                    if args.exposure == "idxl":
                        if idx_ok.get(d, False):
                            idx_f = 1.0
                        elif idx_ok120.get(d, False):
                            idx_f = 0.5
                        else:
                            idx_f = 0.25
                    else:
                        idx_f = 1.0 if idx_ok.get(d, False) else (0.5 if args.exposure == "idx" else 0.0)
                if args.exposure == "dd":
                    f = self._dd_f
                elif args.exposure in ("idx", "idxhard", "idxl"):
                    f = idx_f
                elif args.exposure == "volt":
                    if len(self._nav_hist) >= 21:
                        navs = np.array([x for _, x in self._nav_hist][-21:], dtype=float)
                        rets = np.diff(navs) / navs[:-1]
                        sig = float(np.std(rets) * np.sqrt(244.0))
                        f = float(np.clip(args.vol_target / max(sig, 1e-6), args.vol_floor, 1.0))
                elif args.exposure == "scale":
                    f = float(args.scale_f)
                else:  # combo
                    f = min(self._dd_f, idx_f)
            self._f_used.append(round(f, 2))
            if f <= 0.02:
                for sym in list((self.positions or {}).keys()):
                    q = float((self.positions or {}).get(sym, 0.0) or 0.0)
                    if q > 0:
                        try:
                            self.close_position(sym)
                        except Exception:
                            pass
                self._entry_date.clear()
                self.log(f"exposure {d}: f={f:.2f} -> liquidate")
                return
            eff_total = TARGET_TOTAL * f
            n = len(basket)
            wB = self._basket_weights(basket, trading_date)
            self.rebalance_weights(
                target_weights={sym: eff_total * wB[sym] for sym in basket},
                liquidate_unmentioned=True,
            )
            for sym in basket:
                if sym not in self._entry_date or self._entry_date[sym] is None:
                    self._entry_date[sym] = current_date_obj
            for sym in list(self._entry_date.keys()):
                if sym not in basket:
                    self._entry_date.pop(sym, None)
            self.log(f"rebalance {d}: {n} stocks, expo={f:.2f}")

        def on_after_trading(self, trading_date, timestamp):
            current_date_obj = trading_date.date() if hasattr(trading_date, "date") else trading_date
            self._force_exit_holding_cap(current_date_obj)
            try:
                eq = float(self.equity)
            except Exception:
                eq = None
            if eq is not None and eq > 0:
                self._nav_hist.append((current_date_obj, eq))
            V43Strategy._nav_count = len(self._nav_hist)

    started = time.time()
    real_engine = bte.Engine
    if _CA_ACTIONS:
        bte.Engine = lambda: _EngineProxy()
    try:
        result = aq.run_backtest(
            data=data,
            strategy=V43Strategy,
            initial_cash=INITIAL_CASH,
            commission_rate=COMMISSION_RATE,
            slippage=SLIPPAGE,
            t_plus_one=True,
            fill_policy=aq.NextOpen(),
            lot_size=LOT_SIZE,
        )
    finally:
        bte.Engine = real_engine
    log(f"finished in {time.time() - started:.1f}s | nav_days={V43Strategy._nav_count}")

    metrics = metrics_dict(result)
    trades = extract_records(result.trades)
    orders = extract_records(result.orders)
    nav = [[str(ts)[:10], float(value)] for ts, value in result.equity_curve.items()]
    audit_result = audit(metrics, trades, orders, nav, picks, end_ts)

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False)
    pd.DataFrame(orders).to_csv(out_dir / "orders.csv", index=False)
    pd.DataFrame(nav, columns=["date", "value"]).to_csv(out_dir / "nav.csv", index=False)
    (out_dir / "ledger_audit.json").write_text(json.dumps(audit_result, ensure_ascii=False, indent=2, default=str))
    (out_dir / "result.json").write_text(json.dumps({
        "contract": {
            "runner": "v41_run_akquant_v34",
            "panel": str(PANEL),
            "picks": str(picks_path),
            "window": f"{START.date()} ~ {end_ts.date()}",
            "selector": "factor_return_voting lb=20 (BULL+BEAR), v34 adjusted factors",
            "holding_cap_bars": HOLDING_BARS,
            "panel_days_fixed": True,
            "corporate_actions": args.actions,
            "prices": args.prices,
            "n_corporate_actions": len(_CA_ACTIONS),
            "ca_source": str(CA_PATH),
            "target_total": TARGET_TOTAL,
            "exposure": args.exposure,
            "weight_mode": args.weight_mode,
            "weight_win": args.weight_win,
            "exposure_params": {"idx_ma": args.idx_ma, "dd_soft": args.dd_soft,
                                "dd_hard": args.dd_hard, "dd_restore": args.dd_restore,
                                "vol_target": args.vol_target, "vol_floor": args.vol_floor},
            "t_plus_one": True, "fill_policy": "NextOpen",
            "commission_rate": COMMISSION_RATE, "slippage": SLIPPAGE, "lot_size": LOT_SIZE,
        },
        "metrics": metrics,
        "audit": audit_result,
    }, ensure_ascii=False, indent=2, default=str))
    log(f"saved={out_dir}")


if __name__ == "__main__":
    main()
