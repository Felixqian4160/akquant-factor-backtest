# V33 IC Voting Single-Run Baseline

**Audit generated:** 2026-10-07 00:14:52
**Source artifacts:** evidence/sweep/v33_ic_20261007/

## Strategy Contract

```
Panel:          wavehunter_hs300_v33_with_new_factors_20261003.parquet
Panel cols:     458 (largest available)
Voting factors: 428 (= 458 - 19 base cols - 11 zigzag labels)
Excluded:       11 zigzag labels (v10_1_*, look-ahead bias)
Router:         Causal ZigZag (leg.start, no look-ahead)
Bull selector:  V14 IC voting, MIN_VOTES=2, K_NOM=10
Bear selector:  V14 IC voting, MIN_VOTES=4, K_NOM=10
IC:             60-day rolling Spearman, lag=21, FWD=20
Rebalance:      every 20 trading days
Hold:           21-bar hard cap
Cost:           25 bps commission + 10 bps slippage
Lot:            100 shares
Target:         90% of equity, equal-weighted
Seeds:          1 (single run, offset=0)
Window:         2010-01-04 ~ 2025-12-31 (15.99 years)
```

## Result (Single Run, Offset=0)

| Metric | Value |
|---|---:|
| Total return | +41.07% |
| Annualized return | +2.17% |
| Sharpe ratio | 0.223 |
| Max drawdown | 61.9% |
| Profit factor | 1.293 |
| Win rate | 48.5% |
| Closed trades | 1704 |
| Open positions | 10 |
| Avg holding period | 9.3 bars |
| Total commission | ¥137.8M |
| Rejected orders | 688 |

## Audit Checks

- [PASS] **trades_exist**: 1704
- [PASS] **orders_exist**: 3995
- [PASS] **orders_both_sides**: {'OrderSide.Sell': 2350, 'OrderSide.Buy': 1645}
- [FAIL] **orders_filled_all**: filled=3307, rejected=688
- [PASS] **qty_lot_multiple**: n=1704
- [PASS] **nav_bounded**: last=2025-12-31
- [PASS] **rebalance_dates_in_window**: 195
- [PASS] **trade_pnl_identity**: diff=0.000000
- [PASS] **mdd_recomputed**: -0.618727
- [PASS] **holding_cap_21bars**: within_22=1696/1704=0.995

## Panel Evolution

| Panel | Total Cols | Factor-like |
|---|---:|---:|
| with_talib (09-24) | 419 | 400 |
| v31_refactored (09-25) | 414 | 411 |
| v31_refactored_v2 (09-25) | 420 | 411 |
| v32_complete (10-03) | 441 | 422 |
| **v33_with_new_factors (10-03)** | **458** | **439** |

v33 is the largest panel. Older panels archived to `data/_archive/`.

## Files Saved

```
evidence/sweep/v33_ic_20261007/
├── V14_0/
│   ├── picks.json         (35,344 bytes)
│   ├── picks_meta.json    (32,936 bytes)
│   └── summary.json       (1,038 bytes)
├── akquant/V14_0/
│   ├── result.json        (4,594 bytes)
│   ├── ledger_audit.json  (1,654 bytes)
│   ├── trades.csv         (~395 KB)
│   ├── orders.csv         (~1.3 MB)
│   └── nav.csv            (~110 KB)
├── v33_ic_equity_curve.png
└── REPORT.md (this file)
```
