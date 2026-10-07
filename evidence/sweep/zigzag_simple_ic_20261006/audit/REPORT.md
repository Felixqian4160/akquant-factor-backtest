# ZigZag Simple IC Voting Strategy — 10-Seed Comprehensive Audit

**Audit generated:** 2026-10-06 23:51:41
**Source artifacts:** evidence/sweep/zigzag_simple_ic_20261006/

## Strategy Contract

```
Router:    Causal ZigZag (leg.start, no look-ahead)
Bull:      V14 IC voting, MIN_VOTES=2, K_NOM=10, TOP_K=10
Bear:      V14 IC voting, MIN_VOTES=4, K_NOM=10, TOP_K=10
IC:        60-day rolling Spearman, lag=21, FWD=20
Rebalance: every 20 trading days
Hold:      21-bar hard cap (force-close after 21 trading days)
Universe:  HS300 panel, 410 factors
Cost:      25 bps commission + 10 bps slippage
Lot:       100 shares
Target:    90% of equity, equal-weighted
Seeds:     10 offsets [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
Window:    2010-01-04 ~ 2025-12-31 (15.99 years)
```

## 10-Seed Cross-Seed Summary

| Offset | Total Ret | Annual | Sharpe | MDD | PF | Win | Trades | Rej |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V14_ 0 |  +10.89% |  +0.65% | 0.120 | 59.1% | 1.256 | 47.7% |  1739 |  707 |
| V14_ 2 |   +6.65% |  +0.40% | 0.106 | 44.6% | 1.242 | 51.1% |  1754 |  788 |
| V14_ 4 |  +91.42% |  +4.14% | 0.354 | 42.1% | 1.432 | 49.9% |  1743 |  730 |
| V14_ 6 |   -7.81% |  -0.51% | 0.055 | 69.7% | 1.179 | 50.5% |  1736 |  785 |
| V14_ 8 | +111.92% |  +4.81% | 0.375 | 45.0% | 1.403 | 49.2% |  1734 |  763 |
| V14_10 |  +10.99% |  +0.65% | 0.123 | 57.0% | 1.221 | 47.1% |  1714 |  778 |
| V14_12 |   +9.79% |  +0.59% | 0.116 | 58.7% | 1.261 | 49.5% |  1722 |  724 |
| V14_14 |  +79.46% |  +3.72% | 0.312 | 35.3% | 1.426 | 52.4% |  1719 |  752 |
| V14_16 | +132.22% |  +5.41% | 0.422 | 52.5% | 1.413 | 51.8% |  1732 |  812 |
| V14_18 |  +38.86% |  +2.07% | 0.213 | 49.8% | 1.271 | 50.5% |  1733 |  772 |

**Mean annual return: 2.19%**
**Mean Sharpe: 0.220**
**Mean MDD: 51.4%**
**Mean PF: 1.310**
**Positive offsets: 9/10**
**Stdev (annual): 2.14%**

## Period Split (OOS analysis)

| Period | Mean | Median | Min | Max | n_pos |
|---|---:|---:|---:|---:|---:|
|                      2010-2015 | +84.37% | +83.25% | +43.62% | +150.43% | 10/10 |
|                      2015-2020 | -6.72% | -9.34% | -30.73% | +22.90% | 3/10 |
|                      2020-2025 | -14.30% | -22.72% | -39.98% | +48.42% | 3/10 |
|        2010-2025_OOS_2023_2025 | +46.79% | +36.59% | +3.77% | +147.31% | 10/10 |
|                  OOS_2023_2025 | +0.18% | -0.96% | -21.72% | +26.16% | 5/10 |

**Critical OOS test (2023-2025, mean of 10 seeds):**
- 5/10 positive, mean +0.18%, median -0.96%
- This is the only true OOS window. The 2010-2022 window uses the same ZigZag
  router the strategy was designed around, so its 10/10 positive is partly
  in-sample cherry-picking.

## Yearly Breakdown (mean across 10 seeds)

| Year | Mean Ret | Median | Min | Max |
|---:|---:|---:|---:|---:|
| 2010 | +7.14% | +6.38% | -2.49% | +20.73% |
| 2011 | -9.14% | -9.56% | -16.78% | -1.25% |
| 2012 | -10.66% | -12.12% | -14.46% | -2.80% |
| 2013 | -2.50% | -5.00% | -8.78% | +19.39% |
| 2014 | +29.19% | +26.16% | +12.02% | +47.18% |
| 2015 | +65.44% | +61.23% | +33.39% | +111.91% |
| 2016 | -9.35% | -10.43% | -19.25% | +3.60% |
| 2017 | -5.81% | -7.00% | -11.01% | +1.42% |
| 2018 | -21.75% | -21.88% | -28.99% | -11.99% |
| 2019 | +26.80% | +27.16% | +6.30% | +51.51% |
| 2020 | +11.26% | +16.68% | -18.61% | +46.43% |
| 2021 | +0.49% | +1.16% | -10.53% | +19.05% |
| 2022 | -15.03% | -16.81% | -30.76% | +5.69% |
| 2023 | -7.84% | -8.64% | -12.51% | -2.02% |
| 2024 | -9.10% | -9.47% | -26.45% | +9.39% |
| 2025 | +22.37% | +21.79% | -8.87% | +61.59% |

**Pattern:**
- Strong positive years: 2014, 2015, 2019, 2020, 2025 (bull runs)
- Strong negative years: 2011, 2012, 2016, 2018, 2022, 2023, 2024 (bear markets)
- **2018 (-21.7%) and 2022 (-15.0%) are the worst** — strategy suffers in real bear
- **2023 (-7.8%) is concerning** — Causal ZigZag flagged it as bear, but strategy still lost

## Trade Attribution by Regime

**This is the most important table. It shows the bear regime IS the problem.**

| Regime | n_trades | Total PnL | Wins | Win Rate | Avg PnL/Trade |
|---|---:|---:|---:|---:|---:|
|         bear |  7075 | -1152.08M |  2397 | 33.9% |       -0 |
| bull_neutral | 10251 | +1638.21M |  4996 | 48.7% |       +0 |

**Total PnL: +486.13M | Bear contributes: -1152.08M | Bull contributes: +1638.21M**

**Critical finding:** Bear regime trades net -¥1.15M with 33.9% win rate, while
bull regime trades net +¥1.64M with 48.7% win rate. The ZigZag router correctly
identifies bear regimes, but V14 IC voting is **not robust to bear markets**.

## Trade Attribution by Year (10 seeds aggregate)

| Year | n_trades | Total PnL | Win Rate |
|---:|---:|---:|---:|
| 2010 |  1186 |  +70.77M | 57.1% |
| 2011 |  1148 |  -96.42M | 28.5% |
| 2012 |   944 | -105.47M | 31.2% |
| 2013 |   887 |  -12.26M | 37.9% |
| 2014 |  1182 | +447.30M | 62.2% |
| 2015 |  1175 | +535.84M | 55.8% |
| 2016 |  1217 | -166.45M | 29.8% |
| 2017 |  1217 |  -76.93M | 34.0% |
| 2018 |   967 | -336.75M | 32.4% |
| 2019 |  1196 | +312.94M | 54.7% |
| 2020 |  1223 | +222.34M | 50.9% |
| 2021 |   995 |  -85.51M | 44.5% |
| 2022 |   939 | -230.50M | 31.9% |
| 2023 |  1038 | -131.77M | 32.8% |
| 2024 |   946 | -152.23M | 43.4% |
| 2025 |  1066 | +291.24M | 47.6% |

## Final Honest Assessment

**Overall:**
- 10-seed sweep mean annual return: 2.19%
- Mean Sharpe: 0.220
- Mean MDD: 51.4%
- Positive offsets: 9/10

**Strengths:**
- 9/10 offsets positive (1 negative: V14_6 at -7.81%)
- 100% router agreement with ZigZag ground truth (regime detection works)
- Bull regime alpha: +¥1.64M / 48.7% win rate / +16万 per trade
- Strong 2010-2015 performance: 10/10 positive, mean +84.4%

**Weaknesses:**
- Bear regime -¥1.15M / 33.9% win rate — V14 IC voting loses money in bear
- 2016, 2017, 2018, 2022, 2023, 2024 — all bear years lose money
- OOS 2023-2025: only 5/10 positive, mean +0.18% — strategy doesn't survive OOS
- MDD -51.4% is institutional-unacceptable

**Comparison with V37 v3 baseline (different router):**
- V37 v3 baseline: 5-seed mean annual 4.57%, Sharpe 0.353, MDD -53.0%
- ZigZag Simple IC: 10-seed mean annual 2.19%, Sharpe 0.220, MDD -51.4%
- ZigZag Simple IC underperforms V37 baseline by 240bp annually

**Root cause of ZigZag Simple IC underperformance:**
V37 v3 uses V19 directional bear voting (factor selection based on historical
bear net returns, high vs low direction), which has slight alpha in bear markets
(PF > 1.2 even in bear). ZigZag Simple IC uses V14 IC voting in BOTH regimes,
but V14 IC voting was designed for bull markets. In bear markets it picks
stocks that have historically had high returns over recent 60 days — these are
the stocks about to fall the most when market reverses.

**Recommendation:**
- The V37 v3 baseline (5-seed mean annual 4.57%, with V19 directional bear voting)
  remains the best documented baseline.
- ZigZag Simple IC is NOT an improvement — it removes the V19 directional alpha.
- The ZigZag router itself is correct (100% agreement with truth), but the
  V14 IC voting underperforms in bear regimes regardless of router quality.

## All Artifacts Saved to Evidence/

```
evidence/sweep/zigzag_simple_ic_20261006/
├── V14_{0,2,4,6,8,10,12,14,16,18}/  # picks + meta per offset
│   ├── picks.json
│   ├── picks_meta.json
│   └── summary.json
├── akquant/V14_{0,2,4,6,8,10,12,14,16,18}/  # AKQuant per offset
│   ├── result.json
│   ├── ledger_audit.json
│   ├── trades.csv
│   ├── orders.csv
│   └── nav.csv
└── audit/  # comprehensive audit
    ├── cross_seed_summary.csv
    ├── period_split.csv
    ├── yearly_breakdown.csv
    ├── regime_purity.csv
    ├── factor_rotation.csv
    ├── holdings_overlap.csv
    ├── trade_attribution_by_regime.csv
    ├── trade_attribution_by_year.csv
    ├── trade_ledger_sample_V14_0_first_50.csv
    ├── V14_{0..18}/summary.json
    └── REPORT.md  # this file
```
