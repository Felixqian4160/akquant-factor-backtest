# V33 Factor-Return Voting vs V33 IC Voting | Same-Config A/B

**Generated:** 2026-10-07 00:54:28

Both runs use the same panel (v33 with_new_factors, 428 voting factors after excluding 11 zigzag labels), the same Causal ZigZag router, the same 21-bar hard cap, the same 25bps+10bps cost model, and the same single-run offset=0 schedule. **The only difference is the selector**: IC ranking vs factor-return ranking.

## Selector Difference

```
V33 IC Voting:
  bull: V14 IC voting, MIN_VOTES=2, K_NOM=10
  bear: V14 IC voting, MIN_VOTES=4, K_NOM=10
  IC: 60-day Spearman, lag=21, FWD=20

V33 Factor-Return Voting:
  bull: factor-return ranking voting, MIN_VOTES=2, K_NOM=10, lookback=60 sessions
  bear: factor-return ranking voting, MIN_VOTES=4, K_NOM=10, lookback=60 sessions
  factor score = mean of past 60 sessions' (top10 - bot10) net20 portfolio diff
```

## Result Comparison (Single-Run Offset=0)

| Metric | V33 IC Voting | V33 Factor-Return Voting | Δ |
|---|---:|---:|---:|
| Total return | +41.07% | +2279.84% | +2238.77% |
| Annualized | +2.17% | +21.91% | +19.73% |
| Sharpe | 0.223 | 1.064 | +0.840 |
| Max drawdown | 61.9% | 36.8% | -25.1% |
| Profit factor | 1.293 | 2.361 | +1.068 |
| Win rate | 48.5% | 55.8% | +7.4% |
| Closed trades | 1704 | 1587 | -117 |
| Avg holding | 9.3 bars | 11.2 bars | +1.8 |
| Commission | ¥137.8M | ¥377.9M | ¥+240.1M |
| Rejected orders | 688 | 791 | +103 |

**n_picks:** IC 195 (bull=105 bear=90) | FactorRank 195 (bull=104 bear=91)

## Files Saved

```
evidence/sweep/v33_ic_20261007/                 # IC voting baseline
evidence/sweep/v33_factorrank_20261007/         # Factor-return voting variant
evidence/sweep/v33_ic_vs_factorrank_equity.png  # Combined comparison chart
```
