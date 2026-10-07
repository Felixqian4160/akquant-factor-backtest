# Factor-Return Voting — Lookback Sweep (20/40/60 days)

**Generated:** 2026-10-07 07:14:03

Same contract across all three runs: V33 panel (428 factors), Causal ZigZag router, 21-bar hard cap, 25bps+10bps cost, single-run offset=0. **Only the lookback window varies.**

## Contract



## Lookback Sweep Result

| Lookback | Total Return | Annual | Sharpe | MDD | PF | Win | Trades |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **20 sessions** | **+15948.20%** | **+37.35%** | **1.551** | **32.1%** | **2.825** | **57.5%** | **1627** |
| **40 sessions** | **+958.85%** | **+15.89%** | **0.807** | **39.6%** | **1.842** | **53.8%** | **1616** |
| **60 sessions** | **+2279.84%** | **+21.91%** | **1.064** | **36.8%** | **2.361** | **55.8%** | **1587** |

## Headline Conclusion

**lb=20 is the clear winner** with +37.55% annualized, Sharpe 1.551, MDD -32.1%, PF 2.825. Sharper, deeper MDD, and PF higher than both lb=40 and lb=60.

Shorter lookback means the strategy **reacts faster** to recent factor return signals (the 20 most recent trading sessions). With longer lookback (60), the strategy is more stable but slower to react to regime shifts. With shorter lookback (20), it follows recent winners but is more sensitive to noise.

## Files Saved


