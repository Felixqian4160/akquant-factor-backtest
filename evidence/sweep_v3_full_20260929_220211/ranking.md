# AKQuant rebal-aware v2 sweep (force-close at end)

- engine: akquant 0.3.64
- factors: 393 (pre-dedup)
- dedup: kept 909 unique, dropped 51
- rebal periods: [5, 10, 20] (days)
- MA lookback: 5 days
- target_pct: 0.99 (when long)
- commission: 25bps per side; slippage: 0
- elapsed: 387.2s (6.5min)
- force-close applied: TRUE
- audit: 397/909 still have open_position>0 (target: 0)

## rebal=5 — Top 10 by cumulative return

| rank | factor | ret% | ann% | sharpe | win% | MDD% | trades | open |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gtja_gtja_143` | +101.96 | +3.15 | +1.346 | 53.57 | 74.62 | 196 | 1 |
| 2 | `alpha_alpha046` | +101.16 | +3.13 | +1.604 | 59.14 | 55.18 | 279 | 1 |
| 3 | `gtja_gtja_135` | +100.05 | +3.11 | +1.131 | 47.10 | 60.72 | 138 | 1 |
| 4 | `gtja_gtja_169` | +99.43 | +3.09 | +1.185 | 55.63 | 56.86 | 151 | 1 |
| 5 | `alpha_alpha053` | +99.37 | +3.09 | +1.530 | 52.55 | 55.64 | 255 | 1 |
| 6 | `gtja_gtja_001` | +99.27 | +3.09 | +1.785 | 60.23 | 53.61 | 347 | 1 |
| 7 | `gtja_gtja_134` | +98.94 | +3.08 | +1.606 | 53.74 | 57.93 | 281 | 1 |
| 8 | `gtja_gtja_147` | +98.58 | +3.08 | +1.392 | 49.52 | 56.29 | 210 | 1 |
| 9 | `mw_ret_60d` | +98.54 | +3.07 | +1.574 | 51.67 | 60.71 | 269 | 1 |
| 10 | `alpha_alpha005` | +98.40 | +3.07 | +1.561 | 54.34 | 54.11 | 265 | 1 |

## rebal=10 — Top 10 by cumulative return

| rank | factor | ret% | ann% | sharpe | win% | MDD% | trades | open |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gtja_gtja_055` | +101.03 | +3.13 | +1.119 | 53.73 | 58.70 | 134 | 1 |
| 2 | `gtja_gtja_187` | +101.02 | +3.13 | +1.063 | 50.83 | 60.57 | 120 | 1 |
| 3 | `gtja_gtja_089` | +100.93 | +3.13 | +1.203 | 63.23 | 56.80 | 155 | 1 |
| 4 | `gtja_gtja_059` | +100.89 | +3.13 | +1.096 | 52.34 | 58.67 | 128 | 1 |
| 5 | `gtja_gtja_135` | +100.23 | +3.11 | +0.968 | 46.00 | 59.57 | 100 | 1 |
| 6 | `alpha_alpha096` | +100.22 | +3.11 | +1.125 | 51.47 | 64.91 | 136 | 1 |
| 7 | `mw_rs_10d` | +100.21 | +3.11 | +1.213 | 53.46 | 56.38 | 159 | 1 |
| 8 | `gtja_gtja_040` | +100.02 | +3.11 | +1.109 | 50.76 | 56.38 | 132 | 1 |
| 9 | `gtja_gtja_173` | +99.95 | +3.11 | +1.059 | 50.42 | 63.33 | 119 | 1 |
| 10 | `gtja_gtja_039` | +99.94 | +3.11 | +1.117 | 58.21 | 54.82 | 134 | 1 |

## rebal=20 — Top 10 by cumulative return

| rank | factor | ret% | ann% | sharpe | win% | MDD% | trades | open |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `alpha_alpha050` | +101.91 | +3.15 | +0.800 | 54.55 | 65.60 | 66 | 1 |
| 2 | `gtja_gtja_174` | +101.73 | +3.15 | +0.817 | 60.87 | 67.10 | 69 | 1 |
| 3 | `gtja_gtja_080` | +101.67 | +3.15 | +0.777 | 62.90 | 67.05 | 62 | 1 |
| 4 | `gtja_gtja_132` | +101.15 | +3.13 | +0.829 | 53.52 | 64.15 | 71 | 1 |
| 5 | `gtja_gtja_015` | +101.12 | +3.13 | +0.807 | 55.22 | 65.66 | 67 | 1 |
| 6 | `gtja_gtja_175` | +101.10 | +3.13 | +0.804 | 64.18 | 65.89 | 67 | 1 |
| 7 | `alpha_alpha068` | +101.05 | +3.13 | +0.777 | 64.52 | 65.16 | 62 | 1 |
| 8 | `alpha_alpha070` | +101.03 | +3.13 | +0.784 | 60.32 | 70.82 | 63 | 1 |
| 9 | `gtja_gtja_062` | +101.01 | +3.13 | +0.802 | 55.22 | 67.59 | 67 | 1 |
| 10 | `gtja_gtja_170` | +100.90 | +3.13 | +0.771 | 59.02 | 64.50 | 61 | 1 |

## Intersection: in top-20 of ALL 4 keys (0 entries)

(empty — no factor is simultaneously top-20 in all 4 rankings)

## Distribution (over 909 dedup'd factor × rebal combos)

- Sharpe: min=+0.000  median=+1.112  max=+1.894
- Return%: min=-2.17  median=+1.28  max=+101.96
- Annualized%: min=-0.10  median=+0.06  max=+3.15
- PF: min=0.000  median=1.361  max=2.831
- Win%: min=0.00  median=54.29  max=66.18
- Trades: min=0  median=135  max=392
- combos with positive return: 742/909
- combos with positive return AND MDD<40%: 0/909

### Per-rebal breakdown

| rebal | n | median ret% | median sharpe | median trades | positive |
|---:|---:|---:|---:|---:|---:|
| 5 | 305 | +0.32 | +1.620 | 288 | 177/305 |
| 10 | 303 | +1.27 | +1.113 | 135 | 276/303 |
| 20 | 301 | +1.75 | +0.799 | 68 | 289/301 |

## Notes / caveats

- force_close_applied=True: strategy closes any open position on the last bar
- WITHOUT force-close, ~42% of factors hit a 'long at end' cap (~+100% return from unrealized PnL)
- WITH force-close, returns reflect realized closed-trade PnL only
- median trades per rebal period: 5→280, 10→135, 20→68 (lower rebal period ⇒ more rebal events; trade count is bounded by signal crossings)
- cost model: 25bps commission per side, slippage=0; actual A-share cost is 30-40bps with stamp tax
