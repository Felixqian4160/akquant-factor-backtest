# AKQuant rebal-aware factor sweep (ranked)

- engine: akquant 0.3.64
- factors: 340 (pre-dedup)
- dedup: kept 912 unique, dropped 108
- rebal periods: [5, 10, 20] (days)
- MA lookback: 5 days
- target_pct: 0.99 (when long)
- commission: 25bps per side; slippage: 0
- elapsed: 382.1s (6.4min)

## Top-20 rankings (4 keys)

- top20 by **cumulative_return** → `top_by_cumulative_return.csv`
- top20 by **annualized_return** → `top_by_annualized_return.csv`
- top20 by **sharpe** → `top_by_sharpe.csv`
- top20 by **win_rate** → `top_by_win_rate.csv`

### Top 10 by cumulative_return

| rank | factor | rebal | metric | ann% | sharpe | win% | return% | MDD% | trades |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gtja_gtja_143` | 5 | +101.964 | +3.15 | +1.346 | 53.57 | +101.96 | 74.62 | 196 |
| 2 | `alpha_alpha050` | 20 | +101.910 | +3.15 | +0.800 | 54.55 | +101.91 | 65.60 | 66 |
| 3 | `gtja_gtja_174` | 20 | +101.733 | +3.15 | +0.817 | 60.87 | +101.73 | 67.10 | 69 |
| 4 | `gtja_gtja_080` | 20 | +101.665 | +3.15 | +0.777 | 62.90 | +101.67 | 67.05 | 62 |
| 5 | `alpha_alpha046` | 5 | +101.157 | +3.13 | +1.604 | 59.14 | +101.16 | 55.18 | 279 |
| 6 | `gtja_gtja_132` | 20 | +101.148 | +3.13 | +0.829 | 53.52 | +101.15 | 64.15 | 71 |
| 7 | `gtja_gtja_126` | 10 | +101.123 | +3.13 | +1.100 | 52.71 | +101.12 | 72.65 | 129 |
| 8 | `gtja_gtja_015` | 20 | +101.123 | +3.13 | +0.807 | 55.22 | +101.12 | 65.66 | 67 |
| 9 | `gtja_gtja_175` | 20 | +101.097 | +3.13 | +0.804 | 64.18 | +101.10 | 65.89 | 67 |
| 10 | `alpha_alpha068` | 20 | +101.049 | +3.13 | +0.777 | 64.52 | +101.05 | 65.16 | 62 |

### Top 10 by annualized_return

| rank | factor | rebal | metric | ann% | sharpe | win% | return% | MDD% | trades |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gtja_gtja_143` | 5 | +0.032 | +3.15 | +1.346 | 53.57 | +101.96 | 74.62 | 196 |
| 2 | `alpha_alpha050` | 20 | +0.032 | +3.15 | +0.800 | 54.55 | +101.91 | 65.60 | 66 |
| 3 | `gtja_gtja_174` | 20 | +0.031 | +3.15 | +0.817 | 60.87 | +101.73 | 67.10 | 69 |
| 4 | `gtja_gtja_080` | 20 | +0.031 | +3.15 | +0.777 | 62.90 | +101.67 | 67.05 | 62 |
| 5 | `alpha_alpha046` | 5 | +0.031 | +3.13 | +1.604 | 59.14 | +101.16 | 55.18 | 279 |
| 6 | `gtja_gtja_132` | 20 | +0.031 | +3.13 | +0.829 | 53.52 | +101.15 | 64.15 | 71 |
| 7 | `gtja_gtja_126` | 10 | +0.031 | +3.13 | +1.100 | 52.71 | +101.12 | 72.65 | 129 |
| 8 | `gtja_gtja_015` | 20 | +0.031 | +3.13 | +0.807 | 55.22 | +101.12 | 65.66 | 67 |
| 9 | `gtja_gtja_175` | 20 | +0.031 | +3.13 | +0.804 | 64.18 | +101.10 | 65.89 | 67 |
| 10 | `alpha_alpha068` | 20 | +0.031 | +3.13 | +0.777 | 64.52 | +101.05 | 65.16 | 62 |

### Top 10 by sharpe

| rank | factor | rebal | metric | ann% | sharpe | win% | return% | MDD% | trades |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `gtja_gtja_018` | 5 | +1.894 | +2.93 | +1.894 | 47.19 | +92.36 | 56.41 | 392 |
| 2 | `alpha_alpha089` | 5 | +1.892 | +3.01 | +1.892 | 52.94 | +95.60 | 54.76 | 391 |
| 3 | `mw_up_days_5d` | 5 | +1.880 | -0.10 | +1.880 | 45.90 | -2.17 | 56.47 | 390 |
| 4 | `gtja_gtja_014` | 5 | +1.874 | +2.91 | +1.874 | 47.66 | +91.70 | 56.47 | 384 |
| 5 | `gtja_gtja_022` | 5 | +1.873 | -0.05 | +1.873 | 49.48 | -1.05 | 56.69 | 386 |
| 6 | `alpha_alpha022` | 5 | +1.856 | -0.06 | +1.856 | 53.56 | -1.32 | 55.47 | 379 |
| 7 | `alpha_alpha088` | 5 | +1.825 | -0.08 | +1.825 | 51.50 | -1.74 | 54.77 | 367 |
| 8 | `gtja_gtja_020` | 5 | +1.824 | -0.07 | +1.824 | 47.96 | -1.65 | 55.98 | 367 |
| 9 | `gtja_gtja_080` | 5 | +1.822 | +2.96 | +1.822 | 53.31 | +93.48 | 55.33 | 362 |
| 10 | `gtja_gtja_019` | 5 | +1.818 | -0.07 | +1.818 | 47.67 | -1.59 | 55.95 | 365 |

### Top 10 by win_rate

| rank | factor | rebal | metric | ann% | sharpe | win% | return% | MDD% | trades |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `alpha_alpha026` | 20 | +80.000 | +0.01 | +0.215 | 80.00 | +0.23 | 52.32 | 5 |
| 2 | `alpha_alpha026` | 10 | +70.000 | +0.01 | +0.302 | 70.00 | +0.14 | 52.28 | 10 |
| 3 | `gtja_gtja_111` | 10 | +66.667 | -0.00 | +0.162 | 66.67 | -0.05 | 50.25 | 3 |
| 4 | `alpha_alpha043` | 20 | +66.176 | +0.07 | +0.792 | 66.18 | +1.49 | 59.55 | 68 |
| 5 | `alpha_alpha092` | 20 | +64.706 | +3.03 | +0.806 | 64.71 | +96.40 | 57.28 | 68 |
| 6 | `gtja_gtja_085` | 20 | +64.706 | +0.06 | +0.791 | 64.71 | +1.40 | 59.56 | 68 |
| 7 | `alpha_alpha068` | 20 | +64.516 | +3.13 | +0.777 | 64.52 | +101.05 | 65.16 | 62 |
| 8 | `gtja_gtja_048` | 20 | +64.516 | +3.11 | +0.772 | 64.52 | +100.00 | 57.66 | 62 |
| 9 | `gtja_gtja_109` | 20 | +64.179 | +0.07 | +0.784 | 64.18 | +1.52 | 59.54 | 67 |
| 10 | `gtja_gtja_175` | 20 | +64.179 | +3.13 | +0.804 | 64.18 | +101.10 | 65.89 | 67 |

## Intersection: in top-20 of ALL 4 keys (0 entries)

These are the most consistent winners across ranking metrics.

| factor | rebal | return% | ann% | sharpe | win% | MDD% | trades |
|:---|---:|---:|---:|---:|---:|---:|---:|

## Distribution (over 912 dedup'd factor × rebal combos)

- Sharpe: min=+0.000  median=+1.109  max=+1.894
- Return%: min=-2.17  median=+1.28  max=+101.96
- Annualized%: min=-0.10  median=+0.06  max=+3.15
- PF: min=0.000  median=1.359  max=6.549
- Win%: min=0.00  median=54.23  max=80.00
- combos with positive return: 739/912
- combos with positive return AND MDD<40%: 0/912

## Notes / caveats

- Single-symbol HS300 idx_close backtest; no cross-sectional ranking
- Cost model: 25bps commission per side, slippage=0; actual A-share cost is 30-40bps with stamp tax
- Strategy: at every rebal day, compare factor value to 5-day MA → all-in or flat
- All metrics in this report use AKQuant's Rust engine; not validated against backtesting.py
- High Sharpe + negative return is a known pattern with this rule: low volatility, high turnover
