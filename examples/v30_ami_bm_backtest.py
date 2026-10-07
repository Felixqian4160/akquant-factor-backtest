"""V30: l_ami + v_bm 双因子组合策略 (真实 AKQuant sim backtest)

策略:
- 每 20 个交易日调仓
- 每天横截面 rank 计算 l_ami (Amihud) 和 v_bm (账面市值比)
- 综合分数 = (rank_ami + rank_bm) / 2 (双因子等权)
- 选 top-20 股票 (等权)
- T+1 open 买入, 持有 20 天
- 0.5% round-trip cost (commission + slippage)

评估窗口: 2022-2024 (3 年, 用户合同: 熊市只看 3 年)
"""

import polars as pl
import pandas as pd
import numpy as np
import akquant as aq
from datetime import date, datetime
from typing import Dict, Any
import sys
import os
import json

PANEL = "data/wavehunter_hs300_with_talib_20260924.parquet"
OUTPUT_DIR = "evidence/daily_factor_reselect/pipeline_v30"
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"

INITIAL_CASH = 100_000_000.0
COMMISSION_RATE = 0.0025
SLIPPAGE = {"type": "percent", "value": 0.0010}
LOT_SIZE = 100
MAX_POSITIONS = 20
REBALANCE_DAYS = 20

print("=" * 80)
print("V30: l_ami + v_bm 双因子组合策略 (2022-2024 真实 sim)")
print("=" * 80)

# === Load panel + compute factors ===
print("\n=== Load panel + compute factors ===")
df = pl.read_parquet(PANEL).select([
    'trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount',
    'cap', 'circ_cap', 'pe', 'pb', 'bps'
])
df_pd = df.to_pandas()
df_pd = df_pd.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

# Compute l_ami (Amihud)
df_pd['daily_ret'] = df_pd.groupby('ts_code')['close'].transform(lambda x: x.pct_change())
df_pd['l_ami'] = df_pd.groupby('ts_code').apply(
    lambda g: (g['daily_ret'].abs() / g['amount'].clip(lower=1)).rolling(21).mean()
).reset_index(level=0, drop=True)

# Compute v_bm (book-to-market)
df_pd['v_bm'] = df_pd['bps'] / df_pd['close'].clip(lower=0.01)

df_pd['date'] = pd.to_datetime(df_pd['trade_date']).dt.date
print(f"Loaded: {df_pd.shape}")
print(f"  l_ami null: {df_pd['l_ami'].isna().sum()}")
print(f"  v_bm null: {df_pd['v_bm'].isna().sum()}")

# === AKQuant Strategy ===
print("\n=== Define AKQuant Strategy ===")

# Build date index for rebalance
all_dates = sorted(df_pd['date'].unique())
rebal_dates = [all_dates[i] for i in range(0, len(all_dates), REBALANCE_DAYS)]
print(f"Rebalance dates: {len(rebal_dates)} (every {REBALANCE_DAYS} trading days)")

# Build per-date factor ranks (precompute to avoid slow sim)
print("\n=== Precompute per-date factor ranks ===")
df_pd = df_pd.dropna(subset=['l_ami', 'v_bm'])
df_pd['rank_ami'] = df_pd.groupby('date')['l_ami'].rank(pct=True, na_option='keep')
df_pd['rank_bm'] = df_pd.groupby('date')['v_bm'].rank(pct=True, na_option='keep')
df_pd['composite_score'] = (df_pd['rank_ami'] + df_pd['rank_bm']) / 2

# Pick top-20 per rebalance date
picks = {}
for d in rebal_dates:
    sub = df_pd[df_pd['date'] == d].copy()
    if len(sub) < MAX_POSITIONS:
        continue
    sub = sub.dropna(subset=['composite_score'])
    sub = sub.nlargest(MAX_POSITIONS, 'composite_score')
    picks[d] = sub['ts_code'].tolist()

print(f"Picks per rebalance date: avg = {np.mean([len(v) for v in picks.values()]):.1f}")

# Convert to AKQuant data
print("\n=== Build AKQuant data ===")
akquant_df = df_pd[['trade_date', 'ts_code', 'open', 'high', 'low', 'close', 'vol', 'amount']].copy()
akquant_df = akquant_df.rename(columns={'trade_date': 'date'})
akquant_df['volume'] = akquant_df['vol'] * 1e9  # ensure non-zero volume
akquant_df['date'] = pd.to_datetime(akquant_df['date'])

akquant_df.to_parquet('/tmp/v30_panel.parquet')
print(f"Panel saved: {akquant_df.shape}")

# === Define AKQuant Strategy class ===
class V30AmiBmStrategy(aq.Strategy):
    """V30: Amihud + 账面市值比 双因子等权 top-20 策略"""

    def on_init(self, ctx):
        # Read rebalance picks
        self.picks = picks
        self.max_pos = MAX_POSITIONS
        self.rebal_days = REBALANCE_DAYS
        self.last_rebal_date = None
        self.target_holdings = {}  # symbol -> target weight

    def on_bar(self, bar):
        # Get current date
        current_date = bar.trading_date.date() if hasattr(bar.trading_date, 'date') else bar.trading_date

        # Check if it's a rebalance day
        if current_date not in self.picks:
            return

        new_picks = self.picks[current_date]

        # Compute scores
        scores = {sym: 1.0 for sym in new_picks}

        # Rebalance
        self.rebalance_to_topn(scores, top_n=self.max_pos, weight_mode='equal', long_only=True, liquidate_unmentioned=True)
        self.last_rebal_date = current_date

print("\n=== Run backtest ===")
result = aq.run_backtest(
    data='/tmp/v30_panel.parquet',
    strategy=V30AmiBmStrategy,
    initial_cash=INITIAL_CASH,
    commission_rate=COMMISSION_RATE,
    slippage=SLIPPAGE,
    t_plus_one=False,
    fill_policy=aq.NextOpen(),
    lot_size=LOT_SIZE,
    history_depth=30,
)

# Print metrics
print("\n=== Metrics ===")
metrics = result.metrics_df
print(metrics)

print(f"\n=== Summary ===")
m = result.metrics
print(f"Sharpe Ratio: {m.get('sharpe_ratio', 0):.4f}")
print(f"Max Drawdown: {m.get('max_drawdown_pct', 0):.2f}%")
print(f"Profit Factor: {m.get('profit_factor', 0):.4f}")
print(f"Win Rate: {m.get('win_rate', 0):.4f}")
print(f"Total Return: {m.get('total_return_pct', 0):.2f}%")
print(f"Initial Market Value: ¥{m.get('initial_market_value', 0):,.0f}")
print(f"End Market Value: ¥{m.get('end_market_value', 0):,.0f}")
print(f"Closed Trades: {m.get('closed_trade_count', 0)}")
print(f"Open Positions: {m.get('open_position_count', 0)}")
print(f"Total Commission: ¥{m.get('total_commission', 0):,.0f}")
print(f"Unrealized PnL: ¥{m.get('unrealized_pnl', 0):,.0f}")
print(f"Avg Trade Bars: {m.get('avg_trade_bars', 0):.1f}")

# Per-year breakdown
print("\n=== Per-Year Returns ===")
equity_curve = result.equity_curve
ec_df = pd.DataFrame(list(equity_curve.items()), columns=['date', 'value'])
ec_df['date'] = pd.to_datetime(ec_df['date'])
ec_df['year'] = ec_df['date'].dt.year
for y in [2022, 2023, 2024]:
    yr = ec_df[ec_df['year'] == y]
    if len(yr) > 1:
        ret = (yr['value'].iloc[-1] / yr['value'].iloc[0] - 1) * 100
        print(f"  {y}: {ret:+.2f}%")

# Save evidence
os.makedirs(OUTPUT_DIR, exist_ok=True)
final_value = m.get('end_market_value', INITIAL_CASH)
result_dict = {
    'job_id': 'v30_ami_bm_2022_2024',
    'panel': PANEL,
    'start_date': START_DATE,
    'end_date': END_DATE,
    'strategy': 'V30AmiBmStrategy',
    'config': {
        'rebal_days': REBALANCE_DAYS,
        'max_positions': MAX_POSITIONS,
        'commission_rate': COMMISSION_RATE,
        'slippage': SLIPPAGE,
        'lot_size': LOT_SIZE,
    },
    'metrics': {k: float(v) if hasattr(v, 'item') else v for k, v in m.items()},
    'final_value': float(final_value),
    'total_return_pct': float(m.get('total_return_pct', 0)),
    'final_return_pct': float(m.get('total_return_pct', 0)),
}

with open(f"{OUTPUT_DIR}/result.json", 'w') as f:
    json.dump(result_dict, f, indent=2, default=str)

print(f"\nSaved: {OUTPUT_DIR}/result.json")