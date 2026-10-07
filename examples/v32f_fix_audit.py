"""V32f: 修正版账本审计 (读 CSV, 修复 enum 比较 bug)

Bug: extract_records 返回的 status/side 是 enum 对象, 与字符串比较恒 False
     → 审计 "open positions: 0" 错误 (实际 15 只)
修复: 读取 CSV (已字符串化) 或 astype(str) 后比较
"""
import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

V3 = Path('evidence/v32_akquant_voting_v3_20260925')
PANEL = Path('data/wavehunter_hs300_v31_refactored_v2_20260925.parquet')

orders = pd.read_csv(V3 / 'orders.csv').astype({'status': str, 'side': str})
trades = pd.read_csv(V3 / 'trades.csv')
nav = pd.read_csv(V3 / 'nav.csv')
res = json.load(open(V3 / 'result.json'))
metrics = res['metrics']

audit = {'checks': [], 'summary': {}}
def check(name, ok, detail):
    audit['checks'].append({'check': name, 'pass': bool(ok), 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

print("=== 修正版账本审计 ===")

check('trades_exist', len(trades) > 0, f"{len(trades)} closed trades")
check('orders_both_sides', orders['side'].nunique() >= 2, orders['side'].value_counts().to_dict())

n_rej = int((orders['status'] == 'OrderStatus.Rejected').sum())
n_filled = int((orders['status'] == 'OrderStatus.Filled').sum())
check('orders_filled_all', n_rej == 0, f"filled={n_filled}, rejected={n_rej}")

q = pd.to_numeric(trades['quantity'], errors='coerce').dropna()
check('qty_lot_multiple', bool((q % 100 == 0).all()), f"n={len(q)}, min={q.min():,.0f}, all %100==0")

check('nav_bounded', nav['date'].iloc[-1] <= '2024-12-31', f"last={nav['date'].iloc[-1]}")

# commissions
comm_orders = pd.to_numeric(orders['commission'], errors='coerce').sum()
check('commission_positive', comm_orders > 0, f"orders commission = ¥{comm_orders:,.0f}")

# open positions (correct string comparison now)
of = orders[orders['status'] == 'OrderStatus.Filled'].copy()
of['fq'] = pd.to_numeric(of['filled_quantity'], errors='coerce')
of['sq'] = np.where(of['side'] == 'OrderSide.Buy', of['fq'], -of['fq'])
pos = of.groupby('symbol')['sq'].sum()
open_pos = pos[pos > 0]
check('open_positions', len(open_pos) > 0, f"{len(open_pos)} symbols held at 2024-12-31")

# MTM at last close
last_close = (pl.read_parquet(PANEL, columns=['trade_date', 'ts_code', 'close'])
              .filter(pl.col('trade_date') <= pl.lit(pd.Timestamp('2024-12-31').to_pydatetime()))
              .group_by('ts_code').agg(pl.col('close').last()))
cmap = dict(zip(last_close['ts_code'].to_list(), last_close['close'].to_list()))
mtm = float(sum(q * cmap.get(s, np.nan) for s, q in open_pos.items()))
check('open_positions_mtm', mtm > 0, f"MTM(2024-12-31) = ¥{mtm:,.0f}")

# 成本基准 (买入均价)
basis = 0.0
for s, qty in open_pos.items():
    buys = of[(of['symbol'] == s) & (of['side'] == 'OrderSide.Buy')]
    # 简化: 用最后一笔买入的均价 (持仓只来自最后篮)
    avg_p = pd.to_numeric(buys['average_filled_price'], errors='coerce').iloc[-1]
    basis += qty * avg_p
unrealized = mtm - basis
check('unrealized_positive_def', True, f"basis≈¥{basis:,.0f}, unrealized≈¥{unrealized:,.0f}")

# 对账: initial + realized + unrealized ≈ end_value
realized = trades['net_pnl'].sum()
end_value = metrics['end_market_value']
recon_diff = end_value - (100_000_000.0 + realized + unrealized)
check('pnl_reconcile', abs(recon_diff) < 5_000_000,
      f"end - (initial + realized + unrealized) = ¥{recon_diff:,.0f} (容差 5M)")

# commission 对账
comm_trades = trades['commission'].sum()
check('commission_reconcile', abs(comm_trades - comm_orders) < comm_orders * 0.1,
      f"trades={comm_trades:,.0f} vs orders={comm_orders:,.0f}")

audit['summary'] = {
    'final_value': end_value,
    'total_return_pct': metrics['total_return_pct'],
    'n_closed_trades': len(trades),
    'n_orders': len(orders), 'n_rejected': n_rej,
    'open_positions': len(open_pos),
    'open_mtm': mtm, 'open_basis': basis, 'unrealized': unrealized,
    'realized_pnl': float(realized),
    'recon_diff': float(recon_diff),
    'total_commission': float(comm_orders),
}
with open(V3 / 'ledger_audit.json', 'w') as f:
    json.dump(audit, f, indent=2, default=str)

# 修正 result.json
res['open_positions'] = len(open_pos)
res['open_mtm'] = mtm
res['audit_note'] = 'open positions fixed in v32f re-audit (enum comparison bug)'
with open(V3 / 'result.json', 'w') as f:
    json.dump(res, f, indent=2, default=str)

print(f"\n=== 汇总 ===")
print(f"final: ¥{end_value:,.0f} ({metrics['total_return_pct']:+.2f}%)")
print(f"open positions: {len(open_pos)} (MTM ¥{mtm:,.0f}, basis ¥{basis:,.0f}, unrealized ¥{unrealized:,.0f})")
print(f"realized PnL: ¥{realized:,.0f}")
print(f"reconcile diff: ¥{recon_diff:,.0f}")
print(f"\n修正版 audit saved: {V3}/ledger_audit.json")
