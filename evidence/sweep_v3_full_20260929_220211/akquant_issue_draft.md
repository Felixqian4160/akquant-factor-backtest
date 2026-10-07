## Bug: `force_close_applied=True` flag is set but open_position_count never drops to 0

### Reproduction (Python, single-symbol)

```python
import akquant as aq
from akquant import Bar, Strategy
import pandas as pd

# Long-only strategy with NO on_stop / NO manual close
class LongOnly(Strategy):
    warmup = 5
    def on_bar(self, bar):
        if self.get_position(bar.symbol) == 0:
            self.order_target_percent(target_percent=0.99, symbol=bar.symbol)

# 5498 daily bars (HS300 idx_close, 2004-2026)
df = pd.DataFrame({...})  # OHLCV daily

result = aq.run_backtest(
    data=df, strategy=LongOnly,
    initial_cash=1_000_000.0, commission_rate=0.0025,
    t_plus_one=False, history_depth=5,
)
m = result.metrics_df.loc[:, 'value']
print(m['open_position_count'])        # → 1.0  (BUG: should be 0 if engine forced close)
print(m['total_return_pct'])          # → +378% (inflated by unrealized)
print(m['unrealized_pnl'])            # → +3,784,124 (large)
```

### Expected
- `open_position_count == 0` after backtest ends
- `unrealized_pnl == 0` (position closed at last bar close)
- `total_return_pct == closed_only_pnl / initial_market_value`

### Actual
- `open_position_count == 1.0` (force-close did NOT happen)
- `unrealized_pnl` is huge (3.7x initial cash)
- `force_close_applied=True` is set in metrics → **misleading**

### Root cause
Orders in akquant 0.3.64 fill T+1 (next bar). On the **last bar** of the backtest:
- strategy's `on_bar` calls `order_target_percent(target=0.0)` → submits close order
- but there's no "next bar" for the order to fill
- Rust engine appears to mark `force_close_applied=True` regardless

Same issue observed with `on_stop` callback: orders submitted there also never fill because they're submitted after the backtest engine has already computed final state.

### Proposed fix
Either:
1. Mark `force_close_applied` only if `open_position_count == 0` (truthfulness)
2. Auto-close open positions at the last bar's `close` price before computing final metrics (true mark-to-market close)
3. Allow users to opt-in to a `terminal_liquidation=True` flag in `run_backtest(...)`

### Workaround
Compute closed-only return manually:
```python
total_pnl = metrics['total_pnl']
upnl = metrics['unrealized_pnl']
initial = metrics['initial_market_value']
closed_only_return = (total_pnl - upnl) / initial * 100
```

### Environment
- akquant 0.3.64 (PyPI)
- Python 3.12.3
- Linux x86_64
- Single-symbol stock (HS300 idx_close proxy), 5498 daily bars
