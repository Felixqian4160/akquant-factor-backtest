# akquant-factor-backtest

AKQuant-driven factor backtests reusing (read-only) the v10.2 panel from
`aurumq-rl`. AKQuant provides the Rust execution engine; this project is
a thin Python layer that:

1. loads the v10.2 panel
2. builds a single-symbol HS300 idx_close DataFrame with the factor
   embedded as a `bar.extra` column
3. runs `akquant.run_backtest` with a fixed execution contract
4. writes JSON metrics + HTML report into `evidence/<run_id>/`

## Why this project exists

We have two already-validated factor backtest engines in the workspace:

- `backtesting.py` (Python, event-driven, slow, used by
  `quant_workflow/scripts/v10_2_lightgbm/run_index_factor_threshold_backtest.py`)
- AKQuant (Rust core + Python bindings, fast, with `RiskManager`-level T+1)

This project is the **third independent executor** for the same factor
panel, so we can compare:

- `backtesting.py` baseline metrics
- AKQuant metrics under the same execution contract

If the two engines agree (within tolerance) we trust the simulated
alpha. If they diverge materially we have an executor-level bug to chase
before claiming alpha.

## Layout

```text
akquant-factor-backtest/
├── pyproject.toml
├── README.md
├── src/akquant_factor_backtest/
│   ├── __init__.py
│   └── panel_loader.py          # load v10.2 panel (read-only)
├── examples/
│   └── run_alpha_alpha042.py    # minimum demo
├── tests/                       # placeholder for unit tests
└── evidence/
    └── <YYYYMMDD_HHMMSS>/
        ├── metrics.json
        └── report.html
```

## Install

```bash
# Use system python (no venv per project rules).
/usr/bin/python3.12 -m pip install --break-system-packages \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    akquant polars pandas pyarrow
```

## Run

```bash
cd /media/felix/f/quant/akquant-factor-backtest
/usr/bin/python3.12 examples/run_alpha_alpha042.py
```

The script writes:

- `evidence/<run_id>/metrics.json` — full BacktestResult as JSON
- `evidence/<run_id>/report.html`  — AKQuant viz report

## Locked execution contract

- Symbol: **HS300 idx_close** (single-symbol index-level backtest)
- Signal: factor value at bar.close
- Rule: `signal > 5-day MA` -> all-in (target_pct = 1.0)
- Rule: `signal ≤ 5-day MA` -> flat (target_pct = 0.0)
- Commission: 25 bps per side (50 bps round-trip)
- Slippage: 0 (default)
- Warmup: first 5 bars skip signal (MA not yet defined)

## Scope discipline

- Single-symbol (HS300 index), single-factor demos only for now
- Multi-leg / multi-factor sweeps come after the AKQuant wiring is
  verified end-to-end against the `backtesting.py` baseline
- Reuses v10.2 panel **read-only** — no write-back to aurumq-rl
