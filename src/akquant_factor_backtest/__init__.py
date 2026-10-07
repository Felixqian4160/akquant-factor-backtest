"""AKQuant factor backtest — bootstrap package.

Reuses (read-only) the v10.2 panel from aurumq-rl, runs factor-driven
backtests through the AKQuant Rust engine. Initial scope: one factor,
HS300 idx_close, fixed execution contract (T+1 open, hold 20d, 0.5%
round-trip).
"""
__version__ = "0.1.0"

from .panel_loader import load_idx_close, load_panel_for_factor

__all__ = ["load_idx_close", "load_panel_for_factor"]
