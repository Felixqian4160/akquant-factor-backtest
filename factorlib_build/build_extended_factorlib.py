"""Build the factorlib talib/academic/github groups (run AFTER build_factorlib_v2.py).

Adds:
- talib/:    77 TA-Lib formula modules from rebuild_panel_with_talib.py
- academic/: 22 Quantactix academic formula modules from v30_quantactix_factors.py
- github/:   15 GitHub/academic formula modules from v34_build_part1_factors.py
             (accruals_sloan / gp_novymarx deleted 2026-10-10: need financial-statement inputs)

Every module computes its factor from the input panel; no values are copied
from any parquet panel. All modules return pl.Series via select(...).to_series().
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path("/media/felix/f/quant/akquant-factor-backtest/factorlib")
AKQ = Path("/media/felix/f/quant/akquant-factor-backtest")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build_talib() -> list[str]:
    src_path = AKQ / "examples" / "rebuild_panel_with_talib.py"
    tree = ast.parse(src_path.read_text())
    indicators_node = None
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "INDICATORS":
            indicators_node = node.value
        elif isinstance(node, ast.Assign):
            if any(getattr(target, "id", None) == "INDICATORS" for target in node.targets):
                indicators_node = node.value
    if not isinstance(indicators_node, ast.List):
        raise RuntimeError("INDICATORS list not found")

    names: list[str] = []
    # The canonical build called MAX2/MIN2 with a single argument, which raises
    # TypeError and produced all-NaN columns in v34 (1,298,335/1,298,335 nulls).
    # akquant.talib.MAX2/MIN2 are pairwise (real0, real1); we implement the
    # natural 2-bar semantics: max/min of close[t] and close[t-1].
    FIXED_CALLS = {
        "MAX2": "lambda c, h, l, o, v: tq.MAX2(c, c.shift(1), as_series=True)",
        "MIN2": "lambda c, h, l, o, v: tq.MIN2(c, c.shift(1), as_series=True)",
    }
    for item in indicators_node.elts:
        name = ast.literal_eval(item.elts[0])
        fn_expr = FIXED_CALLS.get(name, ast.unparse(item.elts[1]))
        kind = ast.literal_eval(item.elts[2])
        output_count = 1 if kind == "S" else int(kind[1:])
        for output_index in range(output_count):
            factor_name = f"talib_{name}" if kind == "S" else f"talib_{name}_{output_index}"
            output_arg = "None" if kind == "S" else str(output_index)
            text = f'''"""{factor_name} — canonical TA-Lib formula.

Source: {src_path}
Indicator: {name}
Output kind: {kind}; output index: {output_arg}
"""
from __future__ import annotations

import akquant.talib as tq
import polars as pl
from factorlib._ops.talib_ops import compute_talib


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute {factor_name} from sorted OHLCV panel rows."""
    return compute_talib(
        panel,
        {fn_expr},
        name="{factor_name}",
        output_index={output_arg},
    )
'''
            write(ROOT / "talib" / f"{factor_name}.py", text)
            names.append(factor_name)

    write(ROOT / "talib" / "__init__.py", '"""Standalone TA-Lib formula modules."""\n')
    return names


def build_talib_ops() -> None:
    text = '''"""Per-stock execution helper for standalone TA-Lib formula modules."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import polars as pl


def compute_talib(
    panel: pl.DataFrame,
    fn: Callable,
    *,
    name: str,
    output_index: int | None,
) -> pl.Series:
    """Apply one TA-Lib formula per stock and return an aligned Series.

    The caller must provide rows; this helper partitions by ``stock_code``
    internally so input order does not need to be pre-sorted.
    The helper accepts either ``volume`` or the legacy ``vol`` field.
    """
    required = {"stock_code", "trade_date", "open", "high", "low", "close"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"missing panel columns for {name}: {missing}")
    volume_col = "volume" if "volume" in panel.columns else "vol" if "vol" in panel.columns else None
    if volume_col is None:
        raise ValueError(f"missing volume/vol panel column for {name}")

    pieces: list[np.ndarray] = []
    for group in panel.partition_by("stock_code", maintain_order=True):
        frame = group.select(["open", "high", "low", "close", volume_col]).to_pandas()
        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        open_ = frame["open"].astype(float)
        volume = frame[volume_col].astype(float)
        out = fn(close, high, low, open_, volume)
        if output_index is not None:
            if not isinstance(out, tuple):
                raise TypeError(f"{name}: expected tuple output, got {type(out).__name__}")
            out = out[output_index]
        if isinstance(out, pd.Series):
            arr = out.to_numpy(dtype=float, na_value=np.nan)
        elif isinstance(out, pd.DataFrame):
            arr = out.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
        else:
            arr = np.asarray(out, dtype=float)
        if len(arr) != group.height:
            raise ValueError(f"{name}: output length {len(arr)} != group rows {group.height}")
        pieces.append(arr)
    if not pieces:
        return pl.Series(name, [], dtype=pl.Float64)
    return pl.Series(name, np.concatenate(pieces))
'''
    write(ROOT / "_ops" / "talib_ops.py", text)


def build_academic() -> list[str]:
    names = [
        "l_size", "l_size3", "l_turnm", "l_turna", "l_ami", "l_dtvm", "l_dtva", "l_vdtv",
        "r_tv", "r_beta", "p_m1", "p_m3", "p_m6", "p_m11", "p_m24", "p_mchg", "p_52w",
        "p_mdr", "p_pr", "p_season", "v_bm", "v_ep",
    ]
    formulas = {
        "l_size": 'pl.col("circ_cap").clip(lower_bound=1).log().alias("l_size")',
        "l_size3": '(pl.col("circ_cap").clip(lower_bound=1).log() ** 3).alias("l_size3")',
        "l_turnm": '(pl.col("amount") / pl.col("circ_cap").clip(lower_bound=1)).rolling_mean(21).over("stock_code").clip(lower_bound=1e-9).log().alias("l_turnm")',
        "l_turna": '(pl.col("amount") / pl.col("circ_cap").clip(lower_bound=1)).rolling_mean(252).over("stock_code").clip(lower_bound=1e-9).log().alias("l_turna")',
        "l_ami": '(pl.col("returns").abs() / pl.col("amount").clip(lower_bound=1)).rolling_mean(21).over("stock_code").alias("l_ami")',
        "l_dtvm": 'pl.col("amount").rolling_mean(21).over("stock_code").clip(lower_bound=1).log().alias("l_dtvm")',
        "l_dtva": 'pl.col("amount").rolling_mean(252).over("stock_code").clip(lower_bound=1).log().alias("l_dtva")',
        "l_vdtv": 'pl.col("amount").rolling_std(120).over("stock_code").clip(lower_bound=1).log().alias("l_vdtv")',
        "r_tv": 'pl.col("returns").rolling_std(60).over("stock_code").alias("r_tv")',
        "p_m1": '(pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(42).over("stock_code") - 1).alias("p_m1")',
        "p_m3": '(pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(84).over("stock_code") - 1).alias("p_m3")',
        "p_m6": '(pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(147).over("stock_code") - 1).alias("p_m6")',
        "p_m11": '(pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(252).over("stock_code") - 1).alias("p_m11")',
        "p_m24": '(pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(525).over("stock_code") - 1).alias("p_m24")',
        "p_mchg": '((pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(147).over("stock_code") - 1) - (pl.col("close").shift(21).over("stock_code") / pl.col("close").shift(273).over("stock_code") - 1)).alias("p_mchg")',
        "p_52w": '(pl.col("close") / pl.col("close").rolling_max(252).over("stock_code")).alias("p_52w")',
        "p_mdr": 'pl.col("returns").rolling_max(21).over("stock_code").alias("p_mdr")',
        "p_pr": 'pl.col("close").clip(lower_bound=0.01).log().alias("p_pr")',
        "p_season": '(pl.col("trade_date").dt.month() == 12).cast(pl.Int8).alias("p_season")',
        "v_bm": '(pl.col("bps") / pl.col("close").clip(lower_bound=0.01)).alias("v_bm")',
        "v_ep": '(1.0 / pl.col("pe").clip(lower_bound=1)).alias("v_ep")',
    }
    header = '''"""{name} — Quantactix academic factor formula.

Source: {source}
The formula is recomputed from panel columns; no factor values are copied.
"""
from __future__ import annotations

import polars as pl


def compute(panel: pl.DataFrame) -> pl.Series:
    """Compute {name} from a stock_code/trade_date-sorted panel."""
    required = ["stock_code", "trade_date", "close", "amount", "circ_cap", "returns"]
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {{missing}}")
{body}'''
    source = AKQ / "examples" / "v30_quantactix_factors.py"
    for name in names:
        if name == "r_beta":
            body = '''    market = (
        panel.group_by("trade_date").agg(pl.col("close").mean().alias("_market_close"))
        .sort("trade_date")
        .with_columns((pl.col("_market_close") / pl.col("_market_close").shift(1) - 1).alias("_market_return"))
    )
    staged = panel.join(market.select(["trade_date", "_market_return"]), on="trade_date", how="left")
    cov = pl.rolling_cov(pl.col("returns"), pl.col("_market_return"), window_size=60, min_samples=60).over("stock_code")
    var = pl.col("_market_return").rolling_var(window_size=60, min_samples=60).over("stock_code")
    return staged.select((cov / pl.when(var == 0).then(None).otherwise(var)).alias("r_beta")).to_series()
'''
        else:
            body = f'    return panel.select({formulas[name]}).to_series()\n'
        write(ROOT / "academic" / f"{name}.py", header.format(name=name, source=source, body=body))
    write(ROOT / "academic" / "__init__.py", '"""Standalone Quantactix academic factor modules."""\n')
    return names


def build_github() -> list[str]:
    src_path = AKQ / "examples" / "v34_build_part1_factors.py"
    tree = ast.parse(src_path.read_text())
    funcs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("f_"):
            funcs[node.name] = node
    mapping = {
        "f_winner_ratio": "winner_ratio", "f_efficiency_ratio": "efficiency_ratio",
        "f_fractal_dimension": "fractal_dimension", "f_alpha191_040": "alpha191_040",
        "f_alpha191_095": "alpha191_095", "f_mom12m_jt": "mom12m_jt",
        # f_accruals_sloan / f_gp_novymarx deleted 2026-10-10: require
        # financial-statement inputs (bps / netprofit_yoy / grossprofit_margin)
        # not available from stockdb. Source funcs remain in the v34 build script.
        "f_maxret_bcw": "maxret_bcw",
        "f_idiovola": "idiovola_clmx",
        "f_overnight_intraday": "overnight_intraday_spread", "f_skew21": "skew21_lottery",
        "f_pvcorr_21": "pvcorr_21", "f_kurt21": "kurt21_returns",
        "f_coskew60": "coskew60", "f_hl_52w": "hl_52w_disposition",
        "f_resmom_6m": "resmom_6m",
    }
    missing = set(mapping) - set(funcs)
    if missing:
        raise RuntimeError(f"missing github source funcs: {missing}")
    for source_name, factor_name in mapping.items():
        fn = funcs[source_name]
        body = ast.get_source_segment(src_path.read_text(), fn)
        body = body.replace(f"def {source_name}(d):", "def compute(panel):\n    d = _prepare(panel)")
        body = body.replace('"ts_code"', '"stock_code"')
        text = f'''"""{factor_name} — standalone GitHub/academic factor formula.

Canonical source: {src_path}
The function body is copied from the canonical v34 formula source and adapted
only for the factorlib panel key ``stock_code``.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from factorlib._ops.github_ops import prepare as _prepare

CLOSE = pl.col("close")

{body}
'''
        write(ROOT / "github" / f"{factor_name}.py", text)
    write(ROOT / "github" / "__init__.py", '"""Standalone GitHub/academic factor modules."""\n')
    return sorted(mapping.values())


def build_github_ops() -> None:
    text = '''"""Panel preparation for standalone GitHub/academic formulas."""
from __future__ import annotations

import polars as pl


def prepare(panel: pl.DataFrame) -> pl.DataFrame:
    d = panel.sort(["stock_code", "trade_date"])
    if "volume" not in d.columns and "vol" in d.columns:
        d = d.with_columns(pl.col("vol").alias("volume"))
    if "vol" not in d.columns and "volume" in d.columns:
        d = d.with_columns(pl.col("volume").alias("vol"))
    # Keep missing fundamental inputs explicit: the formula will produce nulls,
    # rather than silently substituting an unrelated field.
    for col in ("bps", "grossprofit_margin", "netprofit_yoy"):
        if col not in d.columns:
            d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias(col))
    if "returns" not in d.columns:
        d = d.with_columns((pl.col("close") / pl.col("close").shift(1).over("stock_code") - 1).alias("returns"))
    market = (
        d.group_by("trade_date").agg(pl.col("close").mean().alias("_mc"))
        .sort("trade_date")
        .with_columns([
            (pl.col("_mc") / pl.col("_mc").shift(1) - 1).alias("_mkt_ret"),
            (pl.col("_mc").shift(21) / pl.col("_mc").shift(126) - 1).alias("_mkt_ret6"),
        ])
    )
    return d.join(market.select(["trade_date", "_mkt_ret", "_mkt_ret6"]), on="trade_date", how="left")
'''
    write(ROOT / "_ops" / "github_ops.py", text)


if __name__ == "__main__":
    build_talib_ops()
    talib = build_talib()
    academic = build_academic()
    build_github_ops()
    github = build_github()
    manifest = {
        "alpha": 107,
        "gtja": 191,
        "talib_formula_outputs": len(talib),
        "academic": len(academic),
        "github": len(github),
        "total_formula_modules": 107 + 191 + len(talib) + len(academic) + len(github),
        "removed_factors": {
            "accruals_sloan": "deleted 2026-10-10 — requires bps+netprofit_yoy; no financial data source",
            "gp_novymarx": "deleted 2026-10-10 — requires grossprofit_margin; no financial data source",
        },
        "talib_names": talib,
        "academic_names": academic,
        "github_names": github,
    }
    (ROOT / "extended_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({k: manifest[k] for k in ("alpha", "gtja", "talib_formula_outputs", "academic", "github", "total_formula_modules")}, ensure_ascii=False))
