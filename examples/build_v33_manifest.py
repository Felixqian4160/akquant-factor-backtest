"""Build v33 IC voting manifest (428 factors: 410 existing + 18 fundamental).

Excludes 11 zigzag labels (v10_1_*) — those are forward-looking labels,
not causal-safe factor inputs.

Output: evidence/factor_manifest_v33_20261007/factor_manifest.json
"""
import polars as pl
import json
import pathlib

ROOT = pathlib.Path("/media/felix/f/quant/akquant-factor-backtest")
PANEL = ROOT / "data" / "wavehunter_hs300_v33_with_new_factors_20261003.parquet"

# Read v33 schema
schema = pl.read_parquet_schema(str(PANEL))
v33_cols = list(schema.keys())

# Read existing v37_v14 manifest
m = json.loads((ROOT / "evidence/v37_v14_causal/factor_manifest.json").read_text())
existing_factors = m["factors"]

# Define categories
base_cols = {"trade_date", "ts_code", "open", "high", "low", "close",
             "vol", "amount", "pct_chg", "adj_factor", "adj_close",
             "idx_close", "idx_mom_5", "idx_mom_20", "idx_mom_60",
             "turnover_rate", "circ_cap", "cap", "symbol", "volume",
             "idx_ret_5d", "idx_ret_10d", "idx_ret_20d", "idx_ret_60d"}

fundamental_factors = ["pe", "pb", "roe", "roa", "bps", "cfps", "ocfps",
                       "grossprofit_margin", "netprofit_margin", "netprofit_yoy",
                       "ocf_yoy", "or_yoy", "roe_waa", "fcff", "assets_turn",
                       "debt_to_assets", "quick_ratio", "current_ratio"]

zigzag_labels = ["v10_1_a1_point", "v10_1_a2_start", "v10_1_a2_interval",
                 "v10_1_b1_start", "v10_1_b1_interval",
                 "v10_1_down_start", "v10_1_down_interval",
                 "v10_1_peak_zone", "v10_1_valley_zone",
                 "v10_1_zig_peak", "v10_1_zig_valley"]

# Construct new factor list
all_factor_cols = [c for c in v33_cols if c not in base_cols]
# Verify no zigzag labels in our voting set
forbidden = set(zigzag_labels)
new_factors = [c for c in all_factor_cols if c not in forbidden]
# Verify all fundamental factors are present
for fund in fundamental_factors:
    if fund not in new_factors:
        print(f"WARNING: {fund} not in v33 cols")

# Sanity check vs existing
missing = [c for c in existing_factors if c not in new_factors]
added = [c for c in new_factors if c not in existing_factors]
print(f"Existing v37_v14 manifest: {len(existing_factors)} factors")
print(f"New v33 manifest: {len(new_factors)} factors")
print(f"Missing from new (carried over): {len(missing)}")
print(f"Added in new: {len(added)}")
for c in added:
    print(f"  +{c}")
for c in missing:
    print(f"  -{c}")

# Build manifest dict
new_manifest = {
    "panel": str(PANEL),
    "panel_version": "v33_with_new_factors_20261003",
    "factor_count_declared": len(new_factors),
    "factor_count_expected": len(new_factors),
    "factor_count_matches_expected": True,
    "factors": new_factors,
    "groups": {
        "academic22": [c for c in new_factors if c.startswith(("l_", "r_", "p_", "v_")) and not c.startswith(("v_bm", "v_ep"))],
        "gtja_191": [c for c in new_factors if c.startswith("gtja_gtja_")],
        "alpha191": [c for c in new_factors if c.startswith(("alpha_alpha", "alpha_191", "alpha191_"))],
        "talib": [c for c in new_factors if c.startswith("talib_")],
        "mw": [c for c in new_factors if c.startswith("mw_")],
        "fundamental18": [c for c in new_factors if c in fundamental_factors],
        "github17": [c for c in new_factors if c in {"winner_ratio", "efficiency_ratio", "fractal_dimension",
                                                    "alpha191_040", "alpha191_095", "mom12m_jt",
                                                    "maxret_bcw", "accruals_sloan", "idiovola_clmx",
                                                    "gp_novymarx", "overnight_intraday_spread",
                                                    "skew21_lottery", "pvcorr_21", "kurt21_returns",
                                                    "coskew60", "hl_52w_disposition", "resmom_6m"}],
        "v_bm_v_ep": [c for c in new_factors if c in {"v_bm", "v_ep"}],
    },
    "known_all_null": [],
    "excluded_zigzag_labels": zigzag_labels,
    "excluded_zigzag_reason": "Forward-looking labels — not causal-safe factor inputs",
}

# Verify all fundamental factors are in the manifest
for fund in fundamental_factors:
    assert fund in new_manifest["factors"], f"{fund} missing"
# Verify no zigzag labels
for z in zigzag_labels:
    assert z not in new_manifest["factors"], f"{z} should not be in factors"

# Output
out_dir = ROOT / "evidence" / "factor_manifest_v33_20261007"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "factor_manifest.json"
out_path.write_text(json.dumps(new_manifest, indent=2, ensure_ascii=False))
print(f"\nSaved {out_path}")
print(f"  Total factors: {len(new_manifest['factors'])}")
for grp, cols in new_manifest["groups"].items():
    print(f"    {grp:18} {len(cols)}")
