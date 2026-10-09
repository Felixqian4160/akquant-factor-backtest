#!/usr/bin/env bash
# Regenerate the full factorlib from canonical sources, then verify it.
#
#   bash REGENERATE.sh
#
# Library location : /media/felix/f/quant/akquant-factor-backtest/factorlib/
# Groups           : alpha (107) + gtja (177; 14 exact duplicates of alpha-* removed
#                    2026-10-10) + talib (77) + academic (22) + github (15;
#                    accruals_sloan / gp_novymarx removed 2026-10-10 — no financial data source)
#
# Steps (order matters — step 1 wipes and rebuilds the library root):
#   1. build_factorlib_v2.py       rebuild alpha + gtja + shared _ops
#   2. build_extended_factorlib.py add talib + academic + github
#   3. verify_factorlib.py         execute ALL modules on a synthetic panel
set -euo pipefail

BUILD_DIR="/media/felix/f/quant/akquant-factor-backtest/factorlib_build"
PY="/usr/bin/python3.12"
export PYTHONPATH="/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages"

"$PY" "$BUILD_DIR/build_factorlib_v2.py"
"$PY" "$BUILD_DIR/build_extended_factorlib.py"
"$PY" "$BUILD_DIR/verify_factorlib.py"
echo "REGENERATE_OK"
