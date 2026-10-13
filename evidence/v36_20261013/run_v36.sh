#!/bin/bash
# v36 signals + build + sim
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
export AKQ_PANEL=/media/felix/f/quant/akquant-factor-backtest/data/wavehunter_hs300_v36_20261013.parquet
export AKQ_MATRIX=/media/felix/f/quant/akquant-factor-backtest/evidence/v34_dedup_20261010/matrix_v36_ADJ.parquet
export AKQ_END=2026-08-19
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool414.json
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"

echo "=== signals pool414 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $FR --factset $FS --signals 99

echo "=== build o0 v36 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $FR --factset $FS --build --tag V5V36_RET_RANK_o0 --offset 0 $BASE \
    --select-trail 120 --select-metric mean --select-topn 5 --select-mode top --select-matrix m

echo "=== sim 2018-2025 o0 v36 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u examples/v42_run_akquant_v34.py \
    --tag V5V36_OOS_o0_2018_2025 \
    --picks-base evidence/v3_resonance_20261010/picks/V5V36_RET_RANK_o0 \
    --out-base evidence/v36_20261013 \
    --end 2025-12-31 --holding-bars 60 \
    --broker-rate 0.00025 --slippage 0.0005 --stamp-tax-rate 0.001 \
    --st-codes-path data/st_codes.json 2>&1 | tail -3

echo "V36_DONE"
