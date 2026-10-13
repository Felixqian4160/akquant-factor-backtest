#!/bin/bash
# H1 step 2: capped picks sim 2018-2025 realistic cost
set -e
cd /media/felix/f/quant/akquant-factor-backtest

# 把 capped picks 包装为 framework 期望的目录结构
SUB=evidence/h1_industry_cap_20261013/H1_capped2_o0
mkdir -p $SUB
cp evidence/v3_resonance_20261010/picks/V5_P413_RET_RANK_o0/picks_meta.json $SUB/picks_meta.json
cp evidence/h1_industry_cap_20261013/V5_P413_RET_RANK_o0_capped2.json $SUB/picks.json

# 过滤 2018+ 用于 OOS 测试
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
/usr/bin/python3.12 - <<PY
import json
p = json.load(open("$SUB/picks.json"))
o = {k: v for k, v in p.items() if k >= "2018-01-01"}
print("OOS 2018+ grids:", len(o))
json.dump(o, open("$SUB/picks.json", "w"))
PY

# Sim (realistic cost + ST filter)
echo "=== H1 sim (capped 2/industry, realistic, 2018-2025) ==="
/usr/bin/python3.12 -u examples/v42_run_akquant_v34.py \
    --tag H1_capped2_o0_2018_2025 \
    --picks-base evidence/h1_industry_cap_20261013 \
    --out-base evidence/h1_industry_cap_20261013 \
    --end 2025-12-31 --holding-bars 60 \
    --broker-rate 0.00025 --slippage 0.0005 --stamp-tax-rate 0.001 \
    --st-codes-path data/st_codes.json 2>&1 | tail -3
