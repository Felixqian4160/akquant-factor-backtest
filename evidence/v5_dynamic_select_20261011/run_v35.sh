#!/bin/bash
# v35 终极验证: 池416 (413 + 3 新因子) 信号 + 冠军重跑
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
export AKQ_PANEL=/media/felix/f/quant/akquant-factor-backtest/data/wavehunter_hs300_v35_20261012.parquet
export AKQ_MATRIX=/media/felix/f/quant/akquant-factor-backtest/evidence/v34_dedup_20261010/matrix_v35_ADJ.parquet
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool416.json
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"

echo "=== signals pool416 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $FR --factset $FS --signals 99

echo "=== champion-on-v35 build ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $FR --factset $FS --build --tag V5V35_RET_RANK_o0 --offset 0 $BASE \
    --select-trail 120 --select-metric mean --select-topn 5

echo "=== sim ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $FR --run 1 --tag V5V35_RET_RANK_o0 --runner v42 --holding-bars 60

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
ensure_journal('V5V35_RET_RANK', 'V5V35_RET_RANK_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "V35_DONE"
