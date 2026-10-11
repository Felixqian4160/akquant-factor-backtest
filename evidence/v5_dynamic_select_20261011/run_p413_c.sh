#!/bin/bash
# 冠军敏感性扫描: V5_P413_RET_RANK 的 trail / topn 单变量变体 (o0)
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"

run_one() {
  TAG=$1; shift
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag ${TAG}_o0 --offset 0 $BASE "$@"
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag ${TAG}_o0 --runner v42 --holding-bars 60
}

run_one V5_P413_RET_T60     --select-trail 60  --select-metric mean --select-topn 5
run_one V5_P413_RET_T250    --select-trail 250 --select-metric mean --select-topn 5
run_one V5_P413_RET_T120_K8 --select-trail 120 --select-metric mean --select-topn 8

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_RET_T60', 'V5_P413_RET_T250', 'V5_P413_RET_T120_K8'):
    ensure_journal(t, t + '_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "P413_C_DONE"
