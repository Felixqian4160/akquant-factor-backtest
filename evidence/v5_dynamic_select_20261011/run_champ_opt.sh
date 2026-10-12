#!/bin/bash
# 冠军参数重调优: 基底参数 (v3 时代调的) 在新选择器上的单变量重扫
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
SEL="--select-trail 120 --select-metric mean --select-topn 5"

run_one() {
  TAG=$1; shift
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag ${TAG}_o0 --offset 0 "$@"
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag ${TAG}_o0 --runner v42 --holding-bars 60
}

BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"

run_one V5_P413_CH_CD6   --hold-pctl 0.65 --cooldown 6 --max-hold-bars 60 --n-pos 15 --rank-by score $SEL
run_one V5_P413_CH_POS20 --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 20 --rank-by score $SEL
run_one V5_P413_CH_POS10 --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 10 --rank-by score $SEL
run_one V5_P413_CH_BP75  --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score --bear-pctl 0.75 $SEL
run_one V5_P413_CH_K4    --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score --select-trail 120 --select-metric mean --select-topn 4
run_one V5_P413_CH_K3    --hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score --select-trail 120 --select-metric mean --select-topn 3

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_CH_CD6', 'V5_P413_CH_POS20', 'V5_P413_CH_POS10', 'V5_P413_CH_BP75',
          'V5_P413_CH_K4', 'V5_P413_CH_K3'):
    ensure_journal(t, t + '_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "CHAMP_OPT_DONE"
