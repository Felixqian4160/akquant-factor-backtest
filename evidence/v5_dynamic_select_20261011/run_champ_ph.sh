#!/bin/bash
# CD6 + POS10 相位验证 + 组合测试
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
SEL="--select-trail 120 --select-metric mean --select-topn 5"

run_one() {
  TAG=$1; OFF=$2; shift 2
  echo "=== $TAG o$OFF build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag ${TAG}_o${OFF} --offset $OFF "$@"
  echo "=== $TAG o$OFF sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag ${TAG}_o${OFF} --runner v42 --holding-bars 60
}

CD6="--hold-pctl 0.65 --cooldown 6 --max-hold-bars 60 --n-pos 15 --rank-by score"
P10="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 10 --rank-by score"

for OFF in 1 2 3 4; do run_one V5_P413_CH_CD6 $OFF $CD6 $SEL; done
for OFF in 1 2 3 4; do run_one V5_P413_CH_POS10 $OFF $P10 $SEL; done
run_one V5_P413_CH_CD6POS10 0 --hold-pctl 0.65 --cooldown 6 --max-hold-bars 60 --n-pos 10 --rank-by score $SEL

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_CH_CD6', 'V5_P413_CH_POS10'):
    for off in (1, 2, 3, 4):
        ensure_journal(t, f'{t}_o{off}', off)
ensure_journal('V5_P413_CH_CD6POS10', 'V5_P413_CH_CD6POS10_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "CHAMP_PH_DONE"
