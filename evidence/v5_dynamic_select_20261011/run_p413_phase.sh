#!/bin/bash
# 相位验证: V5_P413_RET_RANK (收益排名) + V5_P413_BORDA (双榜) × offsets 1-4
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"
SEL="--select-trail 120 --select-topn 5"

for OFF in 1 2 3 4; do
  TAG=V5_P413_RET_RANK_o${OFF}
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag $TAG --offset $OFF $BASE $SEL --select-matrix m --select-metric mean
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag $TAG --runner v42 --holding-bars 60
done
for OFF in 1 2 3 4; do
  TAG=V5_P413_BORDA_o${OFF}
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag $TAG --offset $OFF $BASE $SEL --select-mode borda --select-metric mean
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag $TAG --runner v42 --holding-bars 60
done

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_RET_RANK', 'V5_P413_BORDA'):
    for off in (1, 2, 3, 4):
        ensure_journal(t, f'{t}_o{off}', off)
print('journal total entries:', len(load_journal()))
PY
echo "P413_PHASE_DONE"
