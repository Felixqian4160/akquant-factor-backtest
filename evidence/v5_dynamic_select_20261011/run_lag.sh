#!/bin/bash
# H3: 因子动量跳过期 (清华论文构建) — lag20/lag40/lag40+t220
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

run_one V5L_LAG20_o0     --select-trail 120 --select-metric mean --select-topn 5 --select-lag 20
run_one V5L_LAG40_o0     --select-trail 120 --select-metric mean --select-topn 5 --select-lag 40
run_one V5L_LAG40_T220_o0 --select-trail 220 --select-metric mean --select-topn 5 --select-lag 40

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5L_LAG20_o0', 'V5L_LAG40_o0', 'V5L_LAG40_T220_o0'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "LAG_DONE"
