#!/bin/bash
# LAG20 相位验证: offsets 1-4
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"

for OFF in 1 2 3 4; do
  echo "=== V5L_LAG20 o$OFF build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag V5L_LAG20_o${OFF} --offset $OFF $BASE \
      --select-trail 120 --select-metric mean --select-topn 5 --select-lag 20
  echo "=== V5L_LAG20 o$OFF sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag V5L_LAG20_o${OFF} --runner v42 --holding-bars 60
done

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for off in (1, 2, 3, 4):
    ensure_journal('V5L_LAG20', f'V5L_LAG20_o{off}', off)
print('journal total entries:', len(load_journal()))
PY
echo "LAGPH_DONE"
