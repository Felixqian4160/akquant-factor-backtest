#!/bin/bash
# v5 pool413 (全 413 因子池) 动态选择评测: 硬选择 top5 + 软加权 wsum
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
CK=evidence/v3_resonance_20261010/_cache/votes__v5_pool413__any2_10_20_40_60_120.npz
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"
SEL="--select-trail 120 --select-metric hit --select-topn 5"

echo "[preflight] check cache completeness"
/usr/bin/python3.12 -c "
import numpy as np
z = np.load('$CK')
nb = int(z['next_batch'])
print('next_batch =', nb)
raise SystemExit(0 if nb >= 21 else 1)" || { echo "CACHE_INCOMPLETE"; exit 1; }

for V in "V5_P413_C120HIT:$SEL" "V5_P413_WSUM_HIT:$SEL --select-mode wsum"; do
  TAG="${V%%:*}"; EXTRA="${V#*:}"
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag ${TAG}_o0 --offset 0 $BASE $EXTRA
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag ${TAG}_o0 --runner v42 --holding-bars 60
done

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_C120HIT', 'V5_P413_WSUM_HIT'):
    ensure_journal(t, t + '_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "P413_SCRIPT_DONE"
