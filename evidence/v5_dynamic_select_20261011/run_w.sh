#!/bin/bash
# H1 权重实验: equal (v44 对照) / invvol / hrp  (基于冠军 picks, exposure=none)
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
RUN=examples/v44_run_akquant_v34.py
PB=evidence/v3_resonance_20261010/picks
OB=evidence/v3_resonance_20261010/sims

for T in V5W_EQUAL_o0 V5W_INVVOL_o0 V5W_HRP_o0; do
  rm -rf $PB/$T
  cp -r $PB/V5_P413_RET_RANK_o0 $PB/$T
done

run_w() {
  TAG=$1; MODE=$2
  echo "=== $TAG ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $RUN --tag $TAG --picks-base $PB --out-base $OB --weight-mode $MODE \
      --weight-win 60 --actions on --ca-mode all --prices raw --end 2025-12-31 --holding-bars 60 2>&1 | tail -2
}

run_w V5W_EQUAL_o0  equal
run_w V5W_INVVOL_o0 invvol
run_w V5W_HRP_o0    hrp

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5W_EQUAL_o0', 'V5W_INVVOL_o0', 'V5W_HRP_o0'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "WEIGHT_DONE"
