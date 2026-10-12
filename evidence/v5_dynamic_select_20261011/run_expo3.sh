#!/bin/bash
# 暴露层第三轮: DD 调参 + 纯比例缩仓参照点
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
RUN=examples/v43_run_akquant_v34.py
PB=evidence/v3_resonance_20261010/picks
OB=evidence/v3_resonance_20261010/sims

for T in V5EX_DD2_o0 V5EX_SCALE50_o0 V5EX_SCALE75_o0; do
  rm -rf $PB/$T
  cp -r $PB/V5_P413_RET_RANK_o0 $PB/$T
done

run_x() {
  TAG=$1; MODE=$2; shift 2
  echo "=== $TAG ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $RUN --tag $TAG --picks-base $PB --out-base $OB --exposure $MODE \
      --actions on --ca-mode all --prices raw --end 2025-12-31 --holding-bars 60 "$@" 2>&1 | tail -2
}

run_x V5EX_DD2_o0     dd    --dd-soft -0.10 --dd-hard -0.16 --dd-restore -0.02
run_x V5EX_SCALE50_o0 scale --scale-f 0.50
run_x V5EX_SCALE75_o0 scale --scale-f 0.75

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5EX_DD2_o0', 'V5EX_SCALE50_o0', 'V5EX_SCALE75_o0'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "EXPO3_DONE"
