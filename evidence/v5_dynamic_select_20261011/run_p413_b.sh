#!/bin/bash
# v5 pool413 第二轮: 全库重测 (IC排名/收益排名/disp/MO) + 3 个新方法 (borda/dedup/cluster)
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
FR=examples/v3_resonance_framework.py
FS=evidence/v5_dynamic_select_20261011/run_factsets/v5_pool413.json
CK=evidence/v3_resonance_20261010/_cache/votes__v5_pool413__any2_10_20_40_60_120.npz
BASE="--hold-pctl 0.65 --cooldown 3 --max-hold-bars 60 --n-pos 15 --rank-by score"
SEL="--select-trail 120 --select-topn 5"

echo "[preflight] cache + IC matrix"
/usr/bin/python3.12 -c "
import numpy as np, pathlib, sys
z = np.load('$CK'); print('next_batch =', int(z['next_batch']))
ok = int(z['next_batch']) >= 21 and pathlib.Path('evidence/v34_dedup_20261010/matrix_ic_ADJ.parquet').exists()
sys.exit(0 if ok else 1)" || { echo "PREFLIGHT_FAIL"; exit 1; }

run_one() {
  TAG=$1; shift
  echo "=== $TAG build ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --factset $FS --build --tag ${TAG}_o0 --offset 0 $BASE "$@"
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag ${TAG}_o0 --runner v42 --holding-bars 60
}

# --- A: 前面的方法 (全库重测) ---
run_one V5_P413_IC_RANK   $SEL --select-matrix ic --select-metric mean
run_one V5_P413_RET_RANK  $SEL --select-matrix m  --select-metric mean
run_one V5_P413_DISP      $SEL --select-metric hit --decide disp
run_one V5_P413_MO120     $SEL --select-metric mo
# --- B: 3 个新方法 ---
run_one V5_P413_BORDA     $SEL --select-mode borda   --select-metric mean
run_one V5_P413_DEDUP     $SEL --select-mode dedup   --select-metric hit
run_one V5_P413_CLUSTER   $SEL --select-mode cluster --select-metric hit

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_IC_RANK', 'V5_P413_RET_RANK', 'V5_P413_DISP', 'V5_P413_MO120',
          'V5_P413_BORDA', 'V5_P413_DEDUP', 'V5_P413_CLUSTER'):
    ensure_journal(t, t + '_o0', 0)
print('journal total entries:', len(load_journal()))
PY
echo "P413_B_DONE"
