#!/bin/bash
# 暴露层 4 变体 (基于冠军 picks): idx / idxhard / dd / combo  (+none 已跑)
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
RUN=examples/v43_run_akquant_v34.py
PB=evidence/v3_resonance_20261010/picks
OB=evidence/v3_resonance_20261010/sims

run_x() {
  TAG=$1; MODE=$2; shift 2
  echo "=== $TAG ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $RUN --tag $TAG --picks-base $PB --out-base $OB --exposure $MODE \
      --actions on --ca-mode all --prices raw --end 2025-12-31 --holding-bars 60 "$@" 2>&1 | tail -2
}

run_x V5EX_IDX_o0    idx
run_x V5EX_IDXH_o0   idxhard
run_x V5EX_DD_o0     dd
run_x V5EX_COMBO_o0  combo

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5EX_NONE_o0', 'V5EX_IDX_o0', 'V5EX_IDXH_o0', 'V5EX_DD_o0', 'V5EX_COMBO_o0'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "EXPO_DONE"
