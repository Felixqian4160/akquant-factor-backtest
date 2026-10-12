#!/bin/bash
# HRP 修复版验证: 无上限 + 权重上限 0.20
set -e
cd /media/felix/f/quant/akquant-factor-backtest
export PATH=/usr/bin:/bin HOME=/root LANG=C.UTF-8
export PYTHONPATH=/home/felix/.local/lib/python3.12/site-packages:/usr/lib/python3.12/site-packages
RUN=examples/v44_run_akquant_v34.py
PB=evidence/v3_resonance_20261010/picks
OB=evidence/v3_resonance_20261010/sims

for T in V5W_HRP_FIX_o0 V5W_HRP_CAP20_o0; do
  rm -rf $PB/$T
  cp -r $PB/V5_P413_RET_RANK_o0 $PB/$T
done

echo "=== V5W_HRP_FIX_o0 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $RUN --tag V5W_HRP_FIX_o0 --picks-base $PB --out-base $OB --weight-mode hrp \
    --weight-win 60 --actions on --ca-mode all --prices raw --end 2025-12-31 --holding-bars 60 2>&1 | tail -1
echo "=== V5W_HRP_CAP20_o0 ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 -u $RUN --tag V5W_HRP_CAP20_o0 --picks-base $PB --out-base $OB --weight-mode hrp \
    --weight-win 60 --w-cap 0.20 --actions on --ca-mode all --prices raw --end 2025-12-31 --holding-bars 60 2>&1 | tail -1

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5W_HRP_FIX_o0', 'V5W_HRP_CAP20_o0'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "HRPFIX_DONE"
