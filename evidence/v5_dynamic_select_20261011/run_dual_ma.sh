#!/bin/bash
# 双回看共振 (A) + 双均线过滤 (B) 测试 — 用户假设 2026-10-12
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

# --- A: 因子选择双回看共振 (20d + 120d) ---
run_one V5_P413_RET_T20   --select-trail 20  --select-metric mean --select-topn 5
run_one V5_P413_RET_DUAL  --select-trail 120 --select-mode dual  --select-metric mean --select-topn 5
run_one V5_P413_RET_DUALX --select-trail 120 --select-mode dualx --select-metric mean --select-topn 5

# --- B: 双均线过滤 (基于冠军 picks, 无 build) ---
/usr/bin/python3.12 -u examples/filter_picks_ma.py --src V5_P413_RET_RANK_o0 --mode dw --fstyle entry
/usr/bin/python3.12 -u examples/filter_picks_ma.py --src V5_P413_RET_RANK_o0 --mode dw --fstyle full
/usr/bin/python3.12 -u examples/filter_picks_ma.py --src V5_P413_RET_RANK_o0 --mode d  --fstyle entry
/usr/bin/python3.12 -u examples/filter_picks_ma.py --src V5_P413_RET_RANK_o0 --mode w  --fstyle entry
for TAG in V5_P413_RET_RANK_o0_MA_DW_ENTRY V5_P413_RET_RANK_o0_MA_DW_FULL V5_P413_RET_RANK_o0_MA_D_ENTRY V5_P413_RET_RANK_o0_MA_W_ENTRY; do
  echo "=== $TAG sim ($(date +%H:%M:%S)) ==="
  /usr/bin/python3.12 -u $FR --run 1 --tag $TAG --runner v42 --holding-bars 60
done

echo "=== journal ($(date +%H:%M:%S)) ==="
/usr/bin/python3.12 - <<PY
import sys
sys.path.insert(0, 'examples')
from v5_dynamic_select import ensure_journal, load_journal
for t in ('V5_P413_RET_T20', 'V5_P413_RET_DUAL', 'V5_P413_RET_DUALX'):
    ensure_journal(t, t + '_o0', 0)
for t in ('V5_P413_RET_RANK_o0_MA_DW_ENTRY', 'V5_P413_RET_RANK_o0_MA_DW_FULL',
          'V5_P413_RET_RANK_o0_MA_D_ENTRY', 'V5_P413_RET_RANK_o0_MA_W_ENTRY'):
    ensure_journal(t, t, 0)
print('journal total entries:', len(load_journal()))
PY
echo "DUAL_MA_DONE"
