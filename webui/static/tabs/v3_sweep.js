// Tab 7: v3 共振参数扫描 — WebUI 程序端.
// 启动/继续自动扫描 · 实时进度与日志 · 收益/回撤比排名 · 净值曲线.

const $v3s = (id) => document.getElementById(id);
let _v3sChart = null;
let _v3sCurves = null;
let _v3sMode = 'seg';

function _v3sPct(v, d = 2) {
  if (v === null || v === undefined || isNaN(v)) return 'N/A';
  return (v * 100).toFixed(d) + '%';
}

async function _v3sRefreshOverview() {
  try {
    const r = await fetch('/api/v3sweep/overview');
    const s = await r.json();
    if ($v3s('v3sStage')) $v3s('v3sStage').textContent = s.stage + (s.complete ? ' ✅' : '');
    if ($v3s('v3sProgress')) $v3s('v3sProgress').textContent = `Wave-1: ${s.wave1_done}/${s.wave1_total} | Wave-2: ${s.wave2_done}`;
    if ($v3s('v3sRunning')) {
      $v3s('v3sRunning').textContent = s.running
        ? ('🟡 运行中 ' + (s.running_detail && s.running_detail.pid ? 'pid=' + s.running_detail.pid : '(external)'))
        : (s.complete ? '🟢 已完成' : '⚪ 空闲');
    }
    if ($v3s('v3sUpdated')) $v3s('v3sUpdated').textContent = s.updated || '—';
    if ($v3s('v3sLog') && s.log_tail) {
      $v3s('v3sLog').textContent = s.log_tail.join('\n');
      $v3s('v3sLog').scrollTop = $v3s('v3sLog').scrollHeight;
    }
  } catch (e) { /* silent */ }
}

async function _v3sLoadResults() {
  try {
    const r = await fetch('/api/v3sweep/results');
    const out = await r.json();
    const tbody = $v3s('v3sResultsTable');
    if (!tbody) return;
    tbody.innerHTML = '';
    (out.rows || []).forEach(row => {
      const tr = document.createElement('tr');
      if (row.rank === 1) tr.style.background = 'rgba(46,125,50,0.15)';
      if (row.is_champion) tr.style.outline = '1px solid #f9a825';
      const cells = [
        row.rank,
        `SW${String(row.idx).padStart(3, '0')}${row.is_champion ? ' ★' : ''}`,
        row.params.hold_pctl, row.params.cooldown, row.params.max_hold_bars, row.params.n_pos,
        row.n_phases,
        row.ratio.toFixed(3),
        _v3sPct(row.seg_ann),
        _v3sPct(row.seg_mdd),
        _v3sPct(row.seg_mdd_worst),
        _v3sPct(row.full_ann),
        row.trades,
      ];
      cells.forEach(c => {
        const td = document.createElement('td');
        td.textContent = c;
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    if ($v3s('v3sResultInfo')) {
      $v3s('v3sResultInfo').textContent =
        `${out.total} 组有结果（★ = 当前最优对照 SW${String(out.champion_idx).padStart(3, '0')}）`;
    }
  } catch (e) { /* silent */ }
}

async function _v3sLoadCurves() {
  try {
    const r = await fetch('/api/v3sweep/curves');
    const out = await r.json();
    if (!out.configs || !out.configs.length) {
      if ($v3s('v3sChartHint')) $v3s('v3sChartHint').textContent =
        '尚未生成（至少要有 1 个完成配置）→ 点「生成净值曲线」';
      return;
    }
    _v3sCurves = out;
    if ($v3s('v3sChartHint')) $v3s('v3sChartHint').textContent =
      `generated ${out.generated} · ${out.configs.length} configs + ${out.benchmarks.length} benchmarks`;
    _v3sDrawChart();
  } catch (e) { /* silent */ }
}

function _v3sDrawChart() {
  if (!_v3sCurves) return;
  const canvas = $v3s('v3sChart');
  if (!canvas || typeof Chart === 'undefined') return;
  const colors = ['#1e88e5', '#2e7d32', '#f57c00', '#8e24aa', '#e53935', '#00897b'];
  const mode = _v3sMode;
  const labels = _v3sCurves.configs[0][mode].dates;
  const datasets = _v3sCurves.configs.map((c, j) => ({
    label: `${c.label} (ratio ${c.ratio_mean.toFixed(3)}, ${c.n_phases}相位)`,
    data: c[mode].values,
    borderColor: colors[j % 6],
    borderWidth: j === 0 ? 2.2 : 1.4,
    pointRadius: 0,
    fill: false,
    tension: 0,
  }));
  _v3sCurves.benchmarks.forEach(b => {
    datasets.push({
      label: b.label,
      data: b[mode].values,
      borderColor: '#9e9e9e',
      borderWidth: 1.2,
      borderDash: [5, 4],
      pointRadius: 0,
      fill: false,
    });
  });
  if (_v3sChart) _v3sChart.destroy();
  _v3sChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      animation: false,
      responsive: false,
      scales: {
        x: { ticks: { maxTicksLimit: 10 } },
        y: { type: mode === 'full' ? 'logarithmic' : 'linear' },
      },
      plugins: { legend: { labels: { boxWidth: 12, font: { size: 10 } } } },
    },
  });
}

async function _v3sRunSweep() {
  if (!confirm('启动/继续全自动扫描？（自动跑完剩余组合 → top-6 五相位验证 → 汇总）')) return;
  try {
    const r = await fetch('/api/v3sweep/run', { method: 'POST' });
    if (r.status === 409) {
      const d = await r.json();
      alert(d.detail || '已在运行中');
      return;
    }
    const out = await r.json();
    alert(`已启动 pid=${out.pid}`);
    _v3sRefreshOverview();
  } catch (e) {
    alert('启动失败: ' + e.message);
  }
}

async function _v3sGenCurves() {
  if ($v3s('v3sChartHint')) $v3s('v3sChartHint').textContent = '生成中...（读取全部相位 NAV，约 20s）';
  try {
    const r = await fetch('/api/v3sweep/curves-gen', { method: 'POST' });
    const out = await r.json();
    if (!out.ok) alert('生成失败: ' + JSON.stringify(out.stderr_tail || out));
    await _v3sLoadCurves();
  } catch (e) {
    alert('生成失败: ' + e.message);
  }
}

async function _v3sGenReport() {
  try {
    const r = await fetch('/api/v3sweep/report-gen', { method: 'POST' });
    const out = await r.json();
    if ($v3s('v3sReportBox')) $v3s('v3sReportBox').textContent = out.report_md || '(empty)';
  } catch (e) {
    alert('报告生成失败: ' + e.message);
  }
}

function _v3sBoot() {
  if ($v3s('v3sRunBtn')) $v3s('v3sRunBtn').onclick = _v3sRunSweep;
  if ($v3s('v3sCurvesBtn')) $v3s('v3sCurvesBtn').onclick = _v3sGenCurves;
  if ($v3s('v3sReportBtn')) $v3s('v3sReportBtn').onclick = _v3sGenReport;
  if ($v3s('v3sRefreshBtn')) {
    $v3s('v3sRefreshBtn').onclick = () => {
      _v3sRefreshOverview();
      _v3sLoadResults();
      _v3sLoadCurves();
    };
  }
  if ($v3s('v3sSegBtn')) {
    $v3s('v3sSegBtn').onclick = () => { _v3sMode = 'seg'; _v3sDrawChart(); };
  }
  if ($v3s('v3sFullBtn')) {
    $v3s('v3sFullBtn').onclick = () => { _v3sMode = 'full'; _v3sDrawChart(); };
  }

  _v3sRefreshOverview();
  _v3sLoadResults();
  _v3sLoadCurves();
  setInterval(_v3sRefreshOverview, 4000);
  setInterval(_v3sLoadResults, 15000);
}

window._v3sLoadCurves = _v3sLoadCurves;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _v3sBoot);
} else {
  _v3sBoot();
}
