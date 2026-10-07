// Tab 5: Daily Factor Reselect — full pipeline with AKQuant.

const $ = (id) => document.getElementById(id);
let _dfrCurrentJobId = null;
let _dfrPollInterval = null;
let _dfrChart = null;

function _dfrAppendLog(text) {
  const box = $('dfrLog');
  if (!box) return;
  box.textContent += text + '\n';
  box.scrollTop = box.scrollHeight;
}

function _dfrDrawNavCurve(navData) {
  // navData: [{date, nav, n_factors, N, factors_top3}, ...]
  const canvas = $('dfrNavChart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!navData || navData.length === 0) {
    ctx.fillStyle = '#666';
    ctx.font = '14px sans-serif';
    ctx.fillText('No NAV data yet', w/2 - 60, h/2);
    return;
  }
  const padding = 40;
  const navs = navData.map(d => d.nav);
  const minNav = Math.min(...navs, 0.9);
  const maxNav = Math.max(...navs, 1.2);
  const yScale = (v) => h - padding - (v - minNav) / (maxNav - minNav) * (h - 2 * padding);
  const xScale = (i) => padding + (i / (navData.length - 1)) * (w - 2 * padding);

  // Reference line at 1.0
  ctx.strokeStyle = '#444';
  ctx.setLineDash([5, 5]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, yScale(1.0));
  ctx.lineTo(w - padding, yScale(1.0));
  ctx.stroke();
  ctx.setLineDash([]);

  // NAV curve
  ctx.strokeStyle = '#e74c3c';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  navData.forEach((d, i) => {
    const x = xScale(i);
    const y = yScale(d.nav);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Points
  ctx.fillStyle = '#e74c3c';
  navData.forEach((d, i) => {
    const x = xScale(i);
    const y = yScale(d.nav);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '11px sans-serif';
  ctx.fillText(`min: ${minNav.toFixed(4)}`, 5, h - padding + 5);
  ctx.fillText(`max: ${maxNav.toFixed(4)}`, 5, padding + 5);
  ctx.fillText(`rebal cycles: ${navData.length}`, w - 100, padding + 5);
  if (navData.length > 0) {
    const last = navData[navData.length - 1];
    ctx.fillText(`last: ${last.date} NAV=${last.nav.toFixed(4)}`, w - 250, padding + 20);
  }
}

async function _dfrLoadLatestProgress() {
  try {
    const r = await fetch('/api/daily_factor/latest_progress');
    if (!r.ok) return;
    const d = await r.json();
    const navData = (d.events || [])
      .filter(e => e.stage === 'rebal_done' && e.nav !== undefined)
      .map(e => ({
        date: e.rebal_date,
        nav: e.nav,
        n_factors: e.n_factors,
        N: e.N,
        factors_top3: [],
      }));
    if (navData.length > 0) {
      _dfrDrawNavCurve(navData);
    }
    // Update stage badge
    const latest = d.latest_event;
    if (latest) {
      const stageEl = $('dfrStage');
      if (stageEl) stageEl.textContent = latest.stage || '—';
      const jobEl = $('dfrJobId');
      if (jobEl) jobEl.textContent = latest.job_id || '—';
      if (latest.stage === 'metrics_done') {
        const pctEl = $('dfrPct');
        if (pctEl) pctEl.textContent = `${(latest.pct || 0).toFixed(1)}%`;
        const dateEl = $('dfrDate');
        if (dateEl) dateEl.textContent = latest.date || '';
        const countEl = $('dfrCycleCount');
        if (countEl) countEl.textContent = `${latest.n_dates_done}/${latest.n_dates_total}`;
      } else if (latest.stage === 'rebal_done') {
        const cycleEl = $('dfrCycleCount');
        if (cycleEl) cycleEl.textContent = `${latest.n_cycles_done}/${latest.n_cycles_total}`;
        const navEl = $('dfrNav');
        if (navEl) navEl.textContent = (latest.nav || 1.0).toFixed(4);
        const retEl = $('dfrRet');
        if (retEl) retEl.textContent = `${((latest.nav || 1.0) - 1) * 100 >= 0 ? '+' : ''}${((latest.nav || 1.0) - 1) * 100 | 0}%`;
      } else if (latest.stage === 'done') {
        const navEl = $('dfrNav');
        if (navEl) navEl.textContent = (latest.final_nav || 1.0).toFixed(4);
      }
    }
  } catch (e) {
    console.error('latest_progress failed', e);
  }
}

function _dfrStartPolling(jobId) {
  _dfrCurrentJobId = jobId;
  if (_dfrPollInterval) clearInterval(_dfrPollInterval);
  _dfrPollInterval = setInterval(_dfrLoadLatestProgress, 2000);
}

function _dfrStopPolling() {
  if (_dfrPollInterval) {
    clearInterval(_dfrPollInterval);
    _dfrPollInterval = null;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const runBtn = $('dfrRunBtn');
  if (!runBtn) return;

  // Initial: load latest progress (in case a job is already running)
  _dfrLoadLatestProgress();

  runBtn.addEventListener('click', async () => {
    const params = {
      start_date: $('dfrStartDate').value || '2010-01-01',
      end_date: $('dfrEndDate').value || '2025-12-31',
      rebal_days: parseInt($('dfrRebalDays').value) || 20,
      top_k: parseInt($('dfrTopK').value) || 20,
      window_days: parseInt($('dfrWindowDays').value) || 60,
    };
    $('dfrLog').textContent = '';
    $('dfrResult').innerHTML = 'Running...';
    runBtn.disabled = true;
    try {
      _dfrAppendLog(`POST /api/daily_factor/run ${JSON.stringify(params)}`);
      const r = await fetch('/api/daily_factor/run', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(params),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail || 'failed');
      const jobId = d.job_id;
      $('dfrJobId').textContent = jobId;
      _dfrAppendLog(`started job_id=${jobId}`);
      _dfrStartPolling(jobId);
      // Wait for finish
      const checkFinished = async () => {
        while (true) {
          await new Promise(r => setTimeout(r, 3000));
          const pr = await fetch(`/api/daily_factor/progress/${jobId}`);
          const pd = await pr.json();
          if (pd.finished) return pd;
        }
      };
      const finalProgress = await checkFinished();
      _dfrAppendLog(`finished: ${finalProgress.n_events} events`);
      const rr = await fetch(`/api/daily_factor/result/${jobId}`);
      const rd = await rr.json();
      _dfrAppendLog(`final_nav=${rd.final_nav?.toFixed(4)} return=${rd.final_return_pct?.toFixed(2)}% n_cycles=${rd.n_rebal_cycles}`);
      $('dfrResult').innerHTML = `
        <div style="padding:12px; border:1px solid #444; border-radius:4px;">
          <div><b>Final NAV:</b> ${rd.final_nav?.toFixed(4)} (${rd.final_return_pct?.toFixed(2)}%)</div>
          <div><b>Rebal cycles:</b> ${rd.n_rebal_cycles}</div>
          <div><b>Job ID:</b> ${jobId}</div>
        </div>
      `;
      // Draw final NAV curve from result
      const navData = (rd.nav_curve || []).map(([date, nav, n_factors, N, factors_top3]) => ({
        date, nav, n_factors, N, factors_top3: factors_top3 || []
      }));
      _dfrDrawNavCurve(navData);
    } catch (e) {
      _dfrAppendLog(`ERROR: ${e.message}`);
      $('dfrResult').innerHTML = `<div style="color:red;">ERROR: ${e.message}</div>`;
    } finally {
      runBtn.disabled = false;
      _dfrStopPolling();
    }
  });
});