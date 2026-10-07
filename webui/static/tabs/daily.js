// Tab 3: Daily Strategy — pick top-K stocks for a given date.

document.getElementById('dailyRunBtn').addEventListener('click', async () => {
  const params = {
    pick_date: document.getElementById('dailyDate').value,
    top_k: parseInt(document.getElementById('dailyTopK').value),
  };
  if (!params.pick_date) {
    appendLog('dailyLog', 'ERROR: pick_date is required');
    return;
  }
  const logBox = document.getElementById('dailyLog');
  logBox.textContent = '';
  document.getElementById('dailyResult').innerHTML = 'Running...';
  document.getElementById('dailyRunBtn').disabled = true;
  try {
    appendLog('dailyLog', `POST /api/daily/run date=${params.pick_date} top_k=${params.top_k}`);
    const r = await fetch('/api/daily/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const d = await r.json();
    const jobId = d.job_id;
    appendLog('dailyLog', `started job_id=${jobId}`);
    while (true) {
      await new Promise(r => setTimeout(r, 1500));
      const sr = await fetch(`/api/daily/status/${jobId}`);
      const sd = await sr.json();
      const logLines = (sd.log_tail || []).join('\n');
      if (sd.log_tail && sd.log_tail.length > 0) {
        // Only append last unrendered lines.
        const prev = logBox.textContent.split('\n');
        const lastCmdLine = prev.find(l => l.includes('started job_id')) || '';
        const newLines = sd.log_tail.slice(prev.length - 1).join('\n');
        if (newLines.trim() && newLines !== lastCmdLine) {
          appendLog('dailyLog', newLines);
        }
      }
      if (sd.finished) break;
      if (sd.error) {
        appendLog('dailyLog', '✗ job errored');
        break;
      }
    }
    const rr = await fetch(`/api/daily/result/${jobId}`);
    const result = await rr.json();
    if (result.error) {
      document.getElementById('dailyResult').innerHTML = `<p style="color: var(--danger)">${result.error}</p>`;
      appendLog('dailyLog', `✗ ERROR: ${result.error}`);
      return;
    }
    renderDailyResult(result);
    appendLog('dailyLog', `✓ done: ${result.picks.length} picks (scored ${result.n_stocks_scored} stocks)`);
  } catch (e) {
    appendLog('dailyLog', `ERROR: ${e.message}`);
    document.getElementById('dailyResult').innerHTML = `<p style="color: var(--danger)">${e.message}</p>`;
  } finally {
    document.getElementById('dailyRunBtn').disabled = false;
  }
});

function renderDailyResult(result) {
  const c = document.getElementById('dailyResult');
  let html = `
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="label">Date Requested</div>
        <div class="value" style="font-size: 16px;">${result.date_requested}</div>
      </div>
      <div class="metric-card">
        <div class="label">Date Used (trading day)</div>
        <div class="value pos" style="font-size: 16px;">${result.date_used}</div>
      </div>
      <div class="metric-card">
        <div class="label">Stocks Scored</div>
        <div class="value">${result.n_stocks_scored}</div>
      </div>
      <div class="metric-card">
        <div class="label">Top K</div>
        <div class="value">${result.top_k}</div>
      </div>
    </div>
    <h3 style="margin-top: 24px;">6-Factor Composite Score Ranking</h3>
    <p style="color: var(--text2); font-size: 13px; margin-bottom: 16px;">
      Factors: ${result.factors.join(', ')}
    </p>
    <table class="picks-table"><thead><tr>
      <th class="rank-col">#</th>
      <th>股票代码</th>
      <th>Composite Score</th>
      <th>Close</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Weight (Equal)</th>
    </tr></thead><tbody>
  `;
  result.picks.forEach(p => {
    html += `<tr>
      <td class="rank-col">${p.rank}</td>
      <td class="ts-code">${p.ts_code}</td>
      <td><strong style="color: var(--primary);">${p.score.toFixed(4)}</strong></td>
      <td>¥${p.close.toFixed(2)}</td>
      <td>¥${p.open.toFixed(2)}</td>
      <td>¥${p.high.toFixed(2)}</td>
      <td>¥${p.low.toFixed(2)}</td>
      <td class="weight">${p.weight_pct.toFixed(1)}%</td>
    </tr>`;
  });
  html += '</tbody></table>';
  c.innerHTML = html;
}

// Auto-fill today minus some days as default pick_date.
(async () => {
  try {
    const r = await fetch('/api/daily/dates');
    const d = await r.json();
    document.getElementById('dailyDate').min = d.start;
    document.getElementById('dailyDate').max = d.end;
    // Set default to a recent valid trading day.
    document.getElementById('dailyDate').value = '2024-02-02';  // bull leg #31
  } catch (e) { /* ignore */ }
})();