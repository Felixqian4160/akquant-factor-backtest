// Tab 4: Paper Trading — start session + poll + render.

document.getElementById('paperStartBtn').addEventListener('click', async () => {
  const params = {
    start_date: document.getElementById('paperStart').value,
    end_date: document.getElementById('paperEnd').value,
    initial_cash: parseFloat(document.getElementById('paperCash').value),
    top_k: parseInt(document.getElementById('paperTopK').value),
    rebal_days: parseInt(document.getElementById('paperRebal').value),
  };
  const logBox = document.getElementById('paperLog');
  logBox.textContent = '';
  document.getElementById('paperResult').innerHTML = 'Running...';
  document.getElementById('paperStartBtn').disabled = true;
  try {
    appendLog('paperLog', `POST /api/paper/start params=${JSON.stringify(params)}`);
    const r = await fetch('/api/paper/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const d = await r.json();
    const sessionId = d.session_id;
    appendLog('paperLog', `started session_id=${sessionId}`);
    while (true) {
      await new Promise(r => setTimeout(r, 3000));
      const sr = await fetch(`/api/paper/status/${sessionId}`);
      const sd = await sr.json();
      const logTail = (sd.log_tail || []).join('\n');
      const prev = logBox.textContent.split('\n');
      const lastNew = logTail.split('\n').slice(prev.length - 1).join('\n');
      if (lastNew.trim()) appendLog('paperLog', lastNew);
      if (sd.finished) break;
      if (sd.error) {
        appendLog('paperLog', '✗ job errored');
        break;
      }
    }
    const rr = await fetch(`/api/paper/result/${sessionId}`);
    const result = await rr.json();
    renderPaperResult(result);
    appendLog('paperLog', `✓ done: closed_only_return=${result.metrics.closed_only_return_pct}%`);
    loadPaperSessions();
  } catch (e) {
    appendLog('paperLog', `ERROR: ${e.message}`);
    document.getElementById('paperResult').innerHTML = `<p style="color: var(--danger)">${e.message}</p>`;
  } finally {
    document.getElementById('paperStartBtn').disabled = false;
  }
});

function renderPaperResult(result) {
  const m = result.metrics;
  const c = document.getElementById('paperResult');
  let html = `
    <div class="metrics-grid">
      <div class="metric-card ${m.closed_only_return_pct >= 0 ? 'pos' : 'neg'}">
        <div class="label">Closed-Only Return</div>
        <div class="value">${fmtPct(m.closed_only_return_pct)}</div>
        <div class="sublabel">after costs</div>
      </div>
      <div class="metric-card ${m.annualized_pct >= 12 ? 'pos' : 'neg'}">
        <div class="label">Annualized</div>
        <div class="value">${fmtPct(m.annualized_pct, 1)}</div>
        <div class="sublabel">target ≥ 12%</div>
      </div>
      <div class="metric-card ${m.sharpe_ratio >= 1.2 ? 'pos' : 'neg'}">
        <div class="label">Sharpe Ratio</div>
        <div class="value">${m.sharpe_ratio.toFixed(3)}</div>
        <div class="sublabel">target ≥ 1.2</div>
      </div>
      <div class="metric-card ${m.max_drawdown_pct <= 12 ? 'pos' : 'neg'}">
        <div class="label">Max Drawdown</div>
        <div class="value">${m.max_drawdown_pct.toFixed(2)}%</div>
        <div class="sublabel">target ≤ 12%</div>
      </div>
      <div class="metric-card">
        <div class="label">Trades</div>
        <div class="value">${m.closed_trade_count}</div>
      </div>
      <div class="metric-card">
        <div class="label">Win Rate</div>
        <div class="value">${m.win_rate.toFixed(1)}%</div>
      </div>
      <div class="metric-card">
        <div class="label">Initial → Final</div>
        <div class="value">¥${(m.initial_cash / 1e6).toFixed(2)}M → ¥${(m.end_equity / 1e6).toFixed(2)}M</div>
      </div>
      <div class="metric-card">
        <div class="label">Duration</div>
        <div class="value">${result.elapsed_sec.toFixed(1)}s</div>
        <div class="sublabel">${result.n_rows} bars × ${result.n_stocks} stocks</div>
      </div>
    </div>
    <h3 style="margin-top: 24px;">Strategy Contract</h3>
    <table><tr><td>Factors</td><td style="font-family: var(--mono); font-size: 12px;">${result.contract.factors.join(', ')}</td></tr>
      <tr><td>Top K</td><td>${result.contract.top_k}</td></tr>
      <tr><td>Rebal Days</td><td>${result.contract.rebal_days}</td></tr>
      <tr><td>Kill DD Threshold</td><td>${(result.contract.kill_dd_threshold * 100).toFixed(1)}%</td></tr>
      <tr><td>Kill Cooldown</td><td>${result.contract.kill_cooldown_days} days</td></tr>
      <tr><td>Commission</td><td>${result.contract.commission_bps_per_side} bps/side</td></tr>
    </table>
  `;
  c.innerHTML = html;
}

async function loadPaperSessions() {
  try {
    const r = await fetch('/api/paper/sessions');
    const d = await r.json();
    const tbody = document.querySelector('#paperSessionsTable tbody');
    tbody.innerHTML = '';
    d.sessions.forEach(s => {
      const tr = document.createElement('tr');
      const created = new Date(s.created * 1000).toLocaleString();
      tr.innerHTML = `
        <td style="font-family: var(--mono); font-size: 11px;">${s.session_id}</td>
        <td>${s.finished ? '<span style="color: var(--success);">✓ finished</span>' : '<span style="color: var(--warning);">⏳ running</span>'}</td>
        <td style="font-size: 12px; color: var(--text2);">${created}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) { console.error(e); }
}