// Tab 2: Walk-Forward — run + poll + render.

document.getElementById('wfRunBtn').addEventListener('click', async () => {
  const params = {
    train_start: document.getElementById('wfTrainStart').value,
    train_end: document.getElementById('wfTrainEnd').value,
    val_months: parseInt(document.getElementById('wfValMonths').value),
    oos_months: parseInt(document.getElementById('wfOosMonths').value),
    step_months: parseInt(document.getElementById('wfStepMonths').value),
    max_windows: parseInt(document.getElementById('wfMaxWindows').value),
  };
  const logBox = document.getElementById('wfLog');
  logBox.textContent = '';
  document.getElementById('wfResult').innerHTML = 'Running...';
  document.getElementById('wfRunBtn').disabled = true;
  try {
    appendLog('wfLog', `POST /api/walkforward/run params=${JSON.stringify(params)}`);
    const r = await fetch('/api/walkforward/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    const d = await r.json();
    const jobId = d.job_id;
    appendLog('wfLog', `started job_id=${jobId}`);
    // Poll status every 3s.
    let lastLines = 0;
    while (true) {
      await new Promise(r => setTimeout(r, 3000));
      const sr = await fetch(`/api/walkforward/status/${jobId}`);
      const sd = await sr.json();
      // Update log tail if new lines.
      const logLines = (sd.log_tail || []).join('\n');
      // Append new progress events.
      if (sd.progress && sd.progress.length > lastLines) {
        for (let i = lastLines; i < sd.progress.length; i++) {
          const p = sd.progress[i];
          if (p.event === 'window_start') {
            appendLog('wfLog', `▶ window ${p.idx}: ${p.oos_start} ~ ${p.oos_end}`);
          } else if (p.event === 'window_done') {
            appendLog('wfLog', `  ✓ ${fmtPct(p.metrics.closed_only_return_pct)} sharpe=${p.metrics.sharpe_ratio.toFixed(2)} MDD=${p.metrics.max_drawdown_pct.toFixed(1)}% ann=${p.metrics.annualized_pct.toFixed(1)}% (${p.metrics.elapsed_sec}s)`);
          } else if (p.event === 'window_error') {
            appendLog('wfLog', `  ✗ ERROR: ${p.error}`);
          } else if (p.event === 'finished') {
            appendLog('wfLog', `✓ finished: ${JSON.stringify(p.summary)}`);
          }
        }
        lastLines = sd.progress.length;
      }
      if (sd.finished) break;
      if (sd.error) {
        appendLog('wfLog', '✗ job errored, aborting poll');
        break;
      }
    }
    // Fetch final result.
    const rr = await fetch(`/api/walkforward/result/${jobId}`);
    const result = await rr.json();
    renderWfResult(result);
  } catch (e) {
    appendLog('wfLog', `ERROR: ${e.message}`);
    document.getElementById('wfResult').innerHTML = `<p style="color: var(--danger)">${e.message}</p>`;
  } finally {
    document.getElementById('wfRunBtn').disabled = false;
  }
});

function renderWfResult(result) {
  const summary = result.summary;
  const windows = result.windows;
  const container = document.getElementById('wfResult');
  // Top metrics.
  let html = `
    <div class="metrics-grid">
      <div class="metric-card ${summary.n_pass_target > 0 ? 'pos' : 'neg'}">
        <div class="label">Pass Target</div>
        <div class="value">${summary.n_pass_target}/${summary.n_windows}</div>
        <div class="sublabel">windows pass ann≥12% / Sharpe≥1.2 / MDD≤12%</div>
      </div>
      <div class="metric-card ${summary.mean_oos_ann_pct >= 12 ? 'pos' : 'neg'}">
        <div class="label">Mean OOS Ann%</div>
        <div class="value">${fmtPct(summary.mean_oos_ann_pct, 1)}</div>
      </div>
      <div class="metric-card ${summary.mean_oos_sharpe >= 1.2 ? 'pos' : 'neg'}">
        <div class="label">Mean OOS Sharpe</div>
        <div class="value">${summary.mean_oos_sharpe.toFixed(3)}</div>
      </div>
      <div class="metric-card ${summary.mean_oos_mdd_pct <= 12 ? 'pos' : 'neg'}">
        <div class="label">Mean OOS MDD%</div>
        <div class="value">${summary.mean_oos_mdd_pct.toFixed(2)}%</div>
      </div>
    </div>
    <h3 style="margin-top: 24px;">Per-Window OOS Results</h3>
    <table><thead><tr>
      <th>Window</th><th>OOS Start</th><th>OOS End</th>
      <th>Closed%</th><th>Ann%</th><th>Sharpe</th><th>MDD%</th><th>Trades</th><th>Pass</th>
    </tr></thead><tbody>
  `;
  windows.forEach(w => {
    if (!w.metrics) {
      html += `<tr><td>${w.idx}</td><td colspan="8" style="color: var(--danger)">ERROR: ${w.error || 'no metrics'}</td></tr>`;
      return;
    }
    const m = w.metrics;
    const pass = m.sharpe_ratio >= 1.2 && m.max_drawdown_pct <= 12 && m.annualized_pct >= 12;
    html += `<tr>
      <td>${w.idx}</td>
      <td style="font-size: 11px;">${w.oos_start}</td>
      <td style="font-size: 11px;">${w.oos_end}</td>
      <td class="${m.closed_only_return_pct >= 0 ? 'pos' : 'neg'}">${fmtPct(m.closed_only_return_pct)}</td>
      <td class="${m.annualized_pct >= 12 ? 'pos' : 'neg'}">${fmtPct(m.annualized_pct, 1)}</td>
      <td class="${m.sharpe_ratio >= 1.2 ? 'pos' : 'neg'}">${m.sharpe_ratio.toFixed(3)}</td>
      <td class="${m.max_drawdown_pct <= 12 ? 'pos' : 'neg'}">${m.max_drawdown_pct.toFixed(2)}%</td>
      <td>${m.closed_trade_count}</td>
      <td>${pass ? '✅' : '❌'}</td>
    </tr>`;
  });
  html += '</tbody></table>';
  container.innerHTML = html;
}