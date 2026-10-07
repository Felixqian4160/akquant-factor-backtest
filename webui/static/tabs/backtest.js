// Tab 1: Backtest Results — load + render v1+v2 stages.

async function loadBacktestResults() {
  // Summary card.
  try {
    const r = await fetch('/api/backtest/summary');
    const s = await r.json();
    const c = document.getElementById('summaryContent');
    c.innerHTML = `
      <div class="metrics-grid">
        <div class="metric-card"><div class="label">Annualized ≥ 12%</div><div class="value pos">${s.achieved.annualized}</div></div>
        <div class="metric-card"><div class="label">Sharpe ≥ 1.2</div><div class="value pos">${s.achieved.sharpe}</div></div>
        <div class="metric-card"><div class="label">MDD ≤ 12%</div><div class="value neg">${s.achieved.mdd}</div></div>
        <div class="metric-card"><div class="label">Production-Realistic</div><div class="value neg">${s.production_realistic}</div></div>
      </div>
      <p style="margin-top: 16px; padding: 12px; background: var(--bg3); border-radius: 4px; color: var(--text2); font-size: 13px;">
        <strong>Terminal State:</strong> ${s.terminal_state}<br><br>
        <strong>Interpretation:</strong> ${s.interpretation}
      </p>
    `;
  } catch (e) {
    document.getElementById('summaryContent').innerHTML = `<p style="color: var(--danger)">Error: ${e.message}</p>`;
  }

  // Stages table.
  try {
    const r = await fetch('/api/backtest/stages');
    const d = await r.json();
    const tbody = document.querySelector('#stagesTable tbody');
    tbody.innerHTML = '';
    d.stages.forEach(s => {
      const agg = s.aggregate || {};
      const tr = document.createElement('tr');
      tr.style.cursor = 'pointer';
      tr.innerHTML = `
        <td>${s.stage_id}</td>
        <td style="font-size: 11px; color: var(--text2);">${s.file}</td>
        <td>${agg.n_legs ?? 'N/A'}</td>
        <td class="${agg.mean_closed_only_pct >= 0 ? 'pos' : 'neg'}">${fmtPct(agg.mean_closed_only_pct)}</td>
        <td class="${agg.mean_annualized_pct >= 0 ? 'pos' : 'neg'}">${fmtPct(agg.mean_annualized_pct, 1)}</td>
        <td>${agg.mean_sharpe?.toFixed(3) ?? 'N/A'}</td>
        <td class="${agg.mean_mdd_pct <= 12 ? 'pos' : 'neg'}">${agg.mean_mdd_pct?.toFixed(2) ?? 'N/A'}%</td>
        <td>${agg.n_meet_target ?? 'N/A'}/${agg.n_legs ?? '?'}</td>
      `;
      tr.addEventListener('click', () => loadStageDetail(s.path, s.stage_id, s.file));
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('Failed to load stages', e);
  }
}

async function loadStageDetail(path, stageId, file) {
  const detail = document.getElementById('backtestDetail');
  detail.style.display = 'block';
  document.getElementById('detailTitle').textContent = `Per-Leg Detail: ${stageId}/${file}`;
  const content = document.getElementById('detailContent');
  content.innerHTML = 'Loading...';
  try {
    // path format: "evidence/.../per_leg.json"
    const parts = path.split('/');
    const dirName = parts[parts.length - 2];
    const fileName = parts[parts.length - 1];
    const r = await fetch(`/api/backtest/stage/${encodeURIComponent(dirName)}/${encodeURIComponent(fileName)}`);
    const d = await r.json();
    if (!d.legs) {
      content.innerHTML = '<p>No legs data.</p>';
      return;
    }
    let html = `<table><thead><tr><th>Leg</th><th>Start</th><th>End</th><th>Days</th><th>Closed%</th><th>Ann%</th><th>Sharpe</th><th>MDD%</th><th>Trades</th></tr></thead><tbody>`;
    d.legs.forEach(leg => {
      if (!('closed_only_return_pct' in leg)) return;
      const ann = Math.pow(1 + leg.closed_only_return_pct / 100, 365 / leg.days) * 100 - 1;
      html += `<tr>
        <td>#${leg.leg_idx}</td>
        <td style="font-size: 11px;">${leg.start}</td>
        <td style="font-size: 11px;">${leg.end}</td>
        <td>${leg.days}</td>
        <td class="${leg.closed_only_return_pct >= 0 ? 'pos' : 'neg'}">${fmtPct(leg.closed_only_return_pct)}</td>
        <td class="${ann >= 12 ? 'pos' : 'neg'}">${fmtPct(ann, 1)}</td>
        <td class="${leg.sharpe_ratio >= 1.2 ? 'pos' : 'neg'}">${leg.sharpe_ratio.toFixed(3)}</td>
        <td class="${leg.max_drawdown_pct <= 12 ? 'pos' : 'neg'}">${leg.max_drawdown_pct.toFixed(2)}%</td>
        <td>${leg.closed_trade_count}</td>
      </tr>`;
    });
    html += '</tbody></table>';
    content.innerHTML = html;
  } catch (e) {
    content.innerHTML = `<p style="color: var(--danger)">Error: ${e.message}</p>`;
  }
}