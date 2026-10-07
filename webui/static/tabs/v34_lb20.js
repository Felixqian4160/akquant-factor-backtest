// Tab 6: v34-lb20 Factor-Return Voting — production-locked strategy.
// 6 actions: verify / picks / 1-sim / run-all / report / plot.

const $ = (id) => document.getElementById(id);
let _v34CurrentJobId = null;
let _v34PollInterval = null;

async function _v34RefreshSystemStatus() {
  try {
    const r = await fetch('/api/v34lb20/status');
    const s = await r.json();
    const lines = [
      `lock: ${s.lock_exists ? '🔒 ' + s.lock_locked_at : '❌'}`,
      `strategy doc: ${s.strategy_doc_exists ? '✅' : '❌'}`,
      `panel: ${s.panel_exists ? '✅ ' + s.panel_size_gb + ' GB' : '❌'}`,
      `matrix_v34_ADJ: ${s.matrix_exists ? '✅ ' + s.matrix_size_mb + ' MB' : '❌'}`,
    ];
    if ($('v34SystemStatus')) {
      $('v34SystemStatus').textContent = lines.join(' | ');
    }
  } catch (e) {
    if ($('v34SystemStatus')) $('v34SystemStatus').textContent = 'status fetch failed';
  }
}

function _v34AppendLog(text) {
  const box = $('v34Log');
  if (!box) return;
  box.textContent += text + '\n';
  box.scrollTop = box.scrollHeight;
}

function _v34ClearLog() {
  if ($('v34Log')) $('v34Log').textContent = '';
}

async function _v34PollStatus(jobId) {
  try {
    const r = await fetch(`/api/v34lb20/status/${jobId}`);
    const s = await r.json();
    if (s.log_tail) {
      const cur = $('v34Log') ? $('v34Log').textContent : '';
      const tailStr = s.log_tail.join('\n');
      if (cur !== tailStr) {
        $('v34Log').textContent = tailStr + '\n';
        $('v34Log').scrollTop = $('v34Log').scrollHeight;
      }
    }
    if (s.finished) {
      if (_v34PollInterval) { clearInterval(_v34PollInterval); _v34PollInterval = null; }
      _v34AppendLog(`\n✅ finished job=${jobId}`);
      _v34RefreshSystemStatus();
    }
    if (s.error) {
      if (_v34PollInterval) { clearInterval(_v34PollInterval); _v34PollInterval = null; }
      _v34AppendLog(`\n❌ error in job=${jobId}`);
    }
  } catch (e) {
    // keep polling
  }
}

async function _v34Run(action, payload) {
  _v34ClearLog();
  _v34AppendLog(`>>> ${action} starting...`);
  try {
    const r = await fetch(`/api/v34lb20/${action}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {}),
    });
    const out = await r.json();
    _v34CurrentJobId = out.job_id;
    if (out.cmd) _v34AppendLog(`cmd: ${out.cmd.join(' ')}`);
    if (_v34PollInterval) clearInterval(_v34PollInterval);
    _v34PollInterval = setInterval(() => _v34PollStatus(out.job_id), 2000);
  } catch (e) {
    _v34AppendLog(`ERROR: ${e.message}`);
  }
}

window._v34LoadJob = function(jobId) {
  _v34ClearLog();
  _v34CurrentJobId = jobId;
  _v34PollStatus(jobId);
  if (_v34PollInterval) clearInterval(_v34PollInterval);
  _v34PollInterval = setInterval(() => _v34PollStatus(jobId), 2000);
};

async function _v34LoadJobs() {
  try {
    const r = await fetch('/api/v34lb20/jobs');
    const out = await r.json();
    const tbody = $('v34JobsTable');
    if (!tbody) return;
    tbody.innerHTML = '';
    out.jobs.forEach(j => {
      const tr = document.createElement('tr');
      const td1 = document.createElement('td');
      td1.textContent = j.job_id;
      const td2 = document.createElement('td');
      td2.textContent = j.finished ? '✅' : '⏳';
      const td3 = document.createElement('td');
      td3.textContent = new Date(j.created * 1000).toISOString().replace('T', ' ').slice(0, 19);
      const td4 = document.createElement('td');
      const btn = document.createElement('button');
      btn.textContent = 'load';
      btn.className = 'small';
      btn.onclick = () => window._v34LoadJob(j.job_id);
      td4.appendChild(btn);
      tr.appendChild(td1); tr.appendChild(td2); tr.appendChild(td3); tr.appendChild(td4);
      tbody.appendChild(tr);
    });
  } catch (e) {
    // silent
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if ($('v34VerifyBtn')) $('v34VerifyBtn').onclick = () => _v34Run('verify', {});
  if ($('v34PicksBtn')) $('v34PicksBtn').onclick = () => _v34Run('derive-picks', {});
  if ($('v34SimBtn')) $('v34SimBtn').onclick = () => _v34Run('run-sim', {offset: 0, window: 'oos'});
  if ($('v34RunAllBtn')) $('v34RunAllBtn').onclick = () => {
    if (confirm('Full batch takes ~25 minutes. Continue?')) _v34Run('run-all', {});
  };
  if ($('v34ReportBtn')) $('v34ReportBtn').onclick = () => _v34Run('report', {});
  if ($('v34PlotBtn')) $('v34PlotBtn').onclick = () => _v34Run('plot', {});
  if ($('v34RefreshJobsBtn')) $('v34RefreshJobsBtn').onclick = _v34LoadJobs;

  // Initial render
  _v34RefreshSystemStatus();
  _v34LoadJobs();
  setInterval(_v34RefreshSystemStatus, 30000);  // refresh status every 30s
});
