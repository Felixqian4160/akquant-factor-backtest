// AKQuant v2 WebUI — Main JS (tab switching + health check)

// Tab switching.
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    document.querySelectorAll('.tab-pane').forEach(p => {
      p.classList.toggle('active', p.id === `tab-${target}`);
    });
    // Trigger lazy load.
    if (target === 'backtest' && !window._backtestLoaded) {
      loadBacktestResults();
      window._backtestLoaded = true;
    } else if (target === 'paper' && !window._paperLoaded) {
      loadPaperSessions();
      window._paperLoaded = true;
    }
  });
});

// Health check.
async function checkHealth() {
  const dot = document.getElementById('healthDot');
  const text = document.getElementById('healthText');
  try {
    const r = await fetch('/health');
    if (r.ok) {
      dot.className = 'dot ok';
      text.textContent = '服务正常';
    } else {
      dot.className = 'dot err';
      text.textContent = `HTTP ${r.status}`;
    }
  } catch (e) {
    dot.className = 'dot err';
    text.textContent = '无法连接';
  }
}
checkHealth();
setInterval(checkHealth, 30000);

// Helper: log box appender.
window.appendLog = function(boxId, msg) {
  const box = document.getElementById(boxId);
  if (!box) return;
  const ts = new Date().toLocaleTimeString();
  box.textContent += `[${ts}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
};

// Helper: format percent.
window.fmtPct = function(v, digits = 2) {
  if (v === null || v === undefined || isNaN(v)) return 'N/A';
  return `${v >= 0 ? '+' : ''}${v.toFixed(digits)}%`;
};

// Helper: sign class for positive/negative numbers.
window.signClass = function(v) {
  if (v === null || v === undefined || isNaN(v)) return 'neutral';
  return v > 0 ? 'pos' : (v < 0 ? 'neg' : 'neutral');
};

// Initial load (backtest tab is default-active).
window.addEventListener('DOMContentLoaded', () => {
  if (document.querySelector('.tab-btn.active')?.dataset.tab === 'backtest') {
    loadBacktestResults();
    window._backtestLoaded = true;
  }
});

// Also try after a tick (in case DOMContentLoaded already fired before this listener).
setTimeout(() => {
  if (!window._backtestLoaded && document.querySelector('.tab-btn.active')?.dataset.tab === 'backtest') {
    if (typeof loadBacktestResults === 'function') {
      loadBacktestResults();
      window._backtestLoaded = true;
    }
  }
}, 50);