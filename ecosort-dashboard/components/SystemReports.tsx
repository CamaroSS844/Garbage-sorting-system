import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts';
import {
  FileText, Download, TrendingUp, Archive, ShieldAlert, Activity,
  Trash2, AlertTriangle, RefreshCw, CheckCircle, Clock
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

// ─── CSS ─────────────────────────────────────────────────────────────────────

const injectStyles = () => {
  if (document.getElementById('ecosort-report-styles')) return;
  const style = document.createElement('style');
  style.id = 'ecosort-report-styles';
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap');

    .eco-root {
      --green: #00c471;
      --green-dim: #00c47122;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --blue: #3b82f6;
      --surface: #ffffff;
      --surface2: #f7f9fb;
      --border: #e8ecf0;
      --text: #0f172a;
      --muted: #64748b;
      --faint: #94a3b8;
      font-family: 'DM Sans', sans-serif;
      color: var(--text);
      background: var(--surface2);
      min-height: 100vh;
      padding: 36px 32px;
      box-sizing: border-box;
    }

    .eco-root * { box-sizing: border-box; }

    .eco-heading { font-family: 'Syne', sans-serif; }
    .eco-mono { font-family: 'DM Mono', monospace; }

    /* metric cards */
    .eco-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 22px 24px;
      position: relative;
      overflow: hidden;
      transition: box-shadow .2s;
    }
    .eco-card:hover { box-shadow: 0 8px 28px -8px rgba(0,0,0,.09); }
    .eco-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      border-radius: 18px 18px 0 0;
    }
    .eco-card.green::before { background: var(--green); }
    .eco-card.blue::before  { background: var(--blue); }
    .eco-card.rose::before  { background: var(--rose); }
    .eco-card.amber::before { background: var(--amber); }

    .eco-pill {
      display: inline-flex; align-items: center; gap: 5px;
      padding: 3px 10px; border-radius: 99px; font-size: 11px; font-weight: 600;
    }
    .eco-pill.green { background: var(--green-dim); color: #008048; }
    .eco-pill.amber { background: #fef3c7; color: #92400e; }

    /* chart panels */
    .eco-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 24px 26px;
    }

    /* download buttons */
    .eco-btn {
      display: inline-flex; align-items: center; gap: 7px;
      padding: 8px 16px; border-radius: 10px; font-size: 13px; font-weight: 600;
      cursor: pointer; border: none; transition: all .15s; white-space: nowrap;
    }
    .eco-btn-outline {
      background: var(--surface); border: 1px solid var(--border);
      color: var(--muted);
    }
    .eco-btn-outline:hover { background: var(--surface2); border-color: #c8d0dc; }
    .eco-btn-ghost-red {
      background: #fff0f3; border: 1px solid #fecdd3;
      color: var(--rose);
    }
    .eco-btn-ghost-red:hover { background: #ffe4ea; }
    .eco-btn-csv {
      background: #f0fdf4; border: 1px solid #bbf7d0;
      color: #166534;
    }
    .eco-btn-csv:hover { background: #dcfce7; }
    .eco-btn-pdf {
      background: var(--text); color: #fff;
      box-shadow: 0 4px 12px -3px rgba(15,23,42,.25);
    }
    .eco-btn-pdf:hover { background: #1e293b; }
    .eco-btn:disabled { opacity: .55; cursor: not-allowed; }

    /* toast */
    .eco-toast {
      position: fixed; bottom: 28px; right: 28px; z-index: 999;
      background: var(--text); color: #fff;
      padding: 11px 18px; border-radius: 12px;
      font-size: 13px; font-weight: 500;
      display: flex; align-items: center; gap: 8px;
      box-shadow: 0 8px 28px -6px rgba(0,0,0,.4);
      animation: toastIn .2s ease-out;
    }
    @keyframes toastIn {
      from { opacity:0; transform:translateY(10px); }
      to   { opacity:1; transform:translateY(0); }
    }

    /* modal */
    .eco-overlay {
      position: fixed; inset: 0; background: rgba(0,0,0,.38);
      display: flex; align-items: center; justify-content: center; z-index: 50;
    }
    .eco-modal {
      background: var(--surface); border-radius: 18px;
      padding: 28px 30px; width: 460px; max-width: 92vw;
    }

    /* table */
    .eco-table { width:100%; border-collapse:collapse; font-size:13px; }
    .eco-table th {
      text-align:left; padding:8px 12px; font-size:10px; font-weight:700;
      letter-spacing:.07em; text-transform:uppercase; color:var(--faint);
      border-bottom: 1px solid var(--border);
    }
    .eco-table td {
      padding:10px 12px; border-bottom:1px solid var(--border);
      color:var(--muted);
    }
    .eco-table tr:last-child td { border-bottom:none; }
    .eco-table tr:hover td { background: var(--surface2); }
  `;
  document.head.appendChild(style);
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

const fmt = (n) => n?.toLocaleString() ?? '0';
const fmtTime = (ts) =>
  new Date(ts * 1000).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });

// ── CSV export ────────────────────────────────────────────────────────────────
function downloadCSV(stats, history, failedItems) {
  const lines = [];

  lines.push('EcoSort System Report');
  lines.push(`Generated,${new Date().toISOString()}`);
  lines.push('');

  lines.push('=== SUMMARY ===');
  lines.push('Metric,Value');
  lines.push(`Total Items Processed,${stats.total_items_processed}`);
  lines.push(`Failed Inferences,${stats.failed_inferences}`);
  lines.push(`Accuracy Rate (%),${stats.accuracy_rate}`);
  lines.push(`Online Devices,${stats.online_devices}`);
  lines.push('');

  if (history.length) {
    lines.push('=== STATS HISTORY ===');
    lines.push('Timestamp,Total Processed,Total Failures,Online Devices,Accuracy Rate (%)');
    history.forEach(h => {
      lines.push(
        `${fmtTime(h.timestamp)},${h.total_processed},${h.total_failures},${h.online_devices},${h.accuracy_rate.toFixed(2)}`
      );
    });
    lines.push('');
  }

  if (failedItems.length) {
    lines.push('=== FAILED INFERENCES ===');
    lines.push('ID,Timestamp,Device ID,Confidence,Original Guess,Assigned Category,Reviewed,Notes');
    failedItems.forEach(f => {
      const row = [
        f.id,
        fmtTime(f.timestamp),
        f.device_id || '',
        f.confidence?.toFixed(3) || '',
        f.original_guess || '',
        f.assigned_category || '',
        f.reviewed ? 'Yes' : 'No',
        `"${(f.notes || '').replace(/"/g, '""')}"`
      ];
      lines.push(row.join(','));
    });
  }

  const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ecosort-report-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── PDF export (pure JS, no library needed) ───────────────────────────────────
async function downloadPDF(stats, history, failedItems) {
  // Build a printable HTML page and trigger browser print-to-PDF
  const dateStr = new Date().toLocaleString();
  const successCount = stats.total_items_processed - stats.failed_inferences;

  const histRows = history.map(h => `
    <tr>
      <td>${fmtTime(h.timestamp)}</td>
      <td>${h.total_processed.toLocaleString()}</td>
      <td>${h.total_failures.toLocaleString()}</td>
      <td>${h.accuracy_rate.toFixed(1)}%</td>
      <td>${h.online_devices}</td>
    </tr>`).join('');

  const failRows = failedItems.slice(0, 50).map(f => `
    <tr>
      <td>${fmtTime(f.timestamp)}</td>
      <td>${f.device_id || '—'}</td>
      <td>${f.confidence?.toFixed(3) || '—'}</td>
      <td>${f.original_guess || '—'}</td>
      <td>${f.assigned_category || 'Unreviewed'}</td>
      <td style="color:${f.reviewed ? '#008048' : '#b91c1c'}">${f.reviewed ? 'Yes' : 'No'}</td>
    </tr>`).join('');

  const html = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EcoSort Report — ${dateStr}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'DM Sans', sans-serif; color: #0f172a; background: #fff; font-size: 13px; }
  .page { max-width: 860px; margin: 0 auto; padding: 48px 40px; }

  .header { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:40px; padding-bottom:20px; border-bottom:2px solid #0f172a; }
  .logo { font-family:'Syne',sans-serif; font-size:26px; font-weight:800; letter-spacing:-.5px; }
  .logo span { color:#00c471; }
  .meta { text-align:right; font-size:11px; color:#64748b; line-height:1.7; font-family:'DM Mono',monospace; }

  .kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:36px; }
  .kpi { border:1px solid #e2e8f0; border-radius:12px; padding:16px 18px; }
  .kpi-label { font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:#94a3b8; margin-bottom:6px; }
  .kpi-val { font-family:'Syne',sans-serif; font-size:22px; font-weight:800; color:#0f172a; }
  .kpi-sub { font-size:11px; color:#94a3b8; margin-top:4px; }
  .kpi.highlight { border-color:#00c471; background:#f0fdf4; }
  .kpi.highlight .kpi-val { color:#008048; }

  h2 { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; margin-bottom:14px; padding-bottom:8px; border-bottom:1px solid #e2e8f0; }
  section { margin-bottom:36px; }

  table { width:100%; border-collapse:collapse; font-size:12px; }
  th { text-align:left; padding:7px 10px; font-size:10px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; color:#94a3b8; border-bottom:1px solid #e2e8f0; }
  td { padding:9px 10px; border-bottom:1px solid #f1f5f9; color:#475569; }
  tr:last-child td { border-bottom:none; }
  tr:nth-child(even) td { background:#f8fafc; }

  .accuracy-bar { background:#e2e8f0; border-radius:99px; height:10px; margin-top:8px; overflow:hidden; }
  .accuracy-fill { background: linear-gradient(90deg,#00c471,#10b981); height:100%; border-radius:99px; }

  .footer { margin-top:48px; padding-top:16px; border-top:1px solid #e2e8f0; font-size:11px; color:#94a3b8; font-family:'DM Mono',monospace; display:flex; justify-content:space-between; }
  @media print {
    body { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .no-print { display:none; }
  }
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <div class="logo">Eco<span>Sort</span></div>
      <div style="font-size:12px;color:#64748b;margin-top:4px;">Waste Intelligence Platform — Operational Report</div>
    </div>
    <div class="meta">
      Generated: ${dateStr}<br>
      Period: All-time accumulation<br>
      Report ID: RPT-${Date.now().toString(36).toUpperCase()}
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi highlight">
      <div class="kpi-label">Items processed</div>
      <div class="kpi-val">${fmt(stats.total_items_processed)}</div>
      <div class="kpi-sub">Cumulative total</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Accuracy rate</div>
      <div class="kpi-val">${stats.accuracy_rate.toFixed(1)}%</div>
      <div class="accuracy-bar"><div class="accuracy-fill" style="width:${stats.accuracy_rate}%"></div></div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Failed inferences</div>
      <div class="kpi-val">${fmt(stats.failed_inferences)}</div>
      <div class="kpi-sub">Needs human review</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Active nodes</div>
      <div class="kpi-val">${stats.online_devices}</div>
      <div class="kpi-sub">ESP32 controllers</div>
    </div>
  </div>

  <section>
    <h2>Performance summary</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th><th>Value</th><th>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Total items processed</td><td>${fmt(stats.total_items_processed)}</td><td>Since last reset</td></tr>
        <tr><td>Successfully categorized</td><td>${fmt(successCount)}</td><td>${stats.accuracy_rate.toFixed(1)}% of total</td></tr>
        <tr><td>Failed / low-confidence</td><td>${fmt(stats.failed_inferences)}</td><td>${stats.total_items_processed > 0 ? ((stats.failed_inferences / stats.total_items_processed) * 100).toFixed(1) : 0}% failure rate</td></tr>
        <tr><td>Online ESP32 devices</td><td>${stats.online_devices}</td><td>Active in last 10 s</td></tr>
      </tbody>
    </table>
  </section>

  ${history.length ? `
  <section>
    <h2>Stats history (last ${history.length} snapshots)</h2>
    <table>
      <thead>
        <tr><th>Timestamp</th><th>Total Processed</th><th>Failures</th><th>Accuracy</th><th>Online Devices</th></tr>
      </thead>
      <tbody>${histRows}</tbody>
    </table>
    <p style="font-size:11px;color:#94a3b8;margin-top:8px;">Snapshots flushed every 30 s while pipeline is active.</p>
  </section>` : ''}

  ${failedItems.length ? `
  <section>
    <h2>Failed inferences (${failedItems.length > 50 ? 'first 50 of ' + failedItems.length : failedItems.length})</h2>
    <table>
      <thead>
        <tr><th>Timestamp</th><th>Device</th><th>Confidence</th><th>AI Guess</th><th>Assigned Category</th><th>Reviewed</th></tr>
      </thead>
      <tbody>${failRows}</tbody>
    </table>
  </section>` : ''}

  <div class="footer">
    <span>EcoSort Waste Intelligence Platform</span>
    <span>CONFIDENTIAL — INTERNAL USE ONLY</span>
    <span>Page 1 of 1</span>
  </div>
</div>
<script>
  window.onload = () => { window.print(); };
</script>
</body>
</html>`;

  const win = window.open('', '_blank');
  if (!win) { alert('Pop-ups blocked. Please allow pop-ups for PDF export.'); return; }
  win.document.write(html);
  win.document.close();
}

// ─── Clear dialog ─────────────────────────────────────────────────────────────

const CLEAR_LEVELS = [
  { value: 'stats_only',  label: 'Stats history',        description: 'Clears system_stats snapshots only. Live counters are unaffected.', danger: false },
  { value: 'failed_only', label: 'Failed inferences',    description: 'Removes all failed inference records including demo rows.',          danger: false },
  { value: 'logs_only',   label: 'Detection logs',       description: 'Clears all zone-fire event logs.',                                  danger: false },
  { value: 'all',         label: 'Everything (nuclear)', description: 'Wipes all tables and resets live counters. Cannot be undone.',       danger: true  },
];

const ClearDialog = ({ onClose, onSuccess }) => {
  const [level, setLevel] = useState('stats_only');
  const [step, setStep] = useState('select');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const selected = CLEAR_LEVELS.find(l => l.value === level);

  const handleClear = async () => {
    setLoading(true); setError('');
    try {
      const res = await fetch(`http://localhost:8000/database/clear?level=${level}&confirm=CONFIRM`, { method: 'DELETE' });
      if (!res.ok) { const b = await res.json(); throw new Error(b.detail || 'Clear failed'); }
      onSuccess(); onClose();
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  const btnBase = { flex: 1, padding: '9px 0', borderRadius: 10, fontSize: 13, fontWeight: 600, cursor: 'pointer', border: 'none' };

  return (
    <div className="eco-overlay" onClick={onClose}>
      <div className="eco-modal" onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
          <Trash2 size={18} style={{ color: 'var(--rose)' }} />
          <h3 className="eco-heading" style={{ margin: 0, fontSize: 16, fontWeight: 700 }}>Clear database</h3>
        </div>

        {step === 'select' ? (<>
          <p style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 16 }}>Choose what to remove. This cannot be undone.</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 22 }}>
            {CLEAR_LEVELS.map(opt => (
              <label key={opt.value} style={{
                display: 'flex', alignItems: 'flex-start', gap: 12, padding: '10px 14px',
                borderRadius: 10, cursor: 'pointer',
                border: `1.5px solid ${level === opt.value ? (opt.danger ? '#fda4af' : '#93c5fd') : 'var(--border)'}`,
                background: level === opt.value ? (opt.danger ? '#fff0f3' : '#eff6ff') : 'transparent',
              }}>
                <input type="radio" name="level" value={opt.value} checked={level === opt.value} onChange={() => setLevel(opt.value)} style={{ marginTop: 3 }} />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600 }}>{opt.label}</div>
                  <div style={{ fontSize: 11.5, color: 'var(--muted)', marginTop: 2 }}>{opt.description}</div>
                </div>
              </label>
            ))}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={onClose} style={{ ...btnBase, background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>Cancel</button>
            <button onClick={() => setStep('confirm')} style={{ ...btnBase, background: selected.danger ? '#fff0f3' : '#0f172a', color: selected.danger ? 'var(--rose)' : '#fff', border: selected.danger ? '1px solid #fda4af' : 'none' }}>Continue</button>
          </div>
        </>) : (<>
          <div style={{ display: 'flex', gap: 10, padding: '12px 14px', borderRadius: 10, background: '#fef3c7', marginBottom: 18 }}>
            <AlertTriangle size={17} style={{ color: '#b45309', flexShrink: 0, marginTop: 1 }} />
            <p style={{ margin: 0, fontSize: 13, color: '#92400e', lineHeight: 1.5 }}>
              Clearing <strong>{selected.label}</strong>. {selected.description}
            </p>
          </div>
          {error && <p style={{ fontSize: 12, color: 'var(--rose)', marginBottom: 10 }}>{error}</p>}
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={() => setStep('select')} style={{ ...btnBase, background: 'var(--surface2)', border: '1px solid var(--border)', color: 'var(--muted)' }}>Back</button>
            <button onClick={handleClear} disabled={loading} style={{ ...btnBase, background: 'var(--rose)', color: '#fff', opacity: loading ? .6 : 1, cursor: loading ? 'not-allowed' : 'pointer' }}>
              {loading ? 'Clearing…' : 'Confirm clear'}
            </button>
          </div>
        </>)}
      </div>
    </div>
  );
};

// ─── Toast ────────────────────────────────────────────────────────────────────

const Toast = ({ msg, icon }) => (
  <div className="eco-toast">
    {icon} {msg}
  </div>
);

// ─── Main ─────────────────────────────────────────────────────────────────────

const SystemReports = () => {
  useEffect(() => { injectStyles(); }, []);

  const [stats, setStats] = useState({ total_items_processed: 0, failed_inferences: 0, online_devices: 0, accuracy_rate: 0 });
  const [history, setHistory] = useState([]);
  const [failedItems, setFailedItems] = useState([]);
  const [showClear, setShowClear] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState(new Date());
  const [toast, setToast] = useState(null);
  const [csvLoading, setCsvLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);

  const showToast = (msg, icon) => {
    setToast({ msg, icon });
    setTimeout(() => setToast(null), 3000);
  };

  const fetchReports = useCallback(async () => {
    try {
      const [sRes, hRes, fRes] = await Promise.all([
        fetch('http://localhost:8000/reports'),
        fetch('http://localhost:8000/reports/history?limit=20'),
        fetch('http://localhost:8000/failed-inferences?limit=100'),
      ]);
      if (sRes.ok) setStats(await sRes.json());
      if (hRes.ok) { const d = await hRes.json(); setHistory(d.history || []); }
      if (fRes.ok) { const d = await fRes.json(); setFailedItems(d.items || []); }
      setLastRefreshed(new Date());
    } catch (err) { console.error('Reports fetch failed:', err); }
  }, []);

  useEffect(() => {
    fetchReports();
    const id = setInterval(fetchReports, 30_000);
    return () => clearInterval(id);
  }, [fetchReports]);

  const handleCSV = async () => {
    setCsvLoading(true);
    try {
      downloadCSV(stats, history, failedItems);
      showToast('CSV downloaded', <CheckCircle size={14} />);
    } catch (e) { showToast('CSV export failed', <AlertTriangle size={14} />); }
    finally { setCsvLoading(false); }
  };

  const handlePDF = async () => {
    setPdfLoading(true);
    try {
      await downloadPDF(stats, history, failedItems);
      showToast('PDF opened — use browser print dialog to save', <CheckCircle size={14} />);
    } catch (e) { showToast('PDF export failed', <AlertTriangle size={14} />); }
    finally { setTimeout(() => setPdfLoading(false), 1200); }
  };

  // Derived
  const successCount = stats.total_items_processed - stats.failed_inferences;
  const total = stats.total_items_processed;

  const barData = [
    { name: 'Processed', count: total },
    { name: 'Failures',  count: stats.failed_inferences },
  ];
  const pieData = [
    { name: 'Success', value: successCount,            color: '#00c471' },
    { name: 'Failed',  value: stats.failed_inferences, color: '#f59e0b' },
  ];
  const trendData = history.map(h => ({
    time: new Date(h.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    processed: h.total_processed,
    failures:  h.total_failures,
    accuracy:  parseFloat(h.accuracy_rate.toFixed(1)),
  }));

  const metrics = [
    { label: 'Items Processed', value: fmt(total),                         sub: 'Cumulative total',       icon: Archive,    accent: 'green' },
    { label: 'Accuracy Rate',   value: `${stats.accuracy_rate.toFixed(1)}%`, sub: 'Based on processed',   icon: TrendingUp,  accent: 'blue'  },
    { label: 'Failures',        value: fmt(stats.failed_inferences),        sub: 'Flagged for review',    icon: ShieldAlert, accent: 'rose'  },
    { label: 'Active Nodes',    value: stats.online_devices.toString(),     sub: 'ESP32 controllers',     icon: Activity,    accent: 'amber' },
  ];

  const accentColors = { green: '#00c471', blue: '#3b82f6', rose: '#f43f5e', amber: '#f59e0b' };
  const accentBg     = { green: '#f0fdf4', blue: '#eff6ff', rose: '#fff0f3', amber: '#fffbeb' };

  return (
    <div className="eco-root">
      {toast && <Toast msg={toast.msg} icon={toast.icon} />}
      {showClear && <ClearDialog onClose={() => setShowClear(false)} onSuccess={fetchReports} />}

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-end', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14, marginBottom: 32 }}>
        <div>
          <h1 className="eco-heading" style={{ fontSize: 36, fontWeight: 500, margin: 0, letterSpacing: '-.4px' }}>
            Operational Reports
          </h1>
          <p className="eco-mono" style={{ margin: '5px 0 0', fontSize: 11, color: 'var(--faint)' }}>
            <Clock size={11} style={{ verticalAlign: 'middle', marginRight: 4 }} />
            Refreshed {lastRefreshed.toLocaleTimeString()} · auto every 30s
          </p>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
          <button className="eco-btn eco-btn-outline" onClick={fetchReports}>
            <RefreshCw size={14} /> Refresh
          </button>
          <button className="eco-btn eco-btn-ghost-red" onClick={() => setShowClear(true)}>
            <Trash2 size={14} /> Clear data
          </button>

          {/* CSV */}
          <button className="eco-btn eco-btn-csv" onClick={handleCSV} disabled={csvLoading}>
            <FileText size={14} />
            {csvLoading ? 'Preparing…' : 'Export CSV'}
          </button>

          {/* PDF */}
          <button className="eco-btn eco-btn-pdf" onClick={handlePDF} disabled={pdfLoading}>
            <Download size={14} />
            {pdfLoading ? 'Opening…' : 'Download PDF'}
          </button>
        </div>
      </div>

      {/* ── Metric cards ── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 14, marginBottom: 28 }}>
        {metrics.map((m, i) => (
          <div key={i} className={`eco-card ${m.accent}`}>
            <div style={{
              width: 36, height: 46, borderRadius: 10,
              background: accentBg[m.accent], display: 'flex', alignItems: 'center', justifyContent: 'center',
              marginBottom: 14,
            }}>
              <m.icon size={18} style={{ color: accentColors[m.accent] }} />
            </div>
            <div className="eco-mono" style={{ fontSize: 10, fontWeight: 500, color: 'var(--faint)', letterSpacing: '.06em', textTransform: 'uppercase', marginBottom: 4 }}>
              {m.label}
            </div>
            <div className="eco-heading" style={{ fontSize: 46, fontWeight: 300, lineHeight: 1 }}>{m.value}</div>
            <div style={{ fontSize: 11, color: 'var(--faint)', marginTop: 3 }}>{m.sub}</div>
          </div>
        ))}
      </div>

      {/* ── Charts row 1 ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginBottom: 18 }}>

        {/* Bar */}
        <div className="eco-panel">
          <div className="eco-heading" style={{ fontSize: 14, fontWeight: 700, marginBottom: 20 }}>Throughput Volume</div>
          <div style={{ height: 240 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8', fontFamily: 'DM Sans' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontFamily: 'DM Mono' }} />
                <Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #e2e8f0', fontFamily: 'DM Sans' }} cursor={{ fill: '#f8fafc' }} />
                <Bar dataKey="count" radius={[6,6,0,0]} barSize={52}>
                  <Cell fill="#00c471" /><Cell fill="#f59e0b" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Pie */}
        <div className="eco-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="eco-heading" style={{ fontSize: 14, fontWeight: 700, marginBottom: 30 }}>Categorization Quality</div>
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 32 }}>
            <div style={{ width: 160, height: 160 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={52} outerRadius={72} paddingAngle={6} dataKey="value">
                    {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                  </Pie>
                  <Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #e2e8f0', fontFamily: 'DM Sans' }} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {pieData.map((p, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: p.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, color: 'var(--muted)' }}>{p.name}</span>
                  <span className="eco-mono" style={{ fontSize: 13, fontWeight: 500, marginLeft: 'auto', minWidth: 40, textAlign: 'right' }}>
                    {total > 0 ? ((p.value / total) * 100).toFixed(1) : '0'}%
                  </span>
                </div>
              ))}
              <div style={{ paddingTop: 10, borderTop: '1px solid var(--border)' }}>
                <p style={{ fontSize: 11, color: 'var(--faint)', lineHeight: 1.5 }}>
                  {successCount.toLocaleString()} / {total.toLocaleString()} correctly sorted
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ── Trend line ── */}
      {trendData.length > 1 && (
        <div className="eco-panel" style={{ marginBottom: 18 }}>
          <div className="eco-heading" style={{ fontSize: 14, fontWeight: 700, marginBottom: 20 }}>Accuracy Trend Over Time</div>
          <div style={{ height: 220 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontFamily: 'DM Mono' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8', fontFamily: 'DM Mono' }} domain={[0,100]} unit="%" width={42} />
                <Tooltip contentStyle={{ borderRadius: 10, border: '1px solid #e2e8f0', fontFamily: 'DM Sans' }} formatter={v => [`${v}%`, 'Accuracy']} />
                <Legend wrapperStyle={{ fontSize: 12, fontFamily: 'DM Sans' }} />
                <Line type="monotone" dataKey="accuracy" name="Accuracy %" stroke="#00c471" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: '#00c471' }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Recent failed inferences table ── */}
      {failedItems.length > 0 && (
        <div className="eco-panel">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
            <div className="eco-heading" style={{ fontSize: 14, fontWeight: 700 }}>Recent Failed Inferences</div>
            <span className="eco-pill amber">
              <AlertTriangle size={10} /> {failedItems.filter(f => !f.reviewed).length} unreviewed
            </span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="eco-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Device</th>
                  <th>Conf.</th>
                  <th>AI Guess</th>
                  <th>Assigned</th>
                  <th>Reviewed</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {failedItems.slice(0, 10).map(f => (
                  <tr key={f.id}>
                    <td className="eco-mono" style={{ fontSize: 11 }}>{fmtTime(f.timestamp)}</td>
                    <td className="eco-mono" style={{ fontSize: 11 }}>{f.device_id || '—'}</td>
                    <td className="eco-mono" style={{ fontSize: 11 }}>{f.confidence?.toFixed(3) || '—'}</td>
                    <td>{f.original_guess || '—'}</td>
                    <td>{f.assigned_category || <span style={{ color: 'var(--faint)', fontStyle: 'italic' }}>Unreviewed</span>}</td>
                    <td>
                      {f.reviewed
                        ? <span className="eco-pill green"><CheckCircle size={9} /> Yes</span>
                        : <span className="eco-pill amber">No</span>}
                    </td>
                    <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.notes || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {failedItems.length > 10 && (
            <p style={{ fontSize: 11, color: 'var(--faint)', marginTop: 10 }}>
              Showing 10 of {failedItems.length} items. Export CSV or PDF for the full list.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SystemReports;