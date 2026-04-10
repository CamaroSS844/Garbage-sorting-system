import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, Legend
} from 'recharts';
import {
  FileText, Download, TrendingUp, Archive, ShieldAlert, Activity,
  Trash2, AlertTriangle, ChevronDown, RefreshCw
} from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────

interface ReportStats {
  total_items_processed: number;
  failed_inferences: number;
  online_devices: number;
  accuracy_rate: number;
}

interface HistoryPoint {
  id: number;
  timestamp: number;
  total_processed: number;
  total_failures: number;
  online_devices: number;
  accuracy_rate: number;
}

type ClearLevel = 'stats_only' | 'failed_only' | 'logs_only' | 'all';

const CLEAR_LEVELS: { value: ClearLevel; label: string; description: string; danger: boolean }[] = [
  { value: 'stats_only',  label: 'Stats history',        description: 'Clears system_stats snapshots only. Live counters are unaffected.', danger: false },
  { value: 'failed_only', label: 'Failed inferences',    description: 'Removes all failed inference records including demo rows.',          danger: false },
  { value: 'logs_only',   label: 'Detection logs',       description: 'Clears all zone-fire event logs.',                                  danger: false },
  { value: 'all',         label: 'Everything (nuclear)', description: 'Wipes all tables and resets live counters. Cannot be undone.',       danger: true  },
];

// ─── Clear dialog ─────────────────────────────────────────────────────────────

interface ClearDialogProps {
  onClose: () => void;
  onSuccess: () => void;
}

const ClearDialog: React.FC<ClearDialogProps> = ({ onClose, onSuccess }) => {
  const [level, setLevel] = useState<ClearLevel>('stats_only');
  const [step, setStep] = useState<'select' | 'confirm'>('select');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const selected = CLEAR_LEVELS.find(l => l.value === level)!;

  const handleClear = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch(
        `http://localhost:8000/database/clear?level=${level}&confirm=CONFIRM`,
        { method: 'DELETE' }
      );
      if (!res.ok) {
        const body = await res.json();
        throw new Error(body.detail || 'Clear failed');
      }
      onSuccess();
      onClose();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50
      }}
      onClick={onClose}
    >
      <div
        style={{ background: 'var(--color-background-primary)', borderRadius: 16, padding: 28, width: 440, maxWidth: '90vw' }}
        onClick={e => e.stopPropagation()}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
          <Trash2 size={20} style={{ color: 'var(--color-text-danger)' }} />
          <h3 style={{ margin: 0, fontSize: 16, fontWeight: 500, color: 'var(--color-text-primary)' }}>
            Clear database
          </h3>
        </div>

        {step === 'select' ? (
          <>
            <p style={{ fontSize: 13, color: 'var(--color-text-secondary)', marginBottom: 16 }}>
              Choose what to remove. This cannot be undone.
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 24 }}>
              {CLEAR_LEVELS.map(opt => (
                <label
                  key={opt.value}
                  style={{
                    display: 'flex', alignItems: 'flex-start', gap: 12,
                    padding: '10px 14px', borderRadius: 10, cursor: 'pointer',
                    border: `1px solid ${level === opt.value
                      ? opt.danger ? 'var(--color-border-danger)' : 'var(--color-border-info)'
                      : 'var(--color-border-tertiary)'}`,
                    background: level === opt.value
                      ? opt.danger ? 'var(--color-background-danger)' : 'var(--color-background-info)'
                      : 'transparent',
                  }}
                >
                  <input
                    type="radio" name="level" value={opt.value}
                    checked={level === opt.value}
                    onChange={() => setLevel(opt.value)}
                    style={{ marginTop: 2 }}
                  />
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--color-text-primary)' }}>
                      {opt.label}
                    </div>
                    <div style={{ fontSize: 12, color: 'var(--color-text-secondary)', marginTop: 2 }}>
                      {opt.description}
                    </div>
                  </div>
                </label>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={onClose}
                style={{
                  flex: 1, padding: '9px 0', borderRadius: 10, border: '1px solid var(--color-border-tertiary)',
                  background: 'transparent', fontSize: 13, fontWeight: 500,
                  color: 'var(--color-text-secondary)', cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={() => setStep('confirm')}
                style={{
                  flex: 1, padding: '9px 0', borderRadius: 10, border: 'none',
                  background: selected.danger ? 'var(--color-background-danger)' : '#0f172a',
                  fontSize: 13, fontWeight: 500,
                  color: selected.danger ? 'var(--color-text-danger)' : '#fff',
                  cursor: 'pointer'
                }}
              >
                Continue
              </button>
            </div>
          </>
        ) : (
          <>
            <div style={{
              display: 'flex', gap: 10, padding: '12px 14px', borderRadius: 10,
              background: 'var(--color-background-warning)', marginBottom: 20
            }}>
              <AlertTriangle size={18} style={{ color: 'var(--color-text-warning)', flexShrink: 0, marginTop: 1 }} />
              <p style={{ margin: 0, fontSize: 13, color: 'var(--color-text-warning)', lineHeight: 1.5 }}>
                You are about to clear <strong>{selected.label}</strong>. {selected.description}
              </p>
            </div>

            {error && (
              <p style={{ fontSize: 12, color: 'var(--color-text-danger)', marginBottom: 12 }}>{error}</p>
            )}

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => setStep('select')}
                style={{
                  flex: 1, padding: '9px 0', borderRadius: 10, border: '1px solid var(--color-border-tertiary)',
                  background: 'transparent', fontSize: 13, fontWeight: 500,
                  color: 'var(--color-text-secondary)', cursor: 'pointer'
                }}
              >
                Back
              </button>
              <button
                onClick={handleClear}
                disabled={loading}
                style={{
                  flex: 1, padding: '9px 0', borderRadius: 10, border: 'none',
                  background: 'var(--color-text-danger)', fontSize: 13, fontWeight: 500,
                  color: '#fff', cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1
                }}
              >
                {loading ? 'Clearing…' : 'Confirm clear'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// ─── Main component ───────────────────────────────────────────────────────────

const SystemReports: React.FC = () => {
  const [stats, setStats] = useState<ReportStats>({
    total_items_processed: 0,
    failed_inferences: 0,
    online_devices: 0,
    accuracy_rate: 0,
  });
  const [history, setHistory] = useState<HistoryPoint[]>([]);
  const [showClear, setShowClear] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  const fetchReports = useCallback(async () => {
    try {
      const [statsRes, histRes] = await Promise.all([
        fetch('http://localhost:8000/reports'),
        fetch('http://localhost:8000/reports/history?limit=20'),
      ]);
      if (statsRes.ok) setStats(await statsRes.json());
      if (histRes.ok) {
        const data = await histRes.json();
        setHistory(data.history || []);
      }
      setLastRefreshed(new Date());
    } catch (err) {
      console.error('Reports fetch failed:', err);
    }
  }, []);

  useEffect(() => {
    fetchReports();
    const interval = setInterval(fetchReports, 30_000);
    return () => clearInterval(interval);
  }, [fetchReports]);

  // ── Derived values ──────────────────────────────────────────────────────────

  const successCount = stats.total_items_processed - stats.failed_inferences;
  const total = stats.total_items_processed;

  const barData = [
    { name: 'Processed', count: total },
    { name: 'Failures', count: stats.failed_inferences },
  ];

  const pieData = [
    { name: 'Success', value: successCount,            color: '#10b981' },
    { name: 'Failed',  value: stats.failed_inferences, color: '#f59e0b' },
  ];

  const trendData = history.map(h => ({
    time: new Date(h.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    processed: h.total_processed,
    failures: h.total_failures,
    accuracy: parseFloat(h.accuracy_rate.toFixed(1)),
  }));

  const metrics = [
    {
      label: 'Processed',
      value: total.toLocaleString(),
      sub: 'Cumulative total',
      icon: Archive,
      colorClass: 'emerald',
    },
    {
      label: 'Accuracy rate',
      value: `${stats.accuracy_rate.toFixed(1)}%`,
      sub: 'Based on processed',
      icon: TrendingUp,
      colorClass: 'blue',
    },
    {
      label: 'Failures',
      value: stats.failed_inferences.toLocaleString(),
      sub: 'Needs audit',
      icon: ShieldAlert,
      colorClass: 'rose',
    },
    {
      label: 'Nodes',
      value: stats.online_devices.toString(),
      sub: 'Active controllers',
      icon: Activity,
      colorClass: 'indigo',
    },
  ];

  const iconBg: Record<string, string> = {
    emerald: 'bg-emerald-50 text-emerald-600',
    blue:    'bg-blue-50 text-blue-600',
    rose:    'bg-rose-50 text-rose-600',
    indigo:  'bg-indigo-50 text-indigo-600',
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {showClear && (
        <ClearDialog
          onClose={() => setShowClear(false)}
          onSuccess={fetchReports}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Operational insights</h2>
          <p className="text-slate-500 text-sm">
            Last updated: {lastRefreshed.toLocaleTimeString()} · auto-refreshes every 30s
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={fetchReports}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-600 hover:bg-slate-50 transition-colors"
          >
            <RefreshCw size={16} />
            Refresh
          </button>
          <button
            onClick={() => setShowClear(true)}
            className="flex items-center gap-2 px-4 py-2 bg-white border border-rose-200 rounded-xl text-sm font-semibold text-rose-600 hover:bg-rose-50 transition-colors"
          >
            <Trash2 size={16} />
            Clear data
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-600 hover:bg-slate-50 transition-colors">
            <FileText size={18} />
            CSV export
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-xl text-sm font-semibold hover:bg-slate-800 transition-colors shadow-lg shadow-slate-200">
            <Download size={18} />
            PDF report
          </button>
        </div>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m, i) => (
          <div key={i} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 ${iconBg[m.colorClass]}`}>
              <m.icon size={20} />
            </div>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{m.label}</p>
            <h4 className="text-2xl font-black text-slate-800 mt-1">{m.value}</h4>
            <p className="text-[10px] font-bold mt-2 text-slate-400">{m.sub}</p>
          </div>
        ))}
      </div>

      {/* Charts row 1 — volume + quality */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Throughput volume */}
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 mb-6">Throughput volume</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <Tooltip
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={48}>
                  <Cell fill="#10b981" />
                  <Cell fill="#f59e0b" />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Categorization quality */}
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm flex flex-col">
          <h3 className="text-sm font-bold text-slate-800 mb-6">Categorization quality</h3>
          <div className="flex-1 flex flex-col sm:flex-row items-center justify-center gap-8">
            <div className="h-48 w-48 shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%" cy="50%"
                    innerRadius={60} outerRadius={80}
                    paddingAngle={8} dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-4">
              {pieData.map((p, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: p.color }} />
                  <span className="text-sm font-semibold text-slate-600">{p.name}</span>
                  <span className="text-sm font-bold text-slate-800 ml-auto">
                    {total > 0 ? ((p.value / total) * 100).toFixed(1) : '0'}%
                  </span>
                </div>
              ))}
              <div className="pt-2 border-t border-slate-100">
                <p className="text-xs text-slate-400">
                  {successCount.toLocaleString()} of {total.toLocaleString()} items correctly sorted
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Chart row 2 — trend over time (only shown when history exists) */}
      {trendData.length > 1 && (
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 mb-6">Accuracy trend over time</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: '#94a3b8' }} />
                <YAxis
                  axisLine={false} tickLine={false}
                  tick={{ fontSize: 11, fill: '#94a3b8' }}
                  domain={[0, 100]} unit="%" width={40}
                />
                <Tooltip
                  contentStyle={{ borderRadius: 12, border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                  formatter={(v: number) => [`${v}%`, 'Accuracy']}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line
                  type="monotone" dataKey="accuracy" name="Accuracy %"
                  stroke="#10b981" strokeWidth={2} dot={false} activeDot={{ r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="text-xs text-slate-400 mt-3">
            Snapshots are flushed to the database every 30 seconds while the pipeline is running.
          </p>
        </div>
      )}
    </div>
  );
};

export default SystemReports;
