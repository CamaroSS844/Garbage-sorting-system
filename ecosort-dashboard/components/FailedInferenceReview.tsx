import React, { useState, useEffect, useCallback } from 'react';
import {
  Search, CheckCircle, RotateCcw, Image as ImageIcon, Info, RefreshCw, Loader
} from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────

type TrashCategory = 'Plastic' | 'Paper' | 'Metal' | 'Glass' | 'Organic' | 'Rejected';

interface FailedInference {
  id: number;
  timestamp: number;
  image_path: string | null;
  device_id: string | null;
  confidence: number | null;
  original_guess: string | null;
  assigned_category: string | null;
  reviewed: number;          // 0 = unreviewed, 1 = reviewed
  notes: string | null;
}

const CATEGORIES: TrashCategory[] = ['Plastic', 'Paper', 'Metal', 'Glass', 'Organic', 'Rejected'];

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function confidenceBadge(conf: number | null) {
  if (conf === null) return null;
  const pct = Math.round(conf * 100);
  const color = pct < 20 ? '#ef4444' : pct < 40 ? '#f59e0b' : '#10b981';
  return (
    <span style={{
      fontSize: 10, fontWeight: 700, padding: '2px 6px', borderRadius: 6,
      background: `${color}20`, color,
    }}>
      {pct}% conf.
    </span>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────

const FailedInferenceReview: React.FC = () => {
  const [items, setItems] = useState<FailedInference[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [filterReviewed, setFilterReviewed] = useState<'all' | 'unreviewed' | 'reviewed'>('all');
  const [actionLoading, setActionLoading] = useState<number | null>(null);

  // ── Fetch from DB ───────────────────────────────────────────────────────────

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const url = filterReviewed === 'all'
        ? 'http://localhost:8000/failed-inferences?limit=100'
        : `http://localhost:8000/failed-inferences?limit=100&reviewed=${filterReviewed === 'reviewed' ? 1 : 0}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setItems(data.items || []);
      }
    } catch (err) {
      console.error('Failed to fetch failed inferences:', err);
    } finally {
      setLoading(false);
    }
  }, [filterReviewed]);

  useEffect(() => {
    fetchItems();
  }, [fetchItems]);

  // Also listen for real-time failures via WebSocket and add them to the list
  useEffect(() => {
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket('ws://localhost:8000/ws/status');
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'inference_status' && msg.status === 'failure') {
          // Refresh the list when a new failure comes in
          fetchItems();
        }
      };
    } catch {
      // WebSocket unavailable — silent fallback, polling still works
    }
    return () => { if (ws) ws.close(); };
  }, [fetchItems]);

  // ── Actions ─────────────────────────────────────────────────────────────────

  const handleReview = async (id: number, category: TrashCategory) => {
    setActionLoading(id);
    try {
      const res = await fetch(
        `http://localhost:8000/failed-inferences/${id}/review?assigned_category=${encodeURIComponent(category)}`,
        { method: 'PATCH' }
      );
      if (res.ok) {
        setItems(prev => prev.map(item =>
          item.id === id
            ? { ...item, reviewed: 1, assigned_category: category }
            : item
        ));
      }
    } catch (err) {
      console.error('Review update failed:', err);
    } finally {
      setActionLoading(null);
    }
  };

  const handleRetry = async (id: number) => {
    setActionLoading(id);
    try {
      const res = await fetch(
        `http://localhost:8000/failed-inferences/${id}/retry`,
        { method: 'POST' }
      );
      if (res.ok) {
        setItems(prev => prev.map(item =>
          item.id === id
            ? { ...item, reviewed: 0, assigned_category: null }
            : item
        ));
      }
    } catch (err) {
      console.error('Retry reset failed:', err);
    } finally {
      setActionLoading(null);
    }
  };

  // ── Filtering ───────────────────────────────────────────────────────────────

  const filtered = items.filter(item => {
    const q = search.toLowerCase();
    if (!q) return true;
    return (
      item.device_id?.toLowerCase().includes(q) ||
      item.original_guess?.toLowerCase().includes(q) ||
      item.assigned_category?.toLowerCase().includes(q) ||
      item.notes?.toLowerCase().includes(q) ||
      String(item.id).includes(q)
    );
  });

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="max-w-6xl mx-auto space-y-6">

      {/* Toolbar */}
      <div className="bg-white p-4 rounded-2xl shadow-sm border flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest px-2">
            Failed inference queue
          </h2>
          <span className="text-xs font-bold bg-rose-50 text-rose-600 px-2 py-0.5 rounded-full">
            {items.filter(i => !i.reviewed).length} pending
          </span>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Filter tabs */}
          <div className="flex rounded-lg overflow-hidden border border-slate-200 text-xs font-semibold">
            {(['all', 'unreviewed', 'reviewed'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilterReviewed(f)}
                className={`px-3 py-1.5 transition-colors capitalize ${
                  filterReviewed === f
                    ? 'bg-slate-900 text-white'
                    : 'bg-white text-slate-500 hover:bg-slate-50'
                }`}
              >
                {f}
              </button>
            ))}
          </div>

          {/* Search */}
          <div className="relative flex-1 sm:flex-none">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={14} />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search…"
              className="pl-8 pr-4 py-1.5 text-sm bg-slate-50 border rounded-lg outline-none w-full focus:ring-2 focus:ring-emerald-500/20"
            />
          </div>

          {/* Refresh */}
          <button
            onClick={fetchItems}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Info banner */}
      <div className="bg-amber-50 p-4 rounded-xl border border-amber-100 flex items-start gap-3">
        <Info className="text-amber-500 shrink-0 mt-0.5" size={16} />
        <p className="text-sm text-amber-800">
          Records are persisted in SQLite. Three demo rows are pre-loaded for presentation.
          New failures are logged automatically when inference confidence falls below threshold.
        </p>
      </div>

      {/* Table */}
      <div className="bg-white rounded-2xl shadow-sm border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 border-b border-slate-100">
              <tr>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Item</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Detection data</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Categorization</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider text-right">Actions</th>
              </tr>
            </thead>

            <tbody className="divide-y divide-slate-100">
              {loading ? (
                <tr>
                  <td colSpan={4} className="py-16 text-center">
                    <Loader size={24} className="mx-auto text-slate-400 animate-spin mb-2" />
                    <p className="text-sm text-slate-400">Loading…</p>
                  </td>
                </tr>
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={4} className="py-20 text-center">
                    <CheckCircle size={48} className="mx-auto text-emerald-500 mb-4" strokeWidth={1} />
                    <h3 className="text-lg font-bold text-slate-800">Queue is clear</h3>
                    <p className="text-slate-500 text-sm">
                      {search ? 'No items match your search.' : 'No failed inferences to review.'}
                    </p>
                  </td>
                </tr>
              ) : (
                filtered.map(item => {
                  const isReviewed = item.reviewed === 1;
                  const isBusy = actionLoading === item.id;

                  return (
                    <tr
                      key={item.id}
                      className={`group hover:bg-slate-50/50 transition-colors ${isReviewed ? 'opacity-60' : ''}`}
                    >
                      {/* Item */}
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-4">
                          <div className="relative w-14 h-14 rounded-xl bg-slate-100 overflow-hidden border shrink-0 flex items-center justify-center">
                            {item.image_path ? (
                              <img
                                src={`http://localhost:8000/uploads/${item.image_path}`}
                                alt="Inference capture"
                                className="w-full h-full object-cover"
                                onError={e => (e.currentTarget.style.display = 'none')}
                              />
                            ) : (
                              <ImageIcon size={20} className="text-slate-300" />
                            )}
                          </div>
                          <div>
                            <p className="font-bold text-slate-800 text-sm">FAIL-{String(item.id).padStart(4, '0')}</p>
                            <p className="text-xs text-slate-400 mt-0.5">{formatTime(item.timestamp)}</p>
                            {item.notes && (
                              <p className="text-xs text-slate-400 mt-1 max-w-[180px] truncate" title={item.notes}>
                                {item.notes}
                              </p>
                            )}
                          </div>
                        </div>
                      </td>

                      {/* Detection data */}
                      <td className="px-6 py-4">
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-16">Device</span>
                            <span className="text-xs font-bold text-slate-600">{item.device_id ?? '—'}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-16">Model guess</span>
                            <span className="text-xs font-bold text-slate-600">{item.original_guess ?? 'None'}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-16">Confidence</span>
                            {confidenceBadge(item.confidence)}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs text-slate-400 w-16">Status</span>
                            <span className="text-[10px] font-bold text-rose-500 uppercase tracking-tight">
                              Below threshold
                            </span>
                          </div>
                        </div>
                      </td>

                      {/* Categorization */}
                      <td className="px-6 py-4">
                        {isReviewed ? (
                          <div className="flex items-center gap-2 text-emerald-600">
                            <CheckCircle size={15} />
                            <span className="text-sm font-bold">{item.assigned_category}</span>
                          </div>
                        ) : (
                          <select
                            defaultValue=""
                            disabled={isBusy}
                            onChange={e => handleReview(item.id, e.target.value as TrashCategory)}
                            className="bg-white border border-slate-200 text-sm rounded-lg px-3 py-1.5 outline-none focus:ring-2 focus:ring-emerald-500/20 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <option value="" disabled>Select category…</option>
                            {CATEGORIES.map(cat => (
                              <option key={cat} value={cat}>{cat}</option>
                            ))}
                          </select>
                        )}
                      </td>

                      {/* Actions */}
                      <td className="px-6 py-4 text-right">
                        <button
                          onClick={() => handleRetry(item.id)}
                          disabled={isBusy || !isReviewed}
                          title={isReviewed ? 'Reset to unreviewed' : 'Already unreviewed'}
                          className="p-2 text-slate-400 hover:text-slate-600 hover:bg-white rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                        >
                          {isBusy
                            ? <Loader size={16} className="animate-spin" />
                            : <RotateCcw size={16} />
                          }
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Footer count */}
        {!loading && filtered.length > 0 && (
          <div className="px-6 py-3 border-t border-slate-100 bg-slate-50 flex items-center justify-between">
            <p className="text-xs text-slate-400">
              Showing {filtered.length} of {items.length} records
            </p>
            <p className="text-xs text-slate-400">
              {items.filter(i => i.reviewed).length} reviewed · {items.filter(i => !i.reviewed).length} pending
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FailedInferenceReview;
