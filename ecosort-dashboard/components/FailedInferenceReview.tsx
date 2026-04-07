
import React, { useState, useEffect } from 'react';
import { FailedInference, TrashCategory } from '../types';
import { Search, Filter, CheckCircle, RotateCcw, Image as ImageIcon, Info } from 'lucide-react';

const FailedInferenceReview: React.FC = () => {
  // This list will be populated in real-time as WebSocket failures occur
  const [items, setItems] = useState<FailedInference[]>([]);

  useEffect(() => {
    // Listen for real-time inference failures
    let ws: WebSocket | null = null;
    try {
      ws = new WebSocket('ws://localhost:8000/ws/status');
      ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'inference_status' && message.status === 'failure') {
          // Add newly failed item to the queue (UI only until refreshed)
          const newItem: FailedInference = {
            id: `FAIL-${Date.now()}`,
            timestamp: new Date().toLocaleTimeString(),
            imageUrl: 'https://picsum.photos/seed/failed/200/200',
            confidence: 0,
            originalGuess: 'None',
            deviceId: message.device_id,
            reviewed: false
          };
          setItems(prev => [newItem, ...prev]);
        }
      };
    } catch (err) {
      console.error("WS for FailedReview failed:", err);
    }

    return () => {
      if (ws) ws.close();
    };
  }, []);

  const categories: TrashCategory[] = ['Plastic', 'Paper', 'Metal', 'Glass', 'Organic', 'Rejected'];

  const handleManualReview = (id: string, category: TrashCategory) => {
    setItems(prev => prev.map(item => 
      item.id === id ? { ...item, reviewed: true, originalGuess: category } : item
    ));
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="bg-white p-4 rounded-2xl shadow-sm border flex flex-col sm:flex-row items-center justify-between gap-4">
        <h2 className="text-sm font-bold text-slate-500 uppercase tracking-widest px-2">Real-time Failure Queue</h2>
        <div className="flex items-center gap-2 w-full sm:w-auto">
          <div className="relative flex-1 sm:flex-none">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
            <input type="text" placeholder="Search live session logs..." className="pl-9 pr-4 py-1.5 text-sm bg-slate-50 border rounded-lg outline-none w-full" />
          </div>
        </div>
      </div>

      <div className="bg-amber-50 p-4 rounded-xl border border-amber-100 flex items-start gap-3">
        <Info className="text-amber-500 shrink-0" size={20} />
        <p className="text-sm text-amber-800">
          This queue displays failed inferences captured during your current session via WebSockets. Historical reviews require additional backend storage integration.
        </p>
      </div>

      <div className="bg-white rounded-2xl shadow-sm border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left">
            <thead className="bg-slate-50 border-b border-slate-100">
              <tr>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Item</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Detection Data</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider">Categorization</th>
                <th className="px-6 py-4 text-xs font-bold text-slate-500 uppercase tracking-wider text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {items.map((item) => (
                <tr key={item.id} className={`group hover:bg-slate-50/50 transition-colors ${item.reviewed ? 'opacity-50' : ''}`}>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-4">
                      <div className="relative w-16 h-16 rounded-xl bg-slate-100 overflow-hidden border">
                        <img src={item.imageUrl} alt="Trash Item" className="w-full h-full object-cover" />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 flex items-center justify-center transition-colors">
                          <ImageIcon className="text-white opacity-0 group-hover:opacity-100" size={20} />
                        </div>
                      </div>
                      <div>
                        <p className="font-bold text-slate-800 text-sm">{item.id}</p>
                        <p className="text-xs text-slate-400 mt-1">{item.timestamp}</p>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-slate-400 w-16">Device:</span>
                        <span className="text-xs font-bold text-slate-600">{item.deviceId}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-slate-400 w-16">Status:</span>
                        <span className="text-[10px] font-bold text-red-500 uppercase tracking-tight">Failure Triggered</span>
                      </div>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    {item.reviewed ? (
                      <div className="flex items-center gap-2 text-emerald-600">
                        <CheckCircle size={16} />
                        <span className="text-sm font-bold">Labelled as {item.originalGuess}</span>
                      </div>
                    ) : (
                      <select 
                        onChange={(e) => handleManualReview(item.id, e.target.value as TrashCategory)}
                        className="bg-white border border-slate-200 text-sm rounded-lg px-3 py-1.5 outline-none focus:ring-2 focus:ring-emerald-500/20"
                        defaultValue=""
                      >
                        <option value="" disabled>Select category...</option>
                        {categories.map(cat => <option key={cat} value={cat}>{cat}</option>)}
                      </select>
                    )}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button className="p-2 text-slate-400 hover:text-slate-600 hover:bg-white rounded-lg transition-all">
                      <RotateCcw size={18} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {items.length === 0 && (
          <div className="py-20 text-center">
            <CheckCircle size={48} className="mx-auto text-emerald-500 mb-4" strokeWidth={1} />
            <h3 className="text-lg font-bold text-slate-800">Queue is Clear</h3>
            <p className="text-slate-500">Watching WebSocket for new failed detections...</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default FailedInferenceReview;
