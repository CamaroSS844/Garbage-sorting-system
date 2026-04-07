
import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { FileText, Download, TrendingUp, Archive, ShieldAlert, Activity } from 'lucide-react';

const SystemReports: React.FC = () => {
  const [stats, setStats] = useState({
    total_items_processed: 0,
    failed_inferences: 0,
    online_devices: 0
  });

  const fetchReports = async () => {
    try {
      const response = await fetch('http://localhost:8000/reports');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error("Reports fetch failed:", err);
    }
  };

  useEffect(() => {
    fetchReports();
    const interval = setInterval(fetchReports, 30000); // Poll every 30s
    return () => clearInterval(interval);
  }, []);

  const barData = [
    { name: 'Processed', count: stats.total_items_processed },
    { name: 'Failed', count: stats.failed_inferences },
  ];

  const pieData = [
    { name: 'Success', value: stats.total_items_processed - stats.failed_inferences, color: '#10b981' },
    { name: 'Failed', value: stats.failed_inferences, color: '#f59e0b' },
  ];

  const metrics = [
    { label: 'Processed', value: stats.total_items_processed.toString(), sub: 'Cumulative total', icon: Archive, color: 'emerald' },
    { label: 'Accuracy Rate', value: stats.total_items_processed > 0 ? `${(((stats.total_items_processed - stats.failed_inferences) / stats.total_items_processed) * 100).toFixed(1)}%` : '0%', sub: 'Based on processed', icon: TrendingUp, color: 'blue' },
    { label: 'Failures', value: stats.failed_inferences.toString(), sub: 'Needs audit', icon: ShieldAlert, color: 'rose' },
    { label: 'Nodes', value: stats.online_devices.toString(), sub: 'Active controllers', icon: Activity, color: 'indigo' },
  ];

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-slate-800">Operational Insights</h2>
          <p className="text-slate-500 text-sm">Real-time data from the sorting network.</p>
        </div>
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-600 hover:bg-slate-50 transition-colors">
            <FileText size={18} />
            CSV Export
          </button>
          <button className="flex items-center gap-2 px-4 py-2 bg-slate-900 text-white rounded-xl text-sm font-semibold hover:bg-slate-800 transition-colors shadow-lg shadow-slate-200">
            <Download size={18} />
            PDF Report
          </button>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m, i) => (
          <div key={i} className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-4 
              ${m.color === 'emerald' ? 'bg-emerald-50 text-emerald-600' : ''}
              ${m.color === 'blue' ? 'bg-blue-50 text-blue-600' : ''}
              ${m.color === 'rose' ? 'bg-rose-50 text-rose-600' : ''}
              ${m.color === 'indigo' ? 'bg-indigo-50 text-indigo-600' : ''}
            `}>
              <m.icon size={20} />
            </div>
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">{m.label}</p>
            <h4 className="text-2xl font-black text-slate-800 mt-1">{m.value}</h4>
            <p className="text-[10px] font-bold mt-2 text-slate-400">
              {m.sub}
            </p>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        
        {/* volume */}
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 mb-6">Throughput Volume</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#94a3b8' }} />
                <Tooltip 
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' }}
                />
                <Bar dataKey="count" fill="#10b981" radius={[6, 6, 0, 0]} barSize={40} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Breakdown */}
        <div className="bg-white p-6 rounded-3xl border border-slate-100 shadow-sm flex flex-col">
          <h3 className="text-sm font-bold text-slate-800 mb-6">Categorization Quality</h3>
          <div className="flex-1 flex flex-col sm:flex-row items-center justify-center gap-8">
            <div className="h-48 w-48">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={8}
                    dataKey="value"
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
                    {stats.total_items_processed > 0 ? ((p.value / stats.total_items_processed) * 100).toFixed(1) : '0'}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemReports;
