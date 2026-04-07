
import React, { useState, useEffect } from 'react';
import { Activity, FastForward, CheckCircle2, Target, Gauge } from 'lucide-react';

const SystemTelemetryPanel: React.FC = () => {
  const [systemState, setSystemState] = useState({ mode: 'RUN', fault: null });

  useEffect(() => {
    const fetchState = async () => {
      try {
        const res = await fetch('http://localhost:8000/system/state');
        if (res.ok) {
          const data = await res.json();
          setSystemState(data);
        }
      } catch (err) {}
    };
    fetchState();
    const interval = setInterval(fetchState, 5000);
    return () => clearInterval(interval);
  }, []);

  // Mock data for sorting stats (remain as placeholders for now)
  const telemetry = {
    conveyor: {
      direction: 'Forward',
      speed: 45, // cm/s
      state: systemState.mode === 'FAULT' ? 'Error' : systemState.mode === 'TEST' ? 'Manual' : 'Running'
    },
    sorting: {
      totalClassified: 1248,
      totalSorted: 1205,
      successRate: 96.5,
      currentThroughput: 12, // items/min
      avgThroughput: 10.5
    }
  };

  const metrics = [
    { label: 'Conveyor Direction', value: telemetry.conveyor.direction, icon: FastForward, color: 'text-blue-500' },
    { label: 'Current Speed', value: `${telemetry.conveyor.speed} cm/s`, icon: Gauge, color: 'text-emerald-500' },
    { label: 'System State', value: telemetry.conveyor.state, icon: Activity, color: systemState.mode === 'FAULT' ? 'text-red-500' : 'text-emerald-500' },
  ];

  return (
    <div id="SystemTelemetryPanel" className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
      <div className="flex items-center gap-2 mb-6">
        <Activity className="text-slate-400" size={18} />
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-widest">System Telemetry & Status</h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {metrics.map((m, i) => (
          <div key={i} className="bg-slate-50 p-4 rounded-xl border border-slate-100">
            <p className="text-[10px] font-black text-slate-400 uppercase tracking-tighter mb-1">{m.label}</p>
            <div className="flex items-center justify-between">
              <span className={`text-lg font-black tracking-tight ${m.color}`}>{m.value}</span>
              <m.icon size={18} className="text-slate-300" />
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-6 border-t border-slate-50">
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Total Classified</p>
          <p className="text-xl font-black text-slate-800">{telemetry.sorting.totalClassified}</p>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Success Rate</p>
          <div className="flex items-center gap-2">
            <p className="text-xl font-black text-emerald-600">{telemetry.sorting.successRate}%</p>
            <Target size={14} className="text-emerald-400" />
          </div>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Live Throughput</p>
          <p className="text-xl font-black text-slate-800">{telemetry.sorting.currentThroughput} <span className="text-xs font-medium text-slate-400">i/m</span></p>
        </div>
        <div>
          <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Sorted Success</p>
          <div className="flex items-center gap-2">
            <p className="text-xl font-black text-slate-800">{telemetry.sorting.totalSorted}</p>
            <CheckCircle2 size={14} className="text-emerald-500" />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemTelemetryPanel;
