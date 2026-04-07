
import React, { useState, useEffect } from 'react';
import { Wind, RotateCw, MoveUp } from 'lucide-react';

const ActuatorStatusPanel: React.FC = () => {
  const [armPos, setArmPos] = useState({ az: 90, el: 45 });
  const [vacuumOn, setVacuumOn] = useState(true);

  // Simulate movement for visual effect
  useEffect(() => {
    const interval = setInterval(() => {
      setArmPos(prev => ({
        az: prev.az + (Math.random() > 0.5 ? 0.5 : -0.5),
        el: prev.el + (Math.random() > 0.5 ? 0.2 : -0.2)
      }));
      // Pulse vacuum status occasionally in mock mode
      if (Math.random() > 0.95) setVacuumOn(v => !v);
    }, 100);
    return () => clearInterval(interval);
  }, []);

  return (
    <div id="ActuatorStatusPanel" className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
      <div className="flex items-center gap-2 mb-6">
        <RotateCw className="text-slate-400" size={18} />
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-widest">Actuator Activity</h3>
      </div>

      <div className="space-y-6">
        {/* Vacuum Status */}
        <div className="flex items-center justify-between p-4 rounded-xl bg-slate-50 border border-slate-100">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${vacuumOn ? 'bg-emerald-500 text-white animate-pulse' : 'bg-slate-200 text-slate-400'}`}>
              <Wind size={18} />
            </div>
            <div>
              <p className="text-xs font-bold text-slate-800">Vacuum Pump</p>
              <p className="text-[10px] font-medium text-slate-500">{vacuumOn ? 'Active Suction' : 'Idle'}</p>
            </div>
          </div>
          <span className={`text-[10px] font-black uppercase px-2 py-1 rounded-md ${vacuumOn ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-200 text-slate-500'}`}>
            {vacuumOn ? 'ON' : 'OFF'}
          </span>
        </div>

        {/* Arm Position */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <RotateCw size={14} className="text-indigo-400" />
              <span className="text-xs font-bold text-slate-500 uppercase tracking-tighter">Sorting Arm Azimuth</span>
            </div>
            <span className="text-sm font-mono font-bold text-indigo-600">{armPos.az.toFixed(1)}°</span>
          </div>
          <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div 
              className="h-full bg-indigo-500 transition-all duration-100" 
              style={{ width: `${(armPos.az / 180) * 100}%` }}
            />
          </div>

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center gap-2">
              <MoveUp size={14} className="text-amber-400" />
              <span className="text-xs font-bold text-slate-500 uppercase tracking-tighter">Sorting Arm Elevation</span>
            </div>
            <span className="text-sm font-mono font-bold text-amber-600">{armPos.el.toFixed(1)}°</span>
          </div>
          <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
            <div 
              className="h-full bg-amber-500 transition-all duration-100" 
              style={{ width: `${(armPos.el / 90) * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ActuatorStatusPanel;
