
import React, { useState } from 'react';
import { Settings, Sliders, Info, Zap } from 'lucide-react';

const SystemSettingsPanel: React.FC = () => {
  const [conveyorSpeed, setConveyorSpeed] = useState(45);
  const [targetSpeed, setTargetSpeed] = useState(45);

  return (
    <div id="SystemSettingsPanel" className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white p-8 rounded-3xl border border-slate-100 shadow-sm">
        <div className="flex items-center gap-4 mb-8">
          <div className="p-3 bg-slate-900 text-white rounded-2xl">
            <Settings size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-black text-slate-800">System Configuration</h2>
            <p className="text-sm font-medium text-slate-500">Global operational parameters and motor calibration.</p>
          </div>
        </div>

        <div className="space-y-12">
          {/* Conveyor Speed Setting */}
          <div className="space-y-6">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-50 text-emerald-600 rounded-lg">
                  <Zap size={20} />
                </div>
                <div>
                  <h4 className="font-bold text-slate-800">Conveyor Speed Configuration</h4>
                  <p className="text-xs text-slate-500">Adjust the primary transport motor velocity.</p>
                </div>
              </div>
              <div className="flex items-center gap-4 bg-slate-50 px-6 py-3 rounded-2xl border border-slate-100">
                <div className="text-center">
                  <p className="text-[10px] font-black text-slate-400 uppercase">Current</p>
                  <p className="text-lg font-black text-slate-800">{conveyorSpeed} <span className="text-[10px] text-slate-400">cm/s</span></p>
                </div>
                <div className="w-px h-8 bg-slate-200" />
                <div className="text-center">
                  <p className="text-[10px] font-black text-slate-400 uppercase">Target</p>
                  <p className="text-lg font-black text-emerald-600">{targetSpeed} <span className="text-[10px] text-slate-400">cm/s</span></p>
                </div>
              </div>
            </div>

            <div className="p-8 bg-slate-50 rounded-[2rem] border border-slate-100">
              <input 
                type="range" 
                min="0" 
                max="100" 
                step="1"
                value={targetSpeed}
                onChange={(e) => setTargetSpeed(parseInt(e.target.value))}
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500 mb-6"
              />
              
              <div className="flex items-start gap-3 p-4 bg-blue-50 border border-blue-100 rounded-2xl text-blue-700">
                <Info size={18} className="shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-bold">Scaling Dependency Rule</p>
                  <p className="text-xs leading-relaxed opacity-80">
                    Robotic Arm motion velocity scales automatically with conveyor belt speed to maintain synchronization during item pick-up. High speed may increase mechanical wear.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-12 flex justify-end">
          <button 
            onClick={() => setConveyorSpeed(targetSpeed)}
            className="px-8 py-4 bg-slate-900 text-white rounded-2xl font-black shadow-xl shadow-slate-200 hover:bg-slate-800 transition-all active:scale-95"
          >
            Apply Configurations
          </button>
        </div>
      </div>
    </div>
  );
};

export default SystemSettingsPanel;
