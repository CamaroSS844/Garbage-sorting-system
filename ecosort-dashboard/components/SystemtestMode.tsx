
import React, { useState, useEffect, useRef } from 'react';
import { AlertTriangle, Power, Settings2, RotateCw, ArrowUpCircle, ArrowDownCircle, StopCircle, RefreshCcw } from 'lucide-react';

const SystemTestMode: React.FC = () => {
  const [systemState, setSystemState] = useState({ mode: 'RUN', fault: null });
  const [arm, setArm] = useState({ azimuth: 90, elevation: 90 });
  const [vacuumOn, setVacuumOn] = useState(false);
  const [conveyorSpeed, setConveyorSpeed] = useState(0); // -100 to 100
  const wsRef = useRef<WebSocket | null>(null);

  const fetchSystemState = async () => {
    try {
      const response = await fetch('http://localhost:8000/system/state');
      if (response.ok) {
        const data = await response.json();
        setSystemState(data);
      }
    } catch (err) {
      console.error("Failed to fetch system state:", err);
    }
  };

  const syncControlState = async () => {
    try {
      const response = await fetch('http://localhost:8000/test/control');
      if (response.ok) {
        const data = await response.json();
        setArm({
          azimuth: data.arm.azimuth * 90 + 90,
          elevation: data.arm.elevation * 90 + 90
        });
        setConveyorSpeed(data.conveyor * 100);
        setVacuumOn(data.vacuum);
      }
    } catch (err) {}
  };

  useEffect(() => {
    fetchSystemState();
    const interval = setInterval(fetchSystemState, 3000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (systemState.mode === 'TEST') {
      syncControlState();
      const ws = new WebSocket('ws://localhost:8000/ws/control');
      wsRef.current = ws;
      return () => {
        ws.close();
        wsRef.current = null;
      };
    }
  }, [systemState.mode]);

  const sendControl = (newArm = arm, newConv = conveyorSpeed, newVac = vacuumOn) => {
    if (systemState.mode !== 'TEST') return;
    
    const payload = {
      arm: {
        azimuth: (newArm.azimuth - 90) / 90,
        elevation: (newArm.elevation - 90) / 90
      },
      conveyor: newConv / 100,
      vacuum: newVac
    };

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(payload));
    } else {
      fetch(`http://localhost:8000/test/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).catch(() => {});
    }
  };

  const handleModeToggle = async (target: 'RUN' | 'TEST') => {
    if (systemState.mode === 'FAULT') return;
    try {
      const res = await fetch(`http://localhost:8000/system/mode/${target}`, { method: 'POST' });
      if (res.ok) fetchSystemState();
    } catch (err) {}
  };

  const handleEStop = async () => {
    try {
      await fetch('http://localhost:8000/system/emergency-stop', { method: 'POST' });
      fetchSystemState();
    } catch (err) {}
  };

  const handleReset = async () => {
    try {
      await fetch('http://localhost:8000/system/reset', { method: 'POST' });
      fetchSystemState();
    } catch (err) {}
  };

  const isTestMode = systemState.mode === 'TEST';
  const isEStop = systemState.mode === 'FAULT';
  const isControlsDisabled = !isTestMode || isEStop;

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Test Mode Header */}
      <div className={`p-6 rounded-3xl border-2 flex flex-col md:flex-row items-center justify-between gap-6 transition-all duration-300 ${
        isEStop ? 'bg-red-50 border-red-500 shadow-lg shadow-red-200' : 'bg-amber-50 border-amber-500 shadow-lg shadow-amber-200'
      }`}>
        <div className="flex items-center gap-4 text-center md:text-left">
          <div className={`p-4 rounded-2xl ${isEStop ? 'bg-red-500' : 'bg-amber-500'} text-white shadow-lg`}>
            {isEStop ? <AlertTriangle size={32} className="animate-pulse" /> : <Settings2 size={32} />}
          </div>
          <div>
            <h2 className="text-2xl font-black text-slate-800 flex items-center gap-2 justify-center md:justify-start">
              SYSTEM TEST MODE
              {isEStop && <span className="bg-red-600 text-white px-2 py-0.5 rounded text-xs uppercase tracking-tighter ml-2">Emergency Stop Active</span>}
            </h2>
            <p className={`text-sm font-bold ${isEStop ? 'text-red-700' : 'text-amber-700'}`}>
              Commissioning Interface • Status: {systemState.mode}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-4">
          {/* RUN/TEST TOGGLE */}
          <div className="bg-white p-1 rounded-2xl border-2 border-slate-200 flex shadow-inner">
            <button 
              onClick={() => handleModeToggle('RUN')}
              className={`px-6 py-2 rounded-xl text-sm font-black transition-all ${!isTestMode ? 'bg-emerald-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}
              disabled={isEStop}
            >
              RUN MODE
            </button>
            <button 
              onClick={() => handleModeToggle('TEST')}
              className={`px-6 py-2 rounded-xl text-sm font-black transition-all ${isTestMode ? 'bg-amber-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-600'}`}
              disabled={isEStop}
            >
              TEST MODE
            </button>
          </div>

          {/* E-STOP BUTTON */}
          {isEStop ? (
            <button 
              onClick={handleReset}
              className="flex items-center gap-2 px-8 py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-2xl font-black shadow-xl transition-all active:scale-95"
            >
              <RefreshCcw size={20} />
              RESET SYSTEM
            </button>
          ) : (
            <button 
              onClick={handleEStop}
              className="flex items-center gap-2 px-8 py-3 bg-red-600 hover:bg-red-700 text-white rounded-2xl font-black shadow-xl shadow-red-200 transition-all active:scale-90 animate-pulse"
            >
              <Power size={20} />
              EMERGENCY STOP
            </button>
          )}
        </div>
      </div>

      <div className={`grid grid-cols-1 lg:grid-cols-3 gap-6 transition-opacity duration-300 ${!isTestMode ? 'opacity-40 grayscale pointer-events-none' : ''}`}>
        
        {/* Robotic Arm Test Section */}
        <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-indigo-50 text-indigo-600 rounded-xl"><RotateCw size={24} /></div>
            <h3 className="text-xl font-black text-slate-800">Robotic Arm</h3>
          </div>
          
          <div className="space-y-12 flex-1 flex flex-col justify-center">
            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-400 uppercase tracking-widest">Azimuth Rotation</label>
                <span className="text-2xl font-black text-indigo-600">{arm.azimuth.toFixed(0)}°</span>
              </div>
              <input 
                type="range" min="0" max="180" 
                value={arm.azimuth} 
                disabled={isControlsDisabled}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  const newArm = { ...arm, azimuth: val };
                  setArm(newArm);
                  sendControl(newArm);
                }}
                className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>

            <div className="space-y-4">
              <div className="flex justify-between items-end">
                <label className="text-xs font-black text-slate-400 uppercase tracking-widest">Elevation Height</label>
                <span className="text-2xl font-black text-indigo-600">{arm.elevation.toFixed(0)}°</span>
              </div>
              <input 
                type="range" min="0" max="180" 
                value={arm.elevation} 
                disabled={isControlsDisabled}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  const newArm = { ...arm, elevation: val };
                  setArm(newArm);
                  sendControl(newArm);
                }}
                className="w-full h-2 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-indigo-600"
              />
            </div>
          </div>
        </div>

        {/* Conveyor Belt Joystick Section */}
        <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-emerald-50 text-emerald-600 rounded-xl"><Settings2 size={24} /></div>
            <h3 className="text-xl font-black text-slate-800">Conveyor Belt</h3>
          </div>

          <div className="flex flex-1 items-center justify-center gap-10">
            {/* Joystick Slider */}
            <div className="relative h-64 w-16 bg-slate-100 rounded-3xl flex items-center justify-center border-4 border-slate-50 shadow-inner">
              <div className="absolute top-2 text-[10px] font-black text-slate-400 uppercase">Fwd</div>
              <div className="absolute bottom-2 text-[10px] font-black text-slate-400 uppercase">Rev</div>
              <input 
                type="range" min="-100" max="100" 
                value={conveyorSpeed} 
                disabled={isControlsDisabled}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  setConveyorSpeed(val);
                  sendControl(arm, val, vacuumOn);
                }}
                onMouseUp={() => { setConveyorSpeed(0); sendControl(arm, 0, vacuumOn); }}
                onTouchEnd={() => { setConveyorSpeed(0); sendControl(arm, 0, vacuumOn); }}
                className="vertical-slider h-56 w-2 appearance-none bg-transparent cursor-pointer accent-emerald-500"
                style={{ WebkitAppearance: 'slider-vertical' } as any}
              />
            </div>

            <div className="space-y-4 text-center">
               <div className={`p-6 rounded-3xl flex flex-col items-center justify-center transition-all ${
                 conveyorSpeed > 0 ? 'bg-emerald-500 text-white shadow-xl shadow-emerald-200' : 
                 conveyorSpeed < 0 ? 'bg-blue-500 text-white shadow-xl shadow-blue-200' : 'bg-slate-100 text-slate-400'
               }`}>
                 {conveyorSpeed > 0 && <ArrowUpCircle size={40} className="animate-bounce" />}
                 {conveyorSpeed < 0 && <ArrowDownCircle size={40} className="animate-bounce" />}
                 {conveyorSpeed === 0 && <StopCircle size={40} />}
                 <span className="text-2xl font-black mt-2">{Math.abs(conveyorSpeed)}%</span>
               </div>
               <p className="text-xs font-bold text-slate-400 uppercase tracking-tighter">Manual Speed Override</p>
            </div>
          </div>
        </div>

        {/* Vacuum Pump Section */}
        <div className="bg-white p-8 rounded-[2rem] shadow-sm border border-slate-100 flex flex-col h-full">
          <div className="flex items-center gap-3 mb-8">
            <div className="p-3 bg-amber-50 text-amber-600 rounded-xl"><RefreshCcw size={24} /></div>
            <h3 className="text-xl font-black text-slate-800">Vacuum Pump</h3>
          </div>

          <div className="flex flex-1 flex-col items-center justify-center gap-8">
            <button
              disabled={isControlsDisabled}
              onClick={() => {
                const newState = !vacuumOn;
                setVacuumOn(newState);
                sendControl(arm, conveyorSpeed, newState);
              }}
              className={`w-32 h-32 rounded-full border-8 transition-all duration-300 flex flex-col items-center justify-center gap-2 shadow-2xl ${
                vacuumOn 
                  ? 'bg-amber-500 border-amber-300 text-white shadow-amber-200 scale-110' 
                  : 'bg-slate-100 border-slate-200 text-slate-400'
              }`}
            >
              <Power size={40} />
              <span className="font-black text-xs">{vacuumOn ? 'ACTIVE' : 'OFF'}</span>
            </button>
            <div className="text-center">
              <p className="text-sm font-bold text-slate-700">{vacuumOn ? 'Suction Engaged' : 'Pump Standby'}</p>
              <p className="text-[10px] text-slate-400 uppercase font-black mt-1">Manual Bypass Active</p>
            </div>
          </div>
        </div>

      </div>

      {!isTestMode && (
        <div className="bg-slate-900 rounded-3xl p-8 text-white flex flex-col md:flex-row items-center gap-8 shadow-2xl">
          <div className="bg-emerald-500/20 p-6 rounded-2xl text-emerald-400 border border-emerald-500/30">
            <Settings2 size={40} />
          </div>
          <div className="text-center md:text-left">
            <h4 className="text-2xl font-black mb-2">Commissioning Lock Active</h4>
            <p className="text-slate-400 max-w-xl">
              Manual hardware controls are currently disabled for safety. Switch to <strong>TEST MODE</strong> above to begin hardware calibration and diagnostic testing.
            </p>
          </div>
          <button 
            onClick={() => handleModeToggle('TEST')}
            className="md:ml-auto px-8 py-4 bg-emerald-500 hover:bg-emerald-600 text-white rounded-2xl font-black transition-all shadow-lg active:scale-95"
            disabled={isEStop}
          >
            Unlock Controls
          </button>
        </div>
      )}
    </div>
  );
};

export default SystemTestMode;
