import React, { useState } from 'react';
import { Settings, Sliders, Info, Zap, Activity, ShieldAlert, Cpu, Layers } from 'lucide-react';

// Define the types for our system settings state
interface ArmSettings {
  xRestingAngle: number;
  yRestingAngle: number;
  xPickUpPosition: number;
  yPickUpPosition: number;
  movementSpeed: number;
}

interface ActuatorSettings {
  angle: number;
}

type MaterialType = 'None' | 'Metal' | 'Plastic' | 'Paper' | 'Plastic Bottle';

interface SystemState {
  isOnline: boolean;
  conveyorSpeed: number;
  targetSpeed: number;
  arm: ArmSettings;
  actuator: ActuatorSettings;
  bins: {
    bin1: MaterialType;
    bin2: 'Paper' | 'Plastic' | 'None'; // Restricted per requirements
    bin3: MaterialType;
    bin4: MaterialType;
  };
}

export default function SortingSystemDashboard() {
  // System State Management
  const [system, setSystem] = useState<SystemState>({
    isOnline: true,
    conveyorSpeed: 45,
    targetSpeed: 45,
    arm: {
      xRestingAngle: 90,
      yRestingAngle: 45,
      xPickUpPosition: 120,
      yPickUpPosition: 30,
      movementSpeed: 1,
    },
    actuator: {
      angle: 90,
    },
    bins: {
      bin1: 'Metal',
      bin2: 'Paper',
      bin3: 'Plastic Bottle',
      bin4: 'Plastic',
    },
  });

  // Track currently active interactive element configuration
  const [activeTab, setActiveTab] = useState<'conveyor' | 'arm' | 'actuator' | 'bins'>('conveyor');

  // Dynamic Status Color Utility for SVG & Badges
  const getStatusColor = (activeClassName: string, fallbackClassName: string = 'text-slate-400') => {
    if (!system.isOnline) return 'text-rose-500 fill-rose-50';
    return activeClassName;
  };

  const updateArmSetting = (key: keyof ArmSettings, value: number) => {
    setSystem(prev => ({
      ...prev,
      arm: { ...prev.arm, [key]: value }
    }));
  };

  const updateBinSetting = (binKey: keyof SystemState['bins'], value: any) => {
    setSystem(prev => ({
      ...prev,
      bins: { ...prev.bins, [binKey]: value }
    }));
  };

  return (
    <div className="min-h-screen bg-slate-50 p-4 md:p-8 font-sans antialiased text-slate-800">
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        
        {/* LEFT COLUMN: INTERACTIVE HARDWARE DIAGRAM */}
        <div className="lg:col-span-5 bg-white p-6 rounded-3xl border border-slate-100 shadow-sm space-y-4 sticky top-8">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4">
            <div>
              <h2 className="text-xl font-black tracking-tight">Interactive Schematic</h2>
              <p className="text-xs font-medium text-slate-400">Click highlighted zones to adjust variables</p>
            </div>
            
            {/* Status Toggle Switch */}
            <button 
              onClick={() => setSystem(prev => ({ ...prev, isOnline: !prev.isOnline }))}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-black uppercase tracking-wider transition-all border ${
                system.isOnline 
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200 shadow-sm shadow-emerald-100' 
                  : 'bg-rose-50 text-rose-700 border-rose-200'
              }`}
            >
              <Activity size={14} className={system.isOnline ? 'animate-pulse' : ''} />
              {system.isOnline ? 'Online' : 'Offline'}
            </button>
          </div>

          {/* SYSTEM INTERACTIVE SVG */}
          <div className="relative bg-slate-50 rounded-2xl p-4 flex justify-center items-center overflow-hidden border border-slate-100">
            <svg 
              viewBox="0 0 600 900" 
              className={`w-full max-h-[650px] transition-colors duration-500 ${!system.isOnline ? 'filter grayscale opacity-70' : ''}`}
            >
              {/* Outer Workbench boundaries */}
              <rect x="180" y="150" width="320" height="600" fill="none" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4 4" />

              {/* POWER BOX / VACUUM PUMP */}
              <g className="transition-all duration-300">
                <rect x="150" y="10" width="260" height="130" rx="12" fill={system.isOnline ? '#f0fdf4' : '#fff1f2'} stroke={system.isOnline ? '#10b981' : '#f43f5e'} strokeWidth="3" />
                <circle cx="280" cy="75" r="40" fill="none" stroke={system.isOnline ? '#059669' : '#e11d48'} strokeWidth="2" />
                <path d="M 250 75 L 310 75 M 280 45 L 280 105" stroke={system.isOnline ? '#10b981' : '#f43f5e'} strokeWidth="1.5" />
                <text x="280" y="125" textAnchor="middle" className="text-[11px] font-black tracking-wider fill-slate-400 uppercase">Power & Vacuum unit</text>
              </g>

              {/* CONTROL BOX */}
              <g className="transition-all duration-300">
                <rect x="190" y="150" width="200" height="110" rx="10" fill="#ffffff" stroke="#1e293b" strokeWidth="3" />
                <rect x="210" y="165" width="160" height="65" rx="6" fill="#0f172a" />
                <text x="290" y="200" textAnchor="middle" className="text-xs font-bold fill-emerald-400 font-mono tracking-widest">SYS_OK: 200</text>
                <text x="290" y="280" textAnchor="middle" className="text-[10px] font-black fill-slate-400 uppercase tracking-widest">Main Control Box</text>
              </g>

              {/* CONVEYOR BELT (INTERACTIVE) */}
              <g 
                onClick={() => setActiveTab('conveyor')}
                className={`cursor-pointer group transition-all duration-200`}
              >
                <rect 
                  x="235" y="280" width="120" height="450" rx="8" 
                  fill={activeTab === 'conveyor' ? '#ecfdf5' : '#ffffff'} 
                  stroke={activeTab === 'conveyor' ? '#10b981' : '#475569'} 
                  strokeWidth={activeTab === 'conveyor' ? '4' : '2'} 
                />
                {/* Internal belt horizontal marker slots */}
                {[320, 370, 420, 470, 520, 570, 620, 670].map((y, idx) => (
                  <line 
                    key={idx} x1="245" y1={y} x2="345" y2={y} 
                    stroke={activeTab === 'conveyor' ? '#10b981' : '#cbd5e1'} 
                    strokeWidth="3" 
                    className={system.isOnline ? "animate-pulse" : ""}
                  />
                ))}
                <text x="295" y="510" textAnchor="middle" className={`text-xs font-black tracking-widest uppercase transition-colors ${activeTab === 'conveyor' ? 'fill-emerald-600' : 'fill-slate-400 group-hover:fill-slate-700'}`}>
                  Belt ({system.conveyorSpeed} cm/s)
                </text>
              </g>

              {/* ROBOTIC ARM ASSEMBLY (INTERACTIVE) */}
              <g 
                onClick={() => setActiveTab('arm')}
                className="cursor-pointer group transition-all duration-200"
              >
                {/* Structural mount bases */}
                <circle cx="170" cy="540" r="30" fill={activeTab === 'arm' ? '#ecfdf5' : '#f8fafc'} stroke={activeTab === 'arm' ? '#10b981' : '#64748b'} strokeWidth="2" />
                <rect x="140" y="525" width="60" height="30" rx="4" fill="none" stroke="#94a3b8" />
                {/* Arm linkage outline */}
                <path 
                  d="M 170 540 Q 210 500 240 540" 
                  fill="none" 
                  stroke={activeTab === 'arm' ? '#10b981' : '#1e293b'} 
                  strokeWidth={activeTab === 'arm' ? '6' : '4'} 
                  strokeLinecap="round" 
                />
                <text x="140" y="590" className={`text-xs font-black uppercase tracking-wider ${activeTab === 'arm' ? 'fill-emerald-600' : 'fill-slate-400 group-hover:fill-slate-700'}`}>
                  Servo Arm (2x MG996R)
                </text>
              </g>

              {/* SG90 ACTUATOR (INTERACTIVE) */}
              <g 
                onClick={() => setActiveTab('actuator')}
                className="cursor-pointer group transition-all duration-200"
              >
                <rect 
                  x="355" y="445" width="45" height="25" rx="4" 
                  fill={activeTab === 'actuator' ? '#ecfdf5' : '#f1f5f9'} 
                  stroke={activeTab === 'actuator' ? '#10b981' : '#475569'} 
                  strokeWidth="2" 
                />
                {/* Gate lever arm pin */}
                <line 
                  x1="365" y1="457" x2="310" y2="570" 
                  stroke={activeTab === 'actuator' ? '#10b981' : '#0f172a'} 
                  strokeWidth={activeTab === 'actuator' ? '4' : '2'} 
                />
                <text x="415" y="440" className={`text-xs font-black uppercase tracking-wider ${activeTab === 'actuator' ? 'fill-emerald-600' : 'fill-slate-400 group-hover:fill-slate-700'}`}>
                  Actuator SG90
                </text>
              </g>

              {/* BIN REGIONS */}
              <g onClick={() => setActiveTab('bins')} className="cursor-pointer group">
                {/* Bin 1 Top Right */}
                <rect x="365" y="290" width="70" height="130" rx="6" fill="none" stroke={activeTab === 'bins' ? '#10b981' : '#cbd5e1'} strokeWidth={activeTab === 'bins' ? '3' : '1.5'} />
                <text x="400" y="360" textAnchor="middle" className="text-[11px] font-black fill-slate-400">BIN 1</text>
                <text x="400" y="380" textAnchor="middle" className="text-[10px] font-bold fill-indigo-600">{system.bins.bin1}</text>

                {/* Bin 4 Bottom Right */}
                <rect x="365" y="490" width="70" height="150" rx="6" fill="none" stroke={activeTab === 'bins' ? '#10b981' : '#cbd5e1'} strokeWidth={activeTab === 'bins' ? '3' : '1.5'} />
                <text x="400" y="560" textAnchor="middle" className="text-[11px] font-black fill-slate-400">BIN 4</text>
                <text x="400" y="580" textAnchor="middle" className="text-[10px] font-bold fill-indigo-600">{system.bins.bin4}</text>

                {/* Bin 2 Left Side (Next to Arm) */}
                <rect x="80" y="605" width="140" height="70" rx="6" fill="none" stroke={activeTab === 'bins' ? '#10b981' : '#cbd5e1'} strokeWidth={activeTab === 'bins' ? '3' : '1.5'} />
                <text x="150" y="640" textAnchor="middle" className="text-[11px] font-black fill-slate-400">BIN 2</text>
                <text x="150" y="660" textAnchor="middle" className="text-[10px] font-bold fill-indigo-600">{system.bins.bin2}</text>

                {/* Bin 3 Terminal End */}
                <rect x="190" y="745" width="210" height="40" rx="6" fill="none" stroke={activeTab === 'bins' ? '#10b981' : '#cbd5e1'} strokeWidth={activeTab === 'bins' ? '3' : '1.5'} />
                <text x="295" y="770" textAnchor="middle" className="text-[11px] font-black fill-slate-400">BIN 3 (Terminal runoff)</text>
                <text x="295" y="783" textAnchor="middle" className="text-[10px] font-bold fill-indigo-600">{system.bins.bin3}</text>
              </g>

              {/* Lower Storage Box Storage representation */}
              <rect x="200" y="810" width="190" height="80" rx="8" fill="none" stroke="#94a3b8" strokeWidth="2" />
              <text x="295" y="855" textAnchor="middle" className="text-[11px] font-black fill-slate-400 uppercase tracking-widest">Component Travel Box</text>
            </svg>
          </div>

          {/* Fallback layout control panel switcher */}
          <div className="grid grid-cols-4 gap-1 bg-slate-100 p-1.5 rounded-xl">
            {(['conveyor', 'arm', 'actuator', 'bins'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-2 text-[10px] font-black uppercase tracking-wider rounded-lg transition-all ${
                  activeTab === tab ? 'bg-white shadow-sm text-slate-900' : 'text-slate-500 hover:text-slate-800'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
        </div>

        {/* RIGHT COLUMN: CONFIGURATION CONSOLE PANEL */}
        <div className="lg:col-span-7 space-y-6">
          
          {/* Main Container */}
          <div className="bg-white p-6 md:p-8 rounded-3xl border border-slate-100 shadow-sm">
            
            {/* Header section context banner */}
            <div className="flex items-center gap-4 mb-8 border-b border-slate-100 pb-6">
              <div className="p-3 bg-slate-900 text-white rounded-2xl">
                <Settings size={22} />
              </div>
              <div>
                <h2 className="text-2xl font-black tracking-tight text-slate-800">Control Matrix</h2>
                <p className="text-xs font-medium text-slate-500">Fine-tune device physical kinematics and payload rules.</p>
              </div>
            </div>

            {/* Offline state alert block */}
            {!system.isOnline && (
              <div className="mb-8 p-4 bg-rose-50 border border-rose-100 rounded-2xl flex items-start gap-3 text-rose-800 animate-pulse">
                <ShieldAlert size={20} className="shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm font-bold">System Connection Terminated</p>
                  <p className="text-xs opacity-90">Hardware arrays are currently safe-locked. Changes made below will stack locally until applied to the controller.</p>
                </div>
              </div>
            )}

            {/* CONDITIONAL CONTROLS BASED ON ACTIVE SELECTION */}
            <div className="space-y-6 min-h-[380px]">

              {/* TAB 1: CONVEYOR BELT OPTIONS */}
              {activeTab === 'conveyor' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <div className="p-2.5 bg-emerald-50 text-emerald-600 rounded-xl">
                        <Zap size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-slate-800">Conveyor Velocity Matrix</h4>
                        <p className="text-xs text-slate-500">Calibrates NEMA stepper frequency pulses.</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4 bg-slate-50 px-5 py-2.5 rounded-xl border border-slate-100 self-start md:self-auto">
                      <div className="text-center">
                        <p className="text-[9px] font-black text-slate-400 uppercase">Live Pulse</p>
                        <p className="text-base font-black text-slate-800">{system.conveyorSpeed} <span className="text-[10px] text-slate-400">cm/s</span></p>
                      </div>
                      <div className="w-px h-6 bg-slate-200" />
                      <div className="text-center">
                        <p className="text-[9px] font-black text-slate-400 uppercase">Target</p>
                        <p className="text-base font-black text-emerald-600">{system.targetSpeed} <span className="text-[10px] text-slate-400">cm/s</span></p>
                      </div>
                    </div>
                  </div>

                  <div className="p-6 bg-slate-50 rounded-2xl border border-slate-100 space-y-4">
                    <label className="text-xs font-black uppercase text-slate-400 tracking-wider">Adjustment Slider</label>
                    <input 
                      type="range" min="0" max="100" step="1"
                      value={system.targetSpeed}
                      onChange={(e) => setSystem(p => ({ ...p, targetSpeed: parseInt(e.target.value) }))}
                      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                    <div className="flex justify-between text-[10px] text-slate-400 font-bold">
                      <span>0 cm/s (HALT)</span>
                      <span>50 cm/s</span>
                      <span>100 cm/s (MAX)</span>
                    </div>
                  </div>

                  <div className="p-4 bg-blue-50 border border-blue-100 rounded-xl text-blue-800 flex gap-3">
                    <Info size={16} className="shrink-0 mt-0.5" />
                    <p className="text-xs leading-relaxed">
                      <strong>Dynamic Scaling:</strong> MG996R servo kinematics track the NEMA velocity profile instantly to avoid component overshoot during sort windows.
                    </p>
                  </div>
                </div>
              )}

              {/* TAB 2: ROBOTIC ARM MODULES */}
              {activeTab === 'arm' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
                      <Cpu size={20} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800">Twin MG996R Servo Alignment</h4>
                      <p className="text-xs text-slate-500">Configure angular position profiles and speed limits.</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-3">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">X-Axis Resting Angle</span>
                      <div className="flex items-center justify-between"><span className="text-xl font-mono font-bold text-slate-700">{system.arm.xRestingAngle}°</span></div>
                      <input type="range" min="0" max="180" value={system.arm.xRestingAngle} onChange={(e) => updateArmSetting('xRestingAngle', parseInt(e.target.value))} className="w-full h-1.5 accent-indigo-600" />
                    </div>

                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-3">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">Y-Axis Resting Angle</span>
                      <div className="flex items-center justify-between"><span className="text-xl font-mono font-bold text-slate-700">{system.arm.yRestingAngle}°</span></div>
                      <input type="range" min="0" max="180" value={system.arm.yRestingAngle} onChange={(e) => updateArmSetting('yRestingAngle', parseInt(e.target.value))} className="w-full h-1.5 accent-indigo-600" />
                    </div>

                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-3">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">X Pick-up Intercept</span>
                      <div className="flex items-center justify-between"><span className="text-xl font-mono font-bold text-slate-700">{system.arm.xPickUpPosition}°</span></div>
                      <input type="range" min="0" max="180" value={system.arm.xPickUpPosition} onChange={(e) => updateArmSetting('xPickUpPosition', parseInt(e.target.value))} className="w-full h-1.5 accent-indigo-600" />
                    </div>

                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 space-y-3">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">Y Pick-up Intercept</span>
                      <div className="flex items-center justify-between"><span className="text-xl font-mono font-bold text-slate-700">{system.arm.yPickUpPosition}°</span></div>
                      <input type="range" min="0" max="180" value={system.arm.yPickUpPosition} onChange={(e) => updateArmSetting('yPickUpPosition', parseInt(e.target.value))} className="w-full h-1.5 accent-indigo-600" />
                    </div>
                  </div>

                  <div className="p-4 bg-slate-50 rounded-2xl border border-slate-100 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">Velocity Step Factor</span>
                      <span className="px-2.5 py-1 bg-indigo-100 text-indigo-700 font-mono text-xs font-bold rounded-md">{system.arm.movementSpeed}x Base</span>
                    </div>
                    <div className="grid grid-cols-5 gap-2">
                      {[1, 2, 3, 4, 5].map((multiplier) => (
                        <button
                          key={multiplier}
                          type="button"
                          onClick={() => updateArmSetting('movementSpeed', multiplier)}
                          className={`py-2 text-sm font-mono font-bold rounded-xl border transition-all ${
                            system.arm.movementSpeed === multiplier 
                              ? 'bg-indigo-600 border-indigo-600 text-white shadow-sm' 
                              : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          {multiplier}x
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: SG90 ACTUATOR OPTIONS */}
              {activeTab === 'actuator' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-amber-50 text-amber-600 rounded-xl">
                      <Sliders size={20} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800">SG90 Micro-Actuator Gate</h4>
                      <p className="text-xs text-slate-500">Deflection arm gate mechanics calibration.</p>
                    </div>
                  </div>

                  <div className="p-6 bg-slate-50 rounded-2xl border border-slate-100 space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-black text-slate-500 uppercase tracking-wide">Gate Throw Position Angle</span>
                      <span className="text-2xl font-mono font-black text-amber-600">{system.actuator.angle}°</span>
                    </div>
                    
                    <input 
                      type="range" min="0" max="180" step="1"
                      value={system.actuator.angle}
                      onChange={(e) => setSystem(p => ({ ...p, actuator: { angle: parseInt(e.target.value) } }))}
                      className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-amber-500"
                    />
                    
                    <div className="grid grid-cols-3 text-center text-[11px] font-bold text-slate-400 pt-2">
                      <button onClick={() => setSystem(p => ({ ...p, actuator: { angle: 0 } }))} className="hover:text-slate-700 text-left">0° (Retracted)</button>
                      <button onClick={() => setSystem(p => ({ ...p, actuator: { angle: 90 } }))} className="hover:text-slate-700 text-center">90° (Midway)</button>
                      <button onClick={() => setSystem(p => ({ ...p, actuator: { angle: 180 } }))} className="hover:text-slate-700 text-right">180° (Extended)</button>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 4: BIN ROUTING FILTER CRITERIA */}
              {activeTab === 'bins' && (
                <div className="space-y-6 animate-fadeIn">
                  <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-sky-50 text-sky-600 rounded-xl">
                      <Layers size={20} />
                    </div>
                    <div>
                      <h4 className="font-bold text-slate-800">Material Destination Matrix</h4>
                      <p className="text-xs text-slate-500">Map specific classification outputs to designated collection bins.</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* BIN 1 */}
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 flex flex-col justify-between gap-2">
                      <div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Bin 1 Configuration</span>
                        <p className="text-xs text-slate-500">Upper discharge line</p>
                      </div>
                      <select 
                        value={system.bins.bin1} 
                        onChange={(e) => updateBinSetting('bin1', e.target.value)}
                        className="w-full px-3 py-2 bg-white border border-slate-200 rounded-xl text-sm font-medium focus:outline-none focus:border-sky-500"
                      >
                        {['None', 'Metal', 'Plastic', 'Paper', 'Plastic Bottle'].map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>

                    {/* BIN 2 (RESTRICTED SPECIFICATION REQUIREMENT) */}
                    <div className="p-4 bg-amber-50/50 border border-amber-100 rounded-xl flex flex-col justify-between gap-2">
                      <div>
                        <div className="flex items-center justify-between">
                          <span className="text-[10px] font-black text-amber-800 uppercase tracking-wider">Bin 2 (Restricted)</span>
                          <span className="text-[9px] bg-amber-200 text-amber-900 px-1.5 py-0.5 rounded font-bold uppercase">Arm Proxy</span>
                        </div>
                        <p className="text-xs text-amber-700/80">Restricted explicitly to Paper/Plastic configurations.</p>
                      </div>
                      <select 
                        value={system.bins.bin2} 
                        onChange={(e) => updateBinSetting('bin2', e.target.value)}
                        className="w-full px-3 py-2 bg-white border border-amber-200 rounded-xl text-sm font-bold text-amber-900 focus:outline-none"
                      >
                        {['None', 'Paper', 'Plastic'].map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>

                    {/* BIN 3 */}
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 flex flex-col justify-between gap-2">
                      <div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Bin 3 Configuration</span>
                        <p className="text-xs text-slate-500">Terminal conveyor runoff</p>
                      </div>
                      <select 
                        value={system.bins.bin3} 
                        onChange={(e) => updateBinSetting('bin3', e.target.value)}
                        className="w-full px-3 py-2 bg-white border border-slate-200 rounded-xl text-sm font-medium focus:outline-none focus:border-sky-500"
                      >
                        {['None', 'Metal', 'Plastic', 'Paper', 'Plastic Bottle'].map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>

                    {/* BIN 4 */}
                    <div className="p-4 bg-slate-50 rounded-xl border border-slate-100 flex flex-col justify-between gap-2">
                      <div>
                        <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Bin 4 Configuration</span>
                        <p className="text-xs text-slate-500">Lower sorting bypass</p>
                      </div>
                      <select 
                        value={system.bins.bin4} 
                        onChange={(e) => updateBinSetting('bin4', e.target.value)}
                        className="w-full px-3 py-2 bg-white border border-slate-200 rounded-xl text-sm font-medium focus:outline-none focus:border-sky-500"
                      >
                        {['None', 'Metal', 'Plastic', 'Paper', 'Plastic Bottle'].map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* ACTION CTA ROW */}
            <div className="mt-8 pt-6 border-t border-slate-100 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 italic">
                Active context: <strong className="uppercase text-slate-600">{activeTab}</strong>
              </span>
              <button 
                type="button"
                onClick={() => setSystem(prev => ({ ...prev, conveyorSpeed: prev.targetSpeed }))}
                disabled={!system.isOnline}
                className={`px-6 py-3 rounded-xl font-black text-xs uppercase tracking-wider transition-all shadow-md ${
                  system.isOnline 
                    ? 'bg-slate-900 text-white hover:bg-slate-800 shadow-slate-200 active:scale-95' 
                    : 'bg-slate-100 text-slate-400 cursor-not-allowed shadow-none'
                }`}
              >
                Sync & Commit EEPROM
              </button>
            </div>

          </div>
        </div>

      </div>
    </div>
  );
}