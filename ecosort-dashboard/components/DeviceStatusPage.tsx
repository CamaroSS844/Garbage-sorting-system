import React, { useState, useEffect } from 'react';
import { Device, DeviceStatus } from '../types';
import { Cpu, Activity, Clock, ShieldCheck, Zap, AlertCircle } from 'lucide-react';

const DeviceStatusPage: React.FC = () => {
  const [device, setDevice] = useState<Device>({
    id: 'esp32_actuator_01',
    name: 'Primary Sorting Controller',
    status: DeviceStatus.OFFLINE,
    lastSeen: '---',
    location: 'Central Processing Hub'
  });

  const fetchStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/esp32/status?device_id=esp32_actuator_01');
      if (response.ok) {
        const data = await response.json();
        // Backend format: { "esp32_actuator_01": "online" }
        const status = data[device.id] === 'online' ? DeviceStatus.ONLINE : DeviceStatus.OFFLINE;
        setDevice(prev => ({
          ...prev,
          status,
          lastSeen: status === DeviceStatus.ONLINE ? 'Just now' : prev.lastSeen
        }));
      }
    } catch (err) {
      console.error("Failed to fetch status from backend:", err);
      // Ensure UI reflects offline state if the backend is unreachable
      setDevice(prev => ({ ...prev, status: DeviceStatus.OFFLINE }));
    }
  };

  useEffect(() => {
    // Initial fetch
    fetchStatus();

    // Set up polling interval at 5 seconds
    const intervalId = setInterval(fetchStatus, 5000);

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, []);

  const isOnline = device.status === DeviceStatus.ONLINE;

  return (
    <div className="max-w-4xl mx-auto h-[calc(100vh-12rem)] flex items-center justify-center">
      <div className="w-full space-y-8 animate-in fade-in zoom-in duration-500">
        
        {/* Main Status Hero */}
        <div className="bg-white rounded-[2.5rem] shadow-xl shadow-slate-200/50 border border-slate-100 p-8 md:p-12 relative overflow-hidden">
          <div className={`absolute top-0 right-0 w-64 h-64 -mr-32 -mt-32 rounded-full opacity-5 blur-3xl transition-colors duration-1000 ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`} />
          
          <div className="relative z-10 flex flex-col items-center text-center">
            <div className={`p-6 rounded-[2rem] mb-6 transition-all duration-500 shadow-lg ${
              isOnline 
                ? 'bg-emerald-500 text-white shadow-emerald-200 scale-110' 
                : 'bg-red-500 text-white shadow-red-200 animate-pulse'
            }`}>
              <Cpu size={48} />
            </div>

            <h2 className="text-3xl font-black text-slate-800 tracking-tight mb-2">
              {device.name}
            </h2>
            <p className="text-slate-400 font-medium mb-8">Node ID: {device.id}</p>

            <div className="flex items-center gap-3 mb-10">
              <div className="relative">
                {isOnline && (
                  <span className="absolute inset-0 rounded-full bg-emerald-400 animate-ping opacity-75" />
                )}
                <div className={`relative w-4 h-4 rounded-full ${isOnline ? 'bg-emerald-500' : 'bg-red-500'}`} />
              </div>
              <span className={`text-xl font-bold uppercase tracking-wider ${isOnline ? 'text-emerald-600' : 'text-red-600'}`}>
                {device.status}
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 w-full max-w-2xl border-t border-slate-50 pt-10">
              <div className="flex flex-col items-center">
                <div className="text-slate-400 mb-1 flex items-center gap-2">
                  <Clock size={16} />
                  <span className="text-xs font-bold uppercase tracking-widest">Last Seen</span>
                </div>
                <p className="text-slate-800 font-bold">{device.lastSeen}</p>
              </div>
              
              <div className="flex flex-col items-center">
                <div className="text-slate-400 mb-1 flex items-center gap-2">
                  <Activity size={16} />
                  <span className="text-xs font-bold uppercase tracking-widest">Latency</span>
                </div>
                <p className="text-slate-800 font-bold">{isOnline ? 'Low' : '---'}</p>
              </div>

              <div className="flex flex-col items-center">
                <div className="text-slate-400 mb-1 flex items-center gap-2">
                  <ShieldCheck size={16} />
                  <span className="text-xs font-bold uppercase tracking-widest">Security</span>
                </div>
                <p className="text-slate-800 font-bold">WPA2-AES</p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Bar / Secondary Info */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-900 rounded-3xl p-6 text-white flex items-center gap-6 shadow-xl shadow-slate-200 transition-transform hover:scale-[1.02]">
            <div className="bg-emerald-500/20 p-4 rounded-2xl text-emerald-400">
              <Zap size={24} />
            </div>
            <div>
              <h4 className="font-bold text-lg">Active Polling</h4>
              <p className="text-slate-400 text-sm">Synchronizing with backend every 5s</p>
            </div>
          </div>

          <div className="bg-white rounded-3xl p-6 border border-slate-100 flex items-center gap-6 shadow-sm hover:shadow-md transition-all">
            <div className={`p-4 rounded-2xl ${isOnline ? 'bg-slate-50 text-slate-400' : 'bg-amber-50 text-amber-500'}`}>
              <AlertCircle size={24} />
            </div>
            <div>
              <h4 className="font-bold text-lg text-slate-800">API Status</h4>
              <p className="text-slate-500 text-sm">{isOnline ? 'Polling successful' : 'Verifying local API (Port 8000)'}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DeviceStatusPage;
