
import React, { useState, useEffect } from 'react';
import { Layout, Camera, Cpu, AlertCircle, BarChart3, Menu, X, Trash2, Sliders, Settings } from 'lucide-react';
import LiveCameraFeed from './components/LiveCameraFeed';
import DeviceStatusPage from './components/DeviceStatusPage';
import FailedInferenceReview from './components/FailedInferenceReview';
import SystemReports from './components/SystemReports';
import SystemTestMode from './components/SystemtestMode';
import ActuatorStatusPanel from './components/ActuatorStatusPanel';
import SystemSettingsPanel from './components/SystemSettingsPanel';
import './src/index.css';

enum Page {
  FEED = 'feed',
  DEVICES = 'devices',
  REVIEW = 'review',
  REPORTS = 'reports',
  TEST = 'test',
  SETTINGS = 'settings'
}

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<Page>(Page.FEED);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [activeNodes, setActiveNodes] = useState<number>(0);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await fetch('http://localhost:8000/reports');
        if (response.ok) {
          const data = await response.json();
          setActiveNodes(data.online_devices || 0);
        }
      } catch (err) {
        console.error("Health fetch failed:", err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: Page.FEED, label: 'Live Feed', icon: Camera },
    { id: Page.DEVICES, label: 'Device Status', icon: Cpu },
    { id: Page.REVIEW, label: 'Failed Inference', icon: AlertCircle },
    { id: Page.REPORTS, label: 'System Reports', icon: BarChart3 },
    { id: Page.TEST, label: 'System Test Mode', icon: Sliders },
    { id: Page.SETTINGS, label: 'Settings', icon: Settings },
  ];

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  return (
    <div className="min-h-screen flex bg-slate-50">
      {/* Mobile Backdrop */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={toggleSidebar}
        />
      )}

      {/* Sidebar */}
      <aside className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-slate-900 text-white transform transition-transform duration-200 ease-in-out
        lg:relative lg:translate-x-0
        ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="p-6 flex items-center gap-3">
          <div className="bg-emerald-500 p-2 rounded-lg">
            <Trash2 className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight">AI sorting system</span>
        </div>

        <nav className="mt-6 px-4 space-y-2">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => {
                setCurrentPage(item.id);
                setIsSidebarOpen(false);
              }}
              className={`
                w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-colors
                ${currentPage === item.id 
                  ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' 
                  : item.id === Page.TEST 
                    ? 'text-amber-400 hover:bg-amber-400/10' 
                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'}
                ${currentPage === Page.TEST && item.id === Page.TEST ? 'bg-amber-500 text-white shadow-amber-500/20' : ''}
              `}
            >
              <item.icon size={20} />
              <span className="font-medium">{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="absolute bottom-8 left-0 right-0 px-6">
          <div className="p-4 bg-slate-800/50 rounded-xl border border-slate-700">
            <p className="text-xs text-slate-500 font-medium uppercase tracking-wider mb-1">System Health</p>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full animate-pulse ${activeNodes > 0 ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <span className="text-sm font-semibold">Nodes Active ({activeNodes})</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        <header className="h-16 border-b bg-white flex items-center justify-between px-6 lg:px-10 sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button onClick={toggleSidebar} className="lg:hidden p-2 text-slate-500 hover:bg-slate-100 rounded-lg">
              <Menu size={20} />
            </button>
            <h1 className="text-lg font-semibold text-slate-800">
              {navItems.find(i => i.id === currentPage)?.label}
            </h1>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="hidden sm:flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-full text-xs font-medium text-slate-600">
              <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full" />
              API Connected
            </div>
            <div className="w-8 h-8 rounded-full bg-slate-200 overflow-hidden">
              <img src="https://picsum.photos/32/32" alt="Avatar" />
            </div>
          </div>
        </header>

        <div className="p-6 lg:p-10 flex-1 overflow-auto">
          {currentPage === Page.FEED && <LiveCameraFeed />}
          {currentPage === Page.DEVICES && <DeviceStatusPage />}
          {currentPage === Page.REVIEW && <FailedInferenceReview />}
          {currentPage === Page.REPORTS && <SystemReports />}
          {currentPage === Page.TEST && <SystemTestMode />}
          {currentPage === Page.SETTINGS && <SystemSettingsPanel />}
        </div>
      </main>
    </div>
  );
};

export default App;
