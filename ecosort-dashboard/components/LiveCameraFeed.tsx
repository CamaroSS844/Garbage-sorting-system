import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Camera, RefreshCw, CheckCircle2, XCircle, Clock, Video, X, Tag, Info,
  Globe, Monitor, AlertCircle, Zap, Cpu, Play, Square, SlidersHorizontal,
  Settings2, Activity, Target, Layers
} from 'lucide-react';
import SystemTelemetryPanel from './SystemTelemetryPanel';
import ActuatorStatusPanel from './ActuatorStatusPanel';
import { InferenceMessage, InferenceDetection } from '../types';

const BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/inference';

const LiveCameraFeed: React.FC = () => {
  const [cameraUrl, setCameraUrl] = useState('http://192.168.1.100:81/stream');
  const [isCameraConnected, setIsCameraConnected] = useState(false);
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [inferenceMode, setInferenceMode] = useState<'local' | 'cloud'>('local');
  const [thresholds, setThresholds] = useState({ conf: 0.5, iou: 0.45 });
  const [trackingEnabled, setTrackingEnabled] = useState(false);
  const [cloudInterval, setCloudInterval] = useState(60);
  const [lastInferenceTime, setLastInferenceTime] = useState<number>(0);

  // FIX 1: Store detections in a ref, NOT state.
  // State updates trigger re-renders and re-create drawDetections via useCallback,
  // which restarts the rAF loop. A ref is mutated in-place — the loop always
  // reads the latest value without any of that overhead.
  const detectionsRef = useRef<InferenceDetection[]>([]);

  // Keep a separate state only for the sidebar list — updated at lower priority
  const [detectionsList, setDetectionsList] = useState<InferenceDetection[]>([]);

  const canvasOverlayRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const inferenceModeRef = useRef<'local' | 'cloud'>('local');

  // Keep inferenceModeRef in sync so drawDetections (which captures the ref)
  // always sees the current mode without being recreated.
  useEffect(() => {
    inferenceModeRef.current = inferenceMode;
  }, [inferenceMode]);

  // -------------------------------------------------------
  // WebSocket Connection
  // -------------------------------------------------------
  useEffect(() => {
    const connectWS = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => console.log('[ws] Inference WebSocket connected');

      ws.onmessage = (event) => {
        try {
          const data: InferenceMessage = JSON.parse(event.data);

          if (data.type === 'camera_status') {
            setIsCameraConnected(!!data.connected);

          } else if (data.type === 'detection' || data.type === 'tracked') {
            const objs: InferenceDetection[] = data.objects || data.detections || [];

            // FIX 2: Mutate the ref — does NOT trigger a re-render or restart rAF.
            detectionsRef.current = objs;

            // Update the sidebar list separately (lower frequency is fine here)
            setDetectionsList(objs);

            if (data.inference_time_ms) {
              setLastInferenceTime(data.inference_time_ms);
            }
          }
        } catch (e) {
          console.error('[ws] Parse error', e);
        }
      };

      ws.onclose = () => {
        console.log('[ws] Disconnected, reconnecting in 3s...');
        setTimeout(connectWS, 3000);
      };

      wsRef.current = ws;
    };

    connectWS();
    return () => wsRef.current?.close();
  }, []);

  // -------------------------------------------------------
  // Poll Initial Status
  // -------------------------------------------------------
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [camRes, modeRes, threshRes] = await Promise.all([
          fetch(`${BASE_URL}/inference/camera_status`),
          fetch(`${BASE_URL}/inference/get_mode`),
          fetch(`${BASE_URL}/inference/get_thresholds`)
        ]);
        if (camRes.ok) {
          const d = await camRes.json();
          setIsCameraConnected(d.connected);
        }
        if (modeRes.ok) {
          const d = await modeRes.json();
          setInferenceMode(d.mode);
        }
        if (threshRes.ok) {
          const d = await threshRes.json();
          setThresholds(d);
        }
      } catch (e) {
        console.error('[init] Failed to fetch status', e);
      }
    };
    fetchStatus();
  }, []);

  // -------------------------------------------------------
  // FIX 3: Canvas draw function reads from the ref —
  // stable identity, never recreated, no deps.
  // -------------------------------------------------------
  const drawDetections = useCallback(() => {
    const canvas = canvasOverlayRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Sync canvas size to container once per frame (cheap compare)
    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const detections = detectionsRef.current;
    if (detections.length === 0) return;

    const videoWidth = 640;
    const videoHeight = 480;
    const scaleX = canvas.width / videoWidth;
    const scaleY = canvas.height / videoHeight;

    detections.forEach((det) => {
      let x: number, y: number, w: number, h: number;

      if (inferenceModeRef.current === 'local') {
        // bbox: [x1, y1, x2, y2] pixel coords
        x = det.bbox[0] * scaleX;
        y = det.bbox[1] * scaleY;
        w = (det.bbox[2] - det.bbox[0]) * scaleX;
        h = (det.bbox[3] - det.bbox[1]) * scaleY;
      } else {
        // bbox: [center_x, center_y, width, height] normalized
        const cx = det.bbox[0] * videoWidth;
        const cy = det.bbox[1] * videoHeight;
        const bw = det.bbox[2] * videoWidth;
        const bh = det.bbox[3] * videoHeight;
        x = (cx - bw / 2) * scaleX;
        y = (cy - bh / 2) * scaleY;
        w = bw * scaleX;
        h = bh * scaleY;
      }

      const className = det.class_name || det.class || 'Unknown';
      const color =
        className.toLowerCase().includes('plastic') ? '#10b981' :
        className.toLowerCase().includes('metal')   ? '#3b82f6' : '#f59e0b';

      // Bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // Label
      const label = `${className} ${(det.confidence * 100).toFixed(0)}%${det.track_id ? ` #${det.track_id}` : ''}`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - 20, textWidth + 10, 20);

      ctx.fillStyle = 'white';
      ctx.fillText(label, x + 5, y - 5);
    });
  }, []); // empty deps — stable forever

  // -------------------------------------------------------
  // FIX 4: Continuous rAF loop — runs every frame, completely
  // independent of React state or WebSocket messages.
  // drawDetections is stable so this effect runs only once.
  // -------------------------------------------------------
  useEffect(() => {
    let animId: number;
    const loop = () => {
      drawDetections();
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [drawDetections]);

  // -------------------------------------------------------
  // API Control Handlers
  // -------------------------------------------------------
  const handleSetCamera = async () => {
    try {
      await fetch(`${BASE_URL}/inference/set_camera?url=${encodeURIComponent(cameraUrl)}`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to set camera', e);
    }
  };

  const toggleInference = async () => {
    const endpoint = isInferenceRunning ? 'stop' : 'start';
    try {
      const res = await fetch(`${BASE_URL}/inference/${endpoint}`, { method: 'POST' });
      if (res.ok) setIsInferenceRunning(!isInferenceRunning);
    } catch (e) {
      console.error(`Failed to ${endpoint} inference`, e);
    }
  };

  const handleModeChange = async (mode: 'local' | 'cloud') => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_mode/${mode}`, { method: 'POST' });
      if (res.ok) setInferenceMode(mode);
    } catch (e) {
      console.error('Failed to set mode', e);
    }
  };

  const handleThresholdChange = async (newThresholds: typeof thresholds) => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_thresholds`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newThresholds)
      });
      if (res.ok) setThresholds(newThresholds);
    } catch (e) {
      console.error('Failed to set thresholds', e);
    }
  };

  const handleTrackingToggle = async () => {
    const newState = !trackingEnabled;
    try {
      const res = await fetch(`${BASE_URL}/inference/tracking`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enable: newState })
      });
      if (res.ok) setTrackingEnabled(newState);
    } catch (e) {
      console.error('Failed to toggle tracking', e);
    }
  };

  const handleResetTracker = async () => {
    try {
      await fetch(`${BASE_URL}/inference/tracking/reset`, { method: 'POST' });
    } catch (e) {
      console.error('Failed to reset tracker', e);
    }
  };

  const handleCloudIntervalChange = async (seconds: number) => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_cloud_interval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds })
      });
      if (res.ok) setCloudInterval(seconds);
    } catch (e) {
      console.error('Failed to set cloud interval', e);
    }
  };

  // -------------------------------------------------------
  // Render
  // -------------------------------------------------------
  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-20">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

        {/* Main Feed */}
        <div className="lg:col-span-8 space-y-6">
          <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-emerald-50 text-emerald-600 rounded-xl">
                  <Video size={20} />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-slate-800">Live Vision Feed</h2>
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${isCameraConnected ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`} />
                    <span className="text-xs font-medium text-slate-500">
                      {isCameraConnected ? 'Camera Connected' : 'Camera Disconnected'}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 bg-slate-50 p-1.5 rounded-2xl border border-slate-100 w-full sm:w-auto">
                <input
                  type="text"
                  value={cameraUrl}
                  onChange={(e) => setCameraUrl(e.target.value)}
                  placeholder="Camera Stream URL"
                  className="bg-transparent px-3 py-1.5 text-sm outline-none w-full sm:w-64 font-medium text-slate-600"
                />
                <button
                  onClick={handleSetCamera}
                  className="bg-slate-900 text-white px-4 py-1.5 rounded-xl text-xs font-bold hover:bg-slate-800 transition-all"
                >
                  Set
                </button>
              </div>
            </div>

            {/* FIX 5: Video feed is a plain <img> for MJPEG — correct approach.
                The canvas sits on top. Both are positioned absolute inside
                the relative container so they perfectly overlap. */}
            <div ref={containerRef} className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-inner">
              <img
                src={cameraUrl}
                className="absolute inset-0 w-full h-full object-contain"
                onError={() => setIsCameraConnected(false)}
                onLoad={() => setIsCameraConnected(true)}
                alt="Live camera feed"
              />

              {/* Overlay canvas — pointer-events-none so it never blocks the feed */}
              <canvas
                ref={canvasOverlayRef}
                className="absolute inset-0 w-full h-full pointer-events-none z-10"
              />

              {isInferenceRunning && (
                <div className="absolute top-4 right-4 z-20 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
                  <Activity size={14} className="text-emerald-400 animate-pulse" />
                  <span className="text-[10px] font-black text-white uppercase tracking-wider">Inference Active</span>
                </div>
              )}
            </div>

            {/* Inference Controls */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-indigo-50 text-indigo-600 rounded-xl">
                    <Cpu size={18} />
                  </div>
                  <h3 className="font-bold text-slate-800">Inference Engine</h3>
                </div>

                <div className="flex bg-slate-50 p-1 rounded-2xl border border-slate-100">
                  <button
                    onClick={() => handleModeChange('local')}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold transition-all ${inferenceMode === 'local' ? 'bg-white text-indigo-600 shadow-sm border border-slate-200' : 'text-slate-400 hover:text-slate-600'}`}
                  >
                    <Monitor size={14} /> Local Mode
                  </button>
                  <button
                    onClick={() => handleModeChange('cloud')}
                    className={`flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold transition-all ${inferenceMode === 'cloud' ? 'bg-white text-amber-600 shadow-sm border border-slate-200' : 'text-slate-400 hover:text-slate-600'}`}
                  >
                    <Globe size={14} /> Cloud Mode
                  </button>
                </div>

                <button
                  onClick={toggleInference}
                  className={`w-full py-4 rounded-2xl font-black text-sm flex items-center justify-center gap-3 transition-all shadow-lg active:scale-95 ${
                    isInferenceRunning
                      ? 'bg-rose-500 text-white shadow-rose-200 hover:bg-rose-600'
                      : 'bg-emerald-500 text-white shadow-emerald-200 hover:bg-emerald-600'
                  }`}
                >
                  {isInferenceRunning ? <><Square size={18} /> Stop Inference</> : <><Play size={18} /> Start Inference</>}
                </button>
              </div>

              <div className="space-y-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-slate-100 text-slate-600 rounded-xl">
                    <SlidersHorizontal size={18} />
                  </div>
                  <h3 className="font-bold text-slate-800">Parameters</h3>
                </div>

                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-wider">
                      <span>Confidence Threshold</span>
                      <span className="text-indigo-600">{Math.round(thresholds.conf * 100)}%</span>
                    </div>
                    <input
                      type="range" min="0.1" max="1.0" step="0.05"
                      value={thresholds.conf}
                      onChange={(e) => handleThresholdChange({ ...thresholds, conf: parseFloat(e.target.value) })}
                      className="w-full h-1.5 bg-slate-100 rounded-full appearance-none accent-indigo-600 cursor-pointer"
                    />
                  </div>

                  {inferenceMode === 'cloud' && (
                    <div className="space-y-2 animate-in fade-in slide-in-from-top-2">
                      <div className="flex justify-between text-[10px] font-black text-slate-400 uppercase tracking-wider">
                        <span>Cloud Interval</span>
                        <span className="text-amber-600">{cloudInterval}s</span>
                      </div>
                      <input
                        type="range" min="5" max="300" step="5"
                        value={cloudInterval}
                        onChange={(e) => handleCloudIntervalChange(parseInt(e.target.value))}
                        className="w-full h-1.5 bg-slate-100 rounded-full appearance-none accent-amber-500 cursor-pointer"
                      />
                    </div>
                  )}

                  <div className="flex items-center justify-between p-4 bg-slate-50 rounded-2xl border border-slate-100">
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg transition-colors ${trackingEnabled ? 'bg-emerald-100 text-emerald-600' : 'bg-slate-200 text-slate-400'}`}>
                        <Target size={16} />
                      </div>
                      <div>
                        <p className="text-xs font-bold text-slate-800">Object Tracking</p>
                        <p className="text-[10px] text-slate-500">Persistent ID tracking</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={handleResetTracker}
                        className="p-2 text-slate-400 hover:text-slate-600 transition-colors"
                        title="Reset Tracker"
                      >
                        <RefreshCw size={14} />
                      </button>
                      <button
                        onClick={handleTrackingToggle}
                        className={`w-10 h-5 rounded-full relative transition-colors ${trackingEnabled ? 'bg-emerald-500' : 'bg-slate-300'}`}
                      >
                        <div className={`absolute top-1 w-3 h-3 bg-white rounded-full transition-all ${trackingEnabled ? 'left-6' : 'left-1'}`} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <SystemTelemetryPanel />
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                <Layers size={16} className="text-indigo-500" />
                Live Detections
              </h3>
              <span className="text-[10px] font-black bg-slate-100 text-slate-500 px-2 py-1 rounded-lg">
                {detectionsList.length} Active
              </span>
            </div>

            <div className="space-y-3 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
              {detectionsList.length > 0 ? (
                detectionsList.map((det, idx) => (
                  <div
                    key={idx}
                    className="p-4 bg-slate-50 rounded-2xl border border-slate-100 flex items-center justify-between group hover:border-indigo-200 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        (det.class_name || det.class || '').toLowerCase().includes('plastic') ? 'bg-emerald-500' :
                        (det.class_name || det.class || '').toLowerCase().includes('metal')   ? 'bg-indigo-500' : 'bg-amber-500'
                      }`} />
                      <div>
                        <p className="text-sm font-bold text-slate-800 capitalize">{det.class_name || det.class || 'Unknown'}</p>
                        <p className="text-[10px] font-medium text-slate-500">
                          {det.track_id ? `Track #${det.track_id} · ` : ''}
                          Conf: {Math.round(det.confidence * 100)}%
                        </p>
                      </div>
                    </div>
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                      <Tag size={14} className="text-slate-300" />
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-12 space-y-3">
                  <div className="w-12 h-12 bg-slate-50 rounded-full flex items-center justify-center mx-auto">
                    <Activity size={20} className="text-slate-300" />
                  </div>
                  <p className="text-xs font-medium text-slate-400">No objects detected</p>
                </div>
              )}
            </div>

            {lastInferenceTime > 0 && (
              <div className="mt-6 pt-6 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Inference latency</span>
                <span className="text-xs font-bold text-slate-600">{lastInferenceTime.toFixed(1)} ms</span>
              </div>
            )}
          </div>

          <ActuatorStatusPanel />

          <div className="bg-indigo-600 p-6 rounded-3xl text-white shadow-xl shadow-indigo-100 relative overflow-hidden group">
            <div className="relative z-10">
              <div className="flex items-center gap-2 mb-3">
                <Zap size={16} className="text-indigo-200" />
                <h3 className="text-sm font-bold">Inference Optimization</h3>
              </div>
              <p className="text-xs text-indigo-100 leading-relaxed mb-4">
                Local mode uses the on-site Edge TPU for sub-10ms latency. Switch to Cloud mode for complex multi-material classification.
              </p>
              <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-indigo-200">
                <Activity size={12} />
                System Optimized
              </div>
            </div>
            <div className="absolute -right-4 -bottom-4 opacity-10 group-hover:scale-110 transition-transform duration-500">
              <Cpu size={120} />
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default LiveCameraFeed;
