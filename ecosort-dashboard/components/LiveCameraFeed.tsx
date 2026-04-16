import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  RefreshCw, Video, Tag, Globe, Monitor, AlertCircle, Zap, Cpu,
  Play, Square, SlidersHorizontal, Activity, Target, Layers, Radio, Wifi
} from 'lucide-react';
import SystemTelemetryPanel from './SystemTelemetryPanel';
import ActuatorStatusPanel from './ActuatorStatusPanel';
import { InferenceMessage, InferenceDetection } from '../types';

const BASE_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws/inference';

// Video dimensions — must match VIDEO_WIDTH / VIDEO_HEIGHT in main.py
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

// -------------------------------------------------------
// MediaMTX WebRTC configuration
// -------------------------------------------------------
// Set this to your MediaMTX server's WHEP endpoint.
// Example: "http://192.168.1.100:8889/mystream/whep"
const MEDIAMTX_WHEP_URL = 'http://localhost:8889/stream1/whep';
const MEDIAMTX_USER = '';   // fill if auth is enabled
const MEDIAMTX_PASS = '';   // fill if auth is enabled

// Zone definition type (mirrors TRIGGER_ZONES in main.py)
interface TriggerZone {
  id: string;
  label: string;
  actuator_id: string;
  x_position: number;   // in video pixel space (0–VIDEO_WIDTH)
  color: string;
  cooldown_frames: number;
}

interface ZoneEvent {
  zone_id: string;
  zone_label: string;
  actuator_id: string;
  class_name: string;
  track_id: number | null;
  x_position: number;
  timestamp: number;
}

interface ZoneFlash {
  zone_id: string;
  expires_at: number;   // performance.now() ms
}

// -------------------------------------------------------
// Minimal MediaMTX WebRTC reader (no external reader.js needed)
// Uses the WHEP (WebRTC-HTTP Egress Protocol) standard directly.
// -------------------------------------------------------
class MediaMTXWebRTCReader {
  private pc: RTCPeerConnection | null = null;
  private url: string;
  private user: string;
  private pass: string;
  private onTrack: (evt: RTCTrackEvent) => void;
  private onError: (err: Error) => void;
  private closed = false;

  constructor(opts: {
    url: string;
    user?: string;
    pass?: string;
    onTrack: (evt: RTCTrackEvent) => void;
    onError: (err: Error) => void;
  }) {
    this.url = opts.url;
    this.user = opts.user ?? '';
    this.pass = opts.pass ?? '';
    this.onTrack = opts.onTrack;
    this.onError = opts.onError;
    this.start();
  }

  private async start() {
    try {
      this.pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      });

      this.pc.ontrack = (evt) => {
        if (!this.closed) this.onTrack(evt);
      };

      // Add receive-only transceivers so the offer contains the right direction
      this.pc.addTransceiver('video', { direction: 'recvonly' });
      this.pc.addTransceiver('audio', { direction: 'recvonly' });

      const offer = await this.pc.createOffer();
      await this.pc.setLocalDescription(offer);

      // Wait for ICE gathering to finish
      await new Promise<void>((resolve) => {
        if (this.pc!.iceGatheringState === 'complete') { resolve(); return; }
        const check = () => {
          if (this.pc!.iceGatheringState === 'complete') {
            this.pc!.removeEventListener('icegatheringstatechange', check);
            resolve();
          }
        };
        this.pc!.addEventListener('icegatheringstatechange', check);
        // Fallback timeout
        setTimeout(resolve, 3000);
      });

      const headers: Record<string, string> = { 'Content-Type': 'application/sdp' };
      if (this.user && this.pass) {
        headers['Authorization'] = 'Basic ' + btoa(`${this.user}:${this.pass}`);
      }

      const res = await fetch(this.url, {
        method: 'POST',
        headers,
        body: this.pc.localDescription!.sdp,
      });

      if (!res.ok) throw new Error(`WHEP response ${res.status}: ${await res.text()}`);

      const answerSdp = await res.text();
      await this.pc.setRemoteDescription({ type: 'answer', sdp: answerSdp });
    } catch (err) {
      if (!this.closed) this.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }

  close() {
    this.closed = true;
    this.pc?.close();
    this.pc = null;
  }
}

// -------------------------------------------------------
// Feed source type
// -------------------------------------------------------
type FeedSource = 'mjpeg' | 'webrtc';

const LiveCameraFeed: React.FC = () => {
  const [cameraUrl, setCameraUrl] = useState('http://192.168.100.103:9000/video');
  const [isCameraConnected, setIsCameraConnected] = useState(false);
  const [isInferenceRunning, setIsInferenceRunning] = useState(false);
  const [inferenceMode, setInferenceMode] = useState<'local' | 'cloud'>('local');
  const [thresholds, setThresholds] = useState({ conf: 0.5, iou: 0.45 });
  const [trackingEnabled, setTrackingEnabled] = useState(false);
  const [cloudInterval, setCloudInterval] = useState(60);
  const [lastInferenceTime, setLastInferenceTime] = useState<number>(0);

  // ── Feed source toggle ────────────────────────────────
  const [feedSource, setFeedSource] = useState<FeedSource>('mjpeg');
  const [webrtcStatus, setWebrtcStatus] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const [webrtcError, setWebrtcError] = useState<string | null>(null);
  const webrtcReaderRef = useRef<MediaMTXWebRTCReader | null>(null);
  const webrtcVideoRef = useRef<HTMLVideoElement>(null);

  // Detections stored in ref — avoids re-render thrash on every WS message
  const detectionsRef = useRef<InferenceDetection[]>([]);
  const [detectionsList, setDetectionsList] = useState<InferenceDetection[]>([]);

  // Zone definitions received from backend on WS connect
  const zonesRef = useRef<TriggerZone[]>([]);
  const [zonesList, setZonesList] = useState<TriggerZone[]>([]);

  // Active zone flashes — stored in ref so rAF loop reads latest without re-render
  const zoneFlashesRef = useRef<ZoneFlash[]>([]);

  // Zone event log for the sidebar
  const [zoneLog, setZoneLog] = useState<ZoneEvent[]>([]);

  const canvasOverlayRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const inferenceModeRef = useRef<'local' | 'cloud'>('local');

  useEffect(() => {
    inferenceModeRef.current = inferenceMode;
  }, [inferenceMode]);

  // -------------------------------------------------------
  // WebRTC feed management
  // -------------------------------------------------------
  const startWebRTC = useCallback(() => {
    // Clean up any existing reader
    if (webrtcReaderRef.current) {
      webrtcReaderRef.current.close();
      webrtcReaderRef.current = null;
    }

    setWebrtcStatus('connecting');
    setWebrtcError(null);

    webrtcReaderRef.current = new MediaMTXWebRTCReader({
      url: MEDIAMTX_WHEP_URL,
      user: MEDIAMTX_USER,
      pass: MEDIAMTX_PASS,
      onTrack: (evt) => {
        if (webrtcVideoRef.current && evt.streams[0]) {
          webrtcVideoRef.current.srcObject = evt.streams[0];
          setWebrtcStatus('connected');
          setIsCameraConnected(true);
        }
      },
      onError: (err) => {
        console.error('[webrtc]', err);
        setWebrtcStatus('error');
        setWebrtcError(err.message);
        setIsCameraConnected(false);
      },
    });
  }, []);

  const stopWebRTC = useCallback(() => {
    if (webrtcReaderRef.current) {
      webrtcReaderRef.current.close();
      webrtcReaderRef.current = null;
    }
    if (webrtcVideoRef.current) {
      webrtcVideoRef.current.srcObject = null;
    }
    setWebrtcStatus('idle');
  }, []);

  // Switch feed source
  useEffect(() => {
    if (feedSource === 'webrtc') {
      startWebRTC();
    } else {
      stopWebRTC();
    }
    return () => {
      if (feedSource === 'webrtc') stopWebRTC();
    };
  }, [feedSource, startWebRTC, stopWebRTC]);

  // -------------------------------------------------------
  // WebSocket Connection
  // -------------------------------------------------------
  useEffect(() => {
    const connectWS = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => console.log('[ws] Inference WebSocket connected');

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === 'zone_config') {
            zonesRef.current = data.zones || [];
            setZonesList(data.zones || []);

          } else if (data.type === 'camera_status') {
            // Only update camera status from WS when using MJPEG feed
            if (feedSource === 'mjpeg') {
              setIsCameraConnected(!!data.connected);
            }

          } else if (data.type === 'detection' || data.type === 'tracked') {
            const objs: InferenceDetection[] = data.objects || data.detections || [];
            detectionsRef.current = objs;
            setDetectionsList(objs);

            if (data.inference_time_ms) {
              setLastInferenceTime(data.inference_time_ms);
            }

            if (data.zone_events && data.zone_events.length > 0) {
              const now = performance.now();
              const newFlashes: ZoneFlash[] = data.zone_events.map((e: ZoneEvent) => ({
                zone_id: e.zone_id,
                expires_at: now + 600,
              }));

              zoneFlashesRef.current = [
                ...zoneFlashesRef.current.filter(
                  f => !newFlashes.some(nf => nf.zone_id === f.zone_id)
                ),
                ...newFlashes,
              ];

              setZoneLog(prev => [...data.zone_events, ...prev].slice(0, 20));
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
  }, []); // feedSource intentionally excluded — we only gate isCameraConnected update above

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
        if (camRes.ok) { const d = await camRes.json(); if (feedSource === 'mjpeg') setIsCameraConnected(d.connected); }
        if (modeRes.ok) { const d = await modeRes.json(); setInferenceMode(d.mode); }
        if (threshRes.ok) { const d = await threshRes.json(); setThresholds(d); }
      } catch (e) {
        console.error('[init] Failed to fetch status', e);
      }
    };
    fetchStatus();
  }, []);

  // -------------------------------------------------------
  // Draw loop — zones + detections on the canvas overlay
  // -------------------------------------------------------
  const drawFrame = useCallback(() => {
    const canvas = canvasOverlayRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const rect = container.getBoundingClientRect();
    if (canvas.width !== rect.width || canvas.height !== rect.height) {
      canvas.width = rect.width;
      canvas.height = rect.height;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / VIDEO_WIDTH;
    const scaleY = canvas.height / VIDEO_HEIGHT;

    const now = performance.now();

    zoneFlashesRef.current = zoneFlashesRef.current.filter(f => f.expires_at > now);
    const activeFlashIds = new Set(zoneFlashesRef.current.map(f => f.zone_id));

    // ── DRAW TRIGGER ZONE LINES ──────────────────────────────────
    zonesRef.current.forEach((zone) => {
      const cx = zone.x_position * scaleX;
      const isFlashing = activeFlashIds.has(zone.id);

      ctx.save();

      if (isFlashing) {
        ctx.shadowColor = zone.color;
        ctx.shadowBlur = 12;
        ctx.strokeStyle = zone.color;
        ctx.lineWidth = 3;
        ctx.globalAlpha = 1.0;
      } else {
        ctx.strokeStyle = zone.color;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([6, 4]);
        ctx.globalAlpha = 0.7;
      }

      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, canvas.height);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.globalAlpha = 1.0;
      ctx.shadowBlur = 0;
      ctx.font = 'bold 11px Inter, sans-serif';
      const textWidth = ctx.measureText(zone.label).width;
      const pillW = textWidth + 12;
      const pillH = 20;
      const pillX = cx - pillW / 2;

      ctx.fillStyle = isFlashing ? zone.color : zone.color + 'cc';
      ctx.beginPath();
      ctx.roundRect(pillX, 6, pillW, pillH, 4);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.fillText(zone.label, pillX + 6, 20);

      ctx.font = '9px Inter, sans-serif';
      ctx.fillStyle = zone.color + 'aa';
      const actLabel = zone.actuator_id;
      const actW = ctx.measureText(actLabel).width;
      ctx.fillText(actLabel, cx - actW / 2, 34);

      ctx.restore();
    });

    // ── DRAW DETECTIONS ─────────────────────────────────────────
    const detections = detectionsRef.current;

    detections.forEach((det) => {
      let x: number, y: number, w: number, h: number;

      if (inferenceModeRef.current === 'local') {
        x = det.bbox[0] * scaleX;
        y = det.bbox[1] * scaleY;
        w = (det.bbox[2] - det.bbox[0]) * scaleX;
        h = (det.bbox[3] - det.bbox[1]) * scaleY;
      } else {
        const cx = det.bbox[0] * VIDEO_WIDTH;
        const cy = det.bbox[1] * VIDEO_HEIGHT;
        const bw = det.bbox[2] * VIDEO_WIDTH;
        const bh = det.bbox[3] * VIDEO_HEIGHT;
        x = (cx - bw / 2) * scaleX;
        y = (cy - bh / 2) * scaleY;
        w = bw * scaleX;
        h = bh * scaleY;
      }

      const className = det.class_name || det.class || 'Unknown';
      const color =
        className.toLowerCase().includes('plastic') ? '#10b981' :
        className.toLowerCase().includes('metal')   ? '#3b82f6' : '#f59e0b';

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${className} ${(det.confidence * 100).toFixed(0)}%${det.track_id != null ? ` #${det.track_id}` : ''}`;
      ctx.font = 'bold 12px Inter, sans-serif';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x, y - 20, textWidth + 10, 20);

      ctx.fillStyle = 'white';
      ctx.fillText(label, x + 5, y - 5);

      const leadingX = (det.bbox[2] ?? det.bbox[0]) * scaleX;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(leadingX, y + h / 2, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }, []);

  // Continuous rAF loop
  useEffect(() => {
    let animId: number;
    const loop = () => { drawFrame(); animId = requestAnimationFrame(loop); };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [drawFrame]);

  // -------------------------------------------------------
  // API Control Handlers
  // -------------------------------------------------------
  const handleSetCamera = async () => {
    try {
      await fetch(`${BASE_URL}/inference/set_camera?url=${encodeURIComponent(cameraUrl)}`, { method: 'POST' });
    } catch (e) { console.error('Failed to set camera', e); }
  };

  const toggleInference = async () => {
    const endpoint = isInferenceRunning ? 'stop' : 'start';
    try {
      const res = await fetch(`${BASE_URL}/inference/${endpoint}`, { method: 'POST' });
      if (res.ok) setIsInferenceRunning(!isInferenceRunning);
    } catch (e) { console.error(`Failed to ${endpoint} inference`, e); }
  };

  const handleModeChange = async (mode: 'local' | 'cloud') => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_mode/${mode}`, { method: 'POST' });
      if (res.ok) setInferenceMode(mode);
    } catch (e) { console.error('Failed to set mode', e); }
  };

  const handleThresholdChange = async (newThresholds: typeof thresholds) => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_thresholds`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newThresholds)
      });
      if (res.ok) setThresholds(newThresholds);
    } catch (e) { console.error('Failed to set thresholds', e); }
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
    } catch (e) { console.error('Failed to toggle tracking', e); }
  };

  const handleResetTracker = async () => {
    try {
      await fetch(`${BASE_URL}/inference/tracking/reset`, { method: 'POST' });
    } catch (e) { console.error('Failed to reset tracker', e); }
  };

  const handleCloudIntervalChange = async (seconds: number) => {
    try {
      const res = await fetch(`${BASE_URL}/inference/set_cloud_interval`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds })
      });
      if (res.ok) setCloudInterval(seconds);
    } catch (e) { console.error('Failed to set cloud interval', e); }
  };

  const handleFeedSourceToggle = () => {
    setFeedSource(prev => prev === 'mjpeg' ? 'webrtc' : 'mjpeg');
  };

  const handleRetryWebRTC = () => {
    if (feedSource === 'webrtc') startWebRTC();
  };

  const formatTime = (ts: number) => {
    return new Date(ts * 1000).toLocaleTimeString('en', { hour12: false });
  };

  // -------------------------------------------------------
  // Derived state
  // -------------------------------------------------------
  const isConnected = feedSource === 'webrtc'
    ? webrtcStatus === 'connected'
    : isCameraConnected;

  const connectionLabel = feedSource === 'webrtc'
    ? webrtcStatus === 'connected' ? 'WebRTC Connected'
      : webrtcStatus === 'connecting' ? 'WebRTC Connecting…'
      : webrtcStatus === 'error' ? 'WebRTC Error'
      : 'WebRTC Idle'
    : isCameraConnected ? 'Camera Connected' : 'Camera Disconnected';

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
                    <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500 animate-pulse' : webrtcStatus === 'connecting' ? 'bg-amber-400 animate-pulse' : 'bg-slate-300'}`} />
                    <span className="text-xs font-medium text-slate-500">{connectionLabel}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                {/* Feed source toggle */}
                <div className="flex items-center bg-slate-50 p-1 rounded-2xl border border-slate-100">
                  <button
                    onClick={() => setFeedSource('mjpeg')}
                    title="MJPEG / HTTP stream"
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${feedSource === 'mjpeg' ? 'bg-white text-slate-800 shadow-sm border border-slate-200' : 'text-slate-400 hover:text-slate-600'}`}
                  >
                    <Monitor size={13} /> MJPEG
                  </button>
                  <button
                    onClick={() => setFeedSource('webrtc')}
                    title="MediaMTX WebRTC (WHEP)"
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all ${feedSource === 'webrtc' ? 'bg-white text-indigo-600 shadow-sm border border-slate-200' : 'text-slate-400 hover:text-slate-600'}`}
                  >
                    <Radio size={13} /> WebRTC
                  </button>
                </div>

                {/* Camera URL input — only shown in MJPEG mode */}
                {feedSource === 'mjpeg' && (
                  <div className="flex items-center gap-2 bg-slate-50 p-1.5 rounded-2xl border border-slate-100">
                    <input
                      type="text"
                      value={cameraUrl}
                      onChange={(e) => setCameraUrl(e.target.value)}
                      placeholder="Camera Stream URL"
                      className="bg-transparent px-3 py-1.5 text-sm outline-none w-full sm:w-56 font-medium text-slate-600"
                    />
                    <button
                      onClick={handleSetCamera}
                      className="bg-slate-900 text-white px-4 py-1.5 rounded-xl text-xs font-bold hover:bg-slate-800 transition-all"
                    >
                      Set
                    </button>
                  </div>
                )}

                {/* WebRTC retry / status — only shown in WebRTC mode */}
                {feedSource === 'webrtc' && webrtcStatus === 'error' && (
                  <button
                    onClick={handleRetryWebRTC}
                    className="flex items-center gap-1.5 bg-rose-50 text-rose-600 border border-rose-200 px-3 py-1.5 rounded-xl text-xs font-bold hover:bg-rose-100 transition-all"
                  >
                    <RefreshCw size={13} /> Retry
                  </button>
                )}

                {feedSource === 'webrtc' && webrtcStatus === 'connecting' && (
                  <div className="flex items-center gap-1.5 text-amber-600 text-xs font-bold">
                    <Wifi size={13} className="animate-pulse" /> Connecting…
                  </div>
                )}
              </div>
            </div>

            {/* Video + Canvas overlay */}
            <div ref={containerRef} className="relative aspect-video bg-slate-900 rounded-2xl overflow-hidden border border-slate-800 shadow-inner">

              {/* ── MJPEG feed (img tag) ── */}
              {feedSource === 'mjpeg' && (
                <img
                  src={cameraUrl}
                  className="absolute inset-0 w-full h-full object-contain"
                  onError={() => setIsCameraConnected(false)}
                  onLoad={() => setIsCameraConnected(true)}
                  alt="Live camera feed"
                />
              )}

              {/* ── WebRTC feed (video tag) ── */}
              {feedSource === 'webrtc' && (
                <video
                  ref={webrtcVideoRef}
                  className="absolute inset-0 w-full h-full object-contain"
                  autoPlay
                  muted
                  playsInline
                />
              )}

              {/* WebRTC error / connecting overlay */}
              {feedSource === 'webrtc' && webrtcStatus !== 'connected' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-900/80 gap-3">
                  {webrtcStatus === 'connecting' && (
                    <>
                      <Wifi size={32} className="text-amber-400 animate-pulse" />
                      <p className="text-white text-sm font-semibold">Connecting to MediaMTX…</p>
                      <p className="text-slate-400 text-xs">{MEDIAMTX_WHEP_URL}</p>
                    </>
                  )}
                  {webrtcStatus === 'error' && (
                    <>
                      <AlertCircle size={32} className="text-rose-400" />
                      <p className="text-white text-sm font-semibold">WebRTC Connection Failed</p>
                      <p className="text-slate-400 text-xs max-w-xs text-center">{webrtcError}</p>
                      <button
                        onClick={handleRetryWebRTC}
                        className="mt-2 flex items-center gap-2 bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-xl text-xs font-bold transition-all border border-white/20"
                      >
                        <RefreshCw size={13} /> Retry Connection
                      </button>
                    </>
                  )}
                  {webrtcStatus === 'idle' && (
                    <>
                      <Radio size={32} className="text-slate-500" />
                      <p className="text-slate-400 text-sm">WebRTC feed idle</p>
                    </>
                  )}
                </div>
              )}

              {/* Canvas overlay for zones + detections — always on top */}
              <canvas
                ref={canvasOverlayRef}
                className="absolute inset-0 w-full h-full pointer-events-none z-20"
              />

              {isInferenceRunning && (
                <div className="absolute top-4 right-4 z-30 flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10">
                  <Activity size={14} className="text-emerald-400 animate-pulse" />
                  <span className="text-[10px] font-black text-white uppercase tracking-wider">Inference Active</span>
                </div>
              )}

              {/* Feed source badge */}
              <div className="absolute top-4 left-4 z-30">
                {feedSource === 'webrtc' ? (
                  <div className="flex items-center gap-1.5 bg-indigo-600/80 backdrop-blur-sm px-2.5 py-1 rounded-full border border-indigo-400/30">
                    <Radio size={10} className="text-indigo-200" />
                    <span className="text-[10px] font-black text-white uppercase tracking-wider">WebRTC</span>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 bg-slate-900/60 backdrop-blur-sm px-2.5 py-1 rounded-full border border-white/10">
                    <Monitor size={10} className="text-slate-300" />
                    <span className="text-[10px] font-black text-slate-200 uppercase tracking-wider">MJPEG</span>
                  </div>
                )}
              </div>

              {/* Zone legend overlay — bottom left */}
              {zonesList.length > 0 && (
                <div className="absolute bottom-4 left-4 z-30 flex flex-col gap-1">
                  {zonesList.map(zone => (
                    <div key={zone.id} className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-2 py-1 rounded-lg">
                      <div className="w-2.5 h-2.5 rounded-sm" style={{ background: zone.color }} />
                      <span className="text-[10px] font-bold text-white">{zone.label}</span>
                      <span className="text-[9px] text-white/50">{zone.actuator_id}</span>
                    </div>
                  ))}
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
                    <div className="space-y-2">
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

          {/* Live Detections */}
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

            <div className="space-y-3 max-h-[240px] overflow-y-auto pr-2">
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
                          {det.track_id != null ? `Track #${det.track_id} · ` : ''}
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
                <div className="text-center py-8 space-y-2">
                  <div className="w-10 h-10 bg-slate-50 rounded-full flex items-center justify-center mx-auto">
                    <Activity size={18} className="text-slate-300" />
                  </div>
                  <p className="text-xs font-medium text-slate-400">No objects detected</p>
                </div>
              )}
            </div>

            {lastInferenceTime > 0 && (
              <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider">Inference latency</span>
                <span className="text-xs font-bold text-slate-600">{lastInferenceTime.toFixed(1)} ms</span>
              </div>
            )}
          </div>

          {/* Zone Trigger Log */}
          <div className="bg-white p-6 rounded-3xl shadow-sm border border-slate-100">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                <Zap size={16} className="text-amber-500" />
                Zone Trigger Log
              </h3>
              <button
                onClick={() => setZoneLog([])}
                className="text-[10px] font-bold text-slate-400 hover:text-slate-600 transition-colors"
              >
                Clear
              </button>
            </div>

            <div className="space-y-2 max-h-[280px] overflow-y-auto pr-1">
              {zoneLog.length > 0 ? (
                zoneLog.map((event, idx) => {
                  const zone = zonesList.find(z => z.id === event.zone_id);
                  return (
                    <div key={idx} className="flex items-start gap-3 p-3 bg-slate-50 rounded-xl border border-slate-100">
                      <div
                        className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
                        style={{ background: zone?.color ?? '#888' }}
                      />
                      <div className="flex-1 min-w-0">
                        <p className="text-xs font-bold text-slate-800 truncate">
                          {event.class_name}
                          {event.track_id != null && (
                            <span className="font-normal text-slate-400"> #{event.track_id}</span>
                          )}
                        </p>
                        <p className="text-[10px] text-slate-500">
                          {event.zone_label} · {event.actuator_id}
                        </p>
                      </div>
                      <span className="text-[9px] text-slate-400 flex-shrink-0 mt-0.5">
                        {formatTime(event.timestamp)}
                      </span>
                    </div>
                  );
                })
              ) : (
                <div className="text-center py-8">
                  <p className="text-xs text-slate-400">No triggers yet</p>
                </div>
              )}
            </div>
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