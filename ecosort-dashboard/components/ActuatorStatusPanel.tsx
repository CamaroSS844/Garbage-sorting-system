import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Wind, RotateCw, MoveUp, Zap, Wifi, WifiOff, AlertCircle, Activity } from 'lucide-react';

// ─── Inject styles ────────────────────────────────────────────────────────────
const injectStyles = () => {
  if (document.getElementById('actuator-styles')) return;
  const s = document.createElement('style');
  s.id = 'actuator-styles';
  s.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

    .act-root {
      font-family: 'DM Sans', sans-serif;
      --c-bg:       #0d1117;
      --c-surface:  #161b22;
      --c-surface2: #1c2230;
      --c-border:   #30363d;
      --c-text:     #e6edf3;
      --c-muted:    #8b949e;
      --c-green:    #3fb950;
      --c-green-d:  #1a4226;
      --c-amber:    #d29922;
      --c-amber-d:  #3d2d00;
      --c-blue:     #58a6ff;
      --c-blue-d:   #0d2340;
      --c-red:      #f85149;
      --c-red-d:    #3d1210;
      --c-indigo:   #a371f7;
      --c-indigo-d: #271459;
      background: var(--c-bg);
      border-radius: 18px;
      padding: 24px;
      border: 1px solid var(--c-border);
      color: var(--c-text);
      min-width: 320px;
    }

    .act-mono { font-family: 'IBM Plex Mono', monospace; }
    .act-head { font-family: 'Syne', sans-serif; }

    /* Header */
    .act-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 22px; padding-bottom: 16px;
      border-bottom: 1px solid var(--c-border);
    }
    .act-title {
      font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700;
      letter-spacing: .12em; text-transform: uppercase; color: var(--c-text);
      display: flex; align-items: center; gap: 8px;
    }
    .act-dot {
      width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
    }
    .act-dot.live  { background: var(--c-green); box-shadow: 0 0 6px var(--c-green); animation: blink 1.8s ease-in-out infinite; }
    .act-dot.stale { background: var(--c-amber); }
    .act-dot.off   { background: var(--c-muted); }
    @keyframes blink {
      0%,100% { opacity: 1; } 50% { opacity: .35; }
    }

    /* Poll badge */
    .act-badge {
      font-family: 'IBM Plex Mono', monospace; font-size: 10px; font-weight: 500;
      padding: 3px 8px; border-radius: 6px; background: var(--c-surface2);
      border: 1px solid var(--c-border); color: var(--c-muted);
    }

    /* Section card */
    .act-card {
      background: var(--c-surface); border: 1px solid var(--c-border);
      border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;
    }
    .act-card:last-child { margin-bottom: 0; }

    /* Vacuum card states */
    .act-card.vac-on  { border-color: #2d5a3a; background: #0f1e14; }
    .act-card.vac-off { border-color: var(--c-border); }

    /* Gauge row */
    .act-gauge-row {
      display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
    }
    .act-gauge-label {
      font-size: 10px; font-weight: 600; letter-spacing: .06em;
      text-transform: uppercase; color: var(--c-muted); flex: 1;
    }
    .act-gauge-val {
      font-family: 'IBM Plex Mono', monospace; font-size: 13px; font-weight: 600;
      min-width: 52px; text-align: right;
    }

    /* Track */
    .act-track {
      width: 100%; height: 5px; background: var(--c-surface2);
      border-radius: 99px; overflow: hidden; margin-bottom: 14px;
      border: 1px solid var(--c-border);
    }
    .act-fill {
      height: 100%; border-radius: 99px;
      transition: width .18s ease;
    }

    /* Arc SVG */
    .act-arc-wrap {
      display: flex; justify-content: center; margin: 6px 0 10px;
    }

    /* Action badge */
    .act-action-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 10px 14px; border-radius: 10px;
      background: var(--c-surface2); border: 1px solid var(--c-border);
      margin-top: 10px;
    }
    .act-action-label { font-size: 11px; font-weight: 600; color: var(--c-muted); letter-spacing: .05em; text-transform: uppercase; }
    .act-action-pill {
      font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
      padding: 3px 10px; border-radius: 6px; text-transform: uppercase;
    }
    .act-action-pill.none    { background: var(--c-surface); color: var(--c-muted); border: 1px solid var(--c-border); }
    .act-action-pill.plastic { background: #0d2340; color: #58a6ff; border: 1px solid #1a3a60; }
    .act-action-pill.paper   { background: #3d2d00; color: #d29922; border: 1px solid #5a4200; }
    .act-action-pill.active  { animation: flash .35s ease-out; }
    @keyframes flash {
      0% { filter: brightness(2.5); } 100% { filter: brightness(1); }
    }

    /* Device list */
    .act-device-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 0; border-bottom: 1px solid var(--c-border);
      font-size: 12px;
    }
    .act-device-row:last-child { border-bottom: none; padding-bottom: 0; }
    .act-device-id { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--c-text); }
    .act-device-status {
      font-size: 10px; font-weight: 700; letter-spacing: .07em; text-transform: uppercase;
      padding: 2px 8px; border-radius: 5px;
    }
    .act-device-status.online  { background: var(--c-green-d); color: var(--c-green); }
    .act-device-status.offline { background: var(--c-surface2); color: var(--c-muted); }

    /* Error banner */
    .act-error {
      display: flex; align-items: center; gap: 8px;
      background: var(--c-red-d); border: 1px solid #6b1c1c; border-radius: 10px;
      padding: 10px 14px; font-size: 12px; color: var(--c-red); margin-bottom: 10px;
    }

    /* Empty state */
    .act-empty {
      text-align: center; padding: 18px 0;
      font-size: 12px; color: var(--c-muted);
    }

    /* Conveyor bar */
    .act-conv-row { margin-top: 4px; }
    .act-conv-track {
      width: 100%; height: 8px; background: var(--c-surface2);
      border-radius: 99px; overflow: hidden; border: 1px solid var(--c-border);
      position: relative;
    }
    .act-conv-fill {
      height: 100%; border-radius: 99px;
      background: linear-gradient(90deg, #1a4226, var(--c-green));
      transition: width .22s ease;
    }
    /* moving stripes overlay for conveyor */
    .act-conv-stripes {
      position: absolute; inset: 0;
      background: repeating-linear-gradient(
        -45deg,
        transparent, transparent 4px,
        rgba(255,255,255,.05) 4px, rgba(255,255,255,.05) 8px
      );
      animation: conveyor-move 1.2s linear infinite;
    }
    @keyframes conveyor-move {
      from { background-position: 0 0; }
      to   { background-position: 16px 0; }
    }
  `;
  document.head.appendChild(s);
};

// ─── Arc gauge SVG ────────────────────────────────────────────────────────────
const ArcGauge = ({ value, max, color, label, unit = '°', size = 110 }) => {
  const r = 38;
  const cx = size / 2, cy = size / 2 + 6;
  const startAngle = -210, sweepAngle = 240;
  const pct = Math.min(Math.max(value / max, 0), 1);
  const toRad = deg => (deg * Math.PI) / 180;
  const arcPoint = deg => ({
    x: cx + r * Math.cos(toRad(deg)),
    y: cy + r * Math.sin(toRad(deg)),
  });
  const angStart = startAngle;
  const angEnd = startAngle + sweepAngle;
  const angCurrent = startAngle + pct * sweepAngle;
  const p1 = arcPoint(angStart), p2 = arcPoint(angEnd), pc = arcPoint(angCurrent);
  const largeArc0 = sweepAngle > 180 ? 1 : 0;
  const largeArcC = pct * sweepAngle > 180 ? 1 : 0;
  return (
    <svg width={size} height={size * 0.82} viewBox={`0 0 ${size} ${size * 0.82}`} style={{ overflow: 'visible' }}>
      {/* Track */}
      <path
        d={`M${p1.x},${p1.y} A${r},${r} 0 ${largeArc0},1 ${p2.x},${p2.y}`}
        fill="none" stroke="#30363d" strokeWidth="5" strokeLinecap="round"
      />
      {/* Fill */}
      <path
        d={`M${p1.x},${p1.y} A${r},${r} 0 ${largeArcC},1 ${pc.x},${pc.y}`}
        fill="none" stroke={color} strokeWidth="5" strokeLinecap="round"
        style={{ filter: `drop-shadow(0 0 4px ${color}66)` }}
      />
      {/* Needle dot */}
      <circle cx={pc.x} cy={pc.y} r="4.5" fill={color} style={{ filter: `drop-shadow(0 0 6px ${color})` }} />
      {/* Value text */}
      <text x={cx} y={cy + 4} textAnchor="middle" dominantBaseline="middle"
        fill="#e6edf3" fontSize="16" fontFamily="IBM Plex Mono, monospace" fontWeight="600">
        {value.toFixed(1)}{unit}
      </text>
      <text x={cx} y={cy + 20} textAnchor="middle" dominantBaseline="middle"
        fill="#8b949e" fontSize="8.5" fontFamily="DM Sans, sans-serif" fontWeight="600"
        letterSpacing="0.08em" style={{ textTransform: 'uppercase' }}>
        {label}
      </text>
    </svg>
  );
};

// ─── Main component ───────────────────────────────────────────────────────────
const POLL_MS = 300; // poll /test/control every 300 ms
const DEVICE_POLL_MS = 5000;

const ActuatorStatusPanel = () => {
  useEffect(() => { injectStyles(); }, []);

  const [control, setControl] = useState({
    arm: { azimuth: 0, elevation: 0 },
    conveyor: 0,
    vacuum: false,
    action: 'none',
  });
  const [devices, setDevices] = useState({});
  const [lastPoll, setLastPoll] = useState(null);
  const [error, setError] = useState(null);
  const [actionFlash, setActionFlash] = useState('');
  const prevActionRef = useRef('none');

  // ── Poll /test/control ──────────────────────────────────────
  const pollControl = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/test/control');
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setControl(data);
      setLastPoll(new Date());
      setError(null);

      // Flash when a new non-none action fires
      if (data.action && data.action !== 'none' && data.action !== prevActionRef.current) {
        setActionFlash(data.action);
        setTimeout(() => setActionFlash(''), 600);
      }
      prevActionRef.current = data.action ?? 'none';
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // ── Poll /esp32/status ──────────────────────────────────────
  const pollDevices = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8000/esp32/status');
      if (!res.ok) return;
      setDevices(await res.json());
    } catch (_) {}
  }, []);

  useEffect(() => {
    pollControl();
    pollDevices();
    const t1 = setInterval(pollControl, POLL_MS);
    const t2 = setInterval(pollDevices, DEVICE_POLL_MS);
    return () => { clearInterval(t1); clearInterval(t2); };
  }, [pollControl, pollDevices]);

  // ── Derived ─────────────────────────────────────────────────
  const azimuth   = (control.arm?.azimuth   ?? 0) * 90 + 90; // -1→1 → 0→180 deg display
  const elevation = (control.arm?.elevation ?? 0) * 45 + 45; // -1→1 → 0→90 deg display
  const conveyor  = Math.max(0, control.conveyor ?? 0);       // 0–1
  const vacuumOn  = !!control.vacuum;
  const action    = control.action ?? 'none';
  const isLive    = lastPoll && (Date.now() - lastPoll.getTime()) < 2000;
  const pollAge   = lastPoll ? `${Math.round((Date.now() - lastPoll.getTime()) / 1000)}s ago` : '—';
  const deviceEntries = Object.entries(devices);

  return (
    <div className="act-root">
      {/* ── Header ── */}
      <div className="act-header">
        <div className="act-title">
          <div className={`act-dot ${error ? 'off' : isLive ? 'live' : 'stale'}`} />
          Actuator Activity
        </div>
        <span className="act-badge">
          {error ? 'ERR' : isLive ? `↻ ${pollAge}` : pollAge}
        </span>
      </div>

      {/* ── Error banner ── */}
      {error && (
        <div className="act-error">
          <AlertCircle size={14} />
          Backend unreachable — {error}
        </div>
      )}

      {/* ── Vacuum ── */}
      <div className={`act-card ${vacuumOn ? 'vac-on' : 'vac-off'}`}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: vacuumOn ? '#1a4226' : '#1c2230',
              border: `1px solid ${vacuumOn ? '#2d5a3a' : '#30363d'}`,
            }}>
              <Wind size={16} style={{ color: vacuumOn ? '#3fb950' : '#8b949e' }} />
            </div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: '#e6edf3', marginBottom: 2 }}>Vacuum Pump</div>
              <div style={{ fontSize: 10, color: '#8b949e', fontFamily: 'IBM Plex Mono, monospace' }}>
                {vacuumOn ? 'ACTIVE SUCTION' : 'IDLE'}
              </div>
            </div>
          </div>
          <div style={{
            fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, fontWeight: 700,
            padding: '4px 10px', borderRadius: 7,
            background: vacuumOn ? '#1a4226' : '#1c2230',
            border: `1px solid ${vacuumOn ? '#2d5a3a' : '#30363d'}`,
            color: vacuumOn ? '#3fb950' : '#8b949e',
            letterSpacing: '.06em',
          }}>
            {vacuumOn ? '● ON' : '○ OFF'}
          </div>
        </div>
      </div>

      {/* ── Conveyor ── */}
      <div className="act-card">
        <div className="act-gauge-row">
          <Activity size={13} style={{ color: '#3fb950', flexShrink: 0 }} />
          <span className="act-gauge-label" style={{ color: '#e6edf3' }}>Conveyor Belt</span>
          <span className="act-gauge-val act-mono" style={{ color: '#3fb950' }}>
            {(conveyor * 100).toFixed(0)}%
          </span>
        </div>
        <div className="act-conv-track">
          {conveyor > 0.01 && <div className="act-conv-stripes" />}
          <div className="act-conv-fill" style={{ width: `${conveyor * 100}%` }} />
        </div>
        <div style={{ fontSize: 10, color: '#8b949e', fontFamily: 'IBM Plex Mono, monospace' }}>
          {conveyor < 0.01 ? 'STOPPED' : conveyor < 0.5 ? 'SLOW SPEED' : conveyor < 0.85 ? 'MID SPEED' : 'FULL SPEED'}
        </div>
      </div>

      {/* ── Arm position — dual arc gauges ── */}
      <div className="act-card">
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: '.09em', textTransform: 'uppercase', color: '#8b949e', marginBottom: 12 }}>
          Sorting Arm Position
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'flex-end', gap: 8 }}>
          <div style={{ textAlign: 'center' }}>
            <ArcGauge value={azimuth} max={180} color="#a371f7" label="Azimuth" size={110} />
          </div>
          <div style={{ width: 1, height: 60, background: '#30363d', alignSelf: 'center' }} />
          <div style={{ textAlign: 'center' }}>
            <ArcGauge value={elevation} max={90} color="#d29922" label="Elevation" size={110} />
          </div>
        </div>

        {/* Raw values */}
        <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
          {[
            { label: 'az raw', val: (control.arm?.azimuth ?? 0).toFixed(3), color: '#a371f7' },
            { label: 'el raw', val: (control.arm?.elevation ?? 0).toFixed(3), color: '#d29922' },
          ].map(({ label, val, color }) => (
            <div key={label} style={{
              flex: 1, padding: '6px 10px', borderRadius: 8,
              background: '#0d1117', border: '1px solid #30363d',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            }}>
              <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: '.07em', textTransform: 'uppercase', color: '#8b949e' }}>{label}</span>
              <span style={{ fontFamily: 'IBM Plex Mono, monospace', fontSize: 11, fontWeight: 600, color }}>{val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Last dispatched action ── */}
      <div className="act-action-row">
        <span className="act-action-label">
          <Zap size={11} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
          Last Action
        </span>
        <span className={`act-action-pill ${action === 'none' ? 'none' : action} ${actionFlash ? 'active' : ''}`}>
          {action === 'none' ? '— none —' : action}
        </span>
      </div>

      {/* ── ESP32 devices ── */}
      <div className="act-card" style={{ marginTop: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
          <Wifi size={13} style={{ color: '#58a6ff' }} />
          <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: '.09em', textTransform: 'uppercase', color: '#8b949e' }}>
            ESP32 Nodes
          </span>
          <span style={{
            marginLeft: 'auto', fontFamily: 'IBM Plex Mono, monospace', fontSize: 10, fontWeight: 600,
            padding: '2px 7px', borderRadius: 5, background: '#0d2340', color: '#58a6ff',
            border: '1px solid #1a3a60',
          }}>
            {deviceEntries.filter(([, s]) => s === 'online').length} / {deviceEntries.length} online
          </span>
        </div>

        {deviceEntries.length === 0 ? (
          <div className="act-empty">
            <WifiOff size={20} style={{ margin: '0 auto 6px', display: 'block', color: '#8b949e' }} />
            No devices registered
          </div>
        ) : (
          deviceEntries.map(([id, status]) => (
            <div key={id} className="act-device-row">
              <span className="act-device-id">{id}</span>
              <span className={`act-device-status ${status}`}>{status}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default ActuatorStatusPanel;