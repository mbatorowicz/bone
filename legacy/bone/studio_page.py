"""HTML Bone Studio — panel strefowy + LIVE + hot-apply."""

from __future__ import annotations

import json
from typing import Any

from bone.ui_schema import default_param_values, panel_boot_schema


def build_studio_html(
    trajectory: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    studio_mode: bool = True,
) -> str:
    params = params or default_param_values()
    panel = panel_boot_schema()
    boot: dict[str, Any] = {
        "params": params,
        "panel": panel,
        "studioMode": studio_mode,
    }
    if trajectory is not None:
        boot["trajectory"] = trajectory
    boot_json = json.dumps(boot, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bone Studio — LIVE</title>
<style>
  :root {{
    --bg: #0b0f14; --panel: rgba(12,18,28,.94); --text: #e8eef5; --muted: #8a9bb0;
    --accent: #3d8fb5; --accent2: #c45c26; --line: #243041; --ok: #3d9a6a;
    --spawn: #6b7c8f; --runtime: #3d8fb5;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--text);
    font-family: "Segoe UI", system-ui, sans-serif; overflow:hidden; }}
  #canvas-wrap {{ position:fixed; inset:0; }}
  #side {{
    position:fixed; top:0; right:0; width:min(400px, 100vw); height:100%;
    background:var(--panel); border-left:1px solid var(--line);
    overflow:auto; z-index:5; backdrop-filter: blur(10px);
    padding:0 0 32px;
  }}
  #side.collapsed {{ transform: translateX(calc(100% - 44px)); }}
  .zone {{
    border-bottom:1px solid var(--line); padding:10px 14px 12px;
  }}
  .zone-head {{
    display:flex; align-items:baseline; justify-content:space-between; gap:8px;
    margin-bottom:6px;
  }}
  .zone-title {{
    font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--muted);
    margin:0; font-weight:600;
  }}
  .zone-hint {{ font-size:10px; color:var(--muted); opacity:.85; }}
  .zone.spawn .zone-title {{ color: var(--spawn); }}
  .zone.runtime .zone-title {{ color: var(--runtime); }}
  #toggle {{
    position:sticky; top:0; z-index:6; display:flex; gap:8px; align-items:center;
    background:var(--panel); padding:12px 14px; border-bottom:1px solid var(--line);
  }}
  h1 {{ font-size:15px; margin:0; flex:1; }}
  .row {{ margin:6px 0; }}
  .row label {{ display:flex; justify-content:space-between; font-size:12px; margin-bottom:2px; gap:8px; }}
  .row label span.val {{ color:var(--accent); font-variant-numeric: tabular-nums; }}
  .row.spawn-only label span:first-child::after {{
    content:" start"; font-size:9px; color:var(--spawn); text-transform:uppercase; margin-left:4px;
  }}
  input[type=range] {{ width:100%; accent-color: var(--accent); }}
  input[type=range]:disabled {{ opacity:.35; }}
  button, .btn {{
    background:var(--accent); color:#fff; border:0; border-radius:7px;
    padding:8px 11px; cursor:pointer; font-size:12px; margin:2px 4px 2px 0;
  }}
  button.secondary {{ background:#2a3a4d; }}
  button.danger {{ background:var(--accent2); }}
  button.ok {{ background:var(--ok); }}
  button:disabled {{ opacity:.45; cursor:not-allowed; }}
  #status {{
    font-size:11px; color:var(--muted); margin:6px 0 0; min-height:2.4em; white-space:pre-wrap;
  }}
  .btn-row {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }}
  .checks {{ display:flex; flex-wrap:wrap; gap:10px; font-size:12px; margin:6px 0; }}
  select {{
    width:100%; background:#152033; color:var(--text); border:1px solid var(--line);
    border-radius:6px; padding:7px;
  }}
  details.zone-body {{ border:0; padding:0; margin:0; }}
  details.zone-body > summary {{
    cursor:pointer; font-size:11px; color:var(--muted); list-style:none;
    margin-bottom:4px;
  }}
  details.zone-body > summary::-webkit-details-marker {{ display:none; }}
  #hud {{
    position:fixed; left:14px; bottom:14px; z-index:4;
    background:var(--panel); border:1px solid var(--line); border-radius:10px;
    padding:10px 12px; min-width:260px; font-size:12px;
  }}
  .badge {{
    display:inline-block; padding:1px 6px; border-radius:4px; font-size:10px;
    background:rgba(196,92,38,.25); color:#f0b090; margin-left:4px;
  }}
</style>
</head>
<body>
<div id="canvas-wrap"></div>
<div id="hud">
  <div><b>Bone Studio</b><span class="badge">LIVE</span></div>
  <div>t=<span id="tlabel">0</span> · krok <span id="steplabel">0</span> · N <span id="nlabel">0</span></div>
  <div style="font-size:11px;color:var(--muted);margin-top:4px">
    G=<span id="gLabel">—</span> · Gini=<span id="giniLabel">—</span>
    · |L|=<span id="lLabel">—</span> · flat=<span id="flatLabel">—</span>
  </div>
  <div style="font-size:10px;color:var(--muted);margin-top:2px">
    GPU <span id="gpuLabel">—</span>
  </div>
</div>
<aside id="side">
  <div id="toggle">
    <h1>Panel</h1>
    <button id="collapseBtn" class="secondary" title="Zwin">»</button>
  </div>
  <div id="panelRoot"></div>
</aside>

<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script>
const BOOT = {boot_json};
const state = {{
  params: {{ ...BOOT.params }},
  viz: {{ live_fps: 8, replay_speed: 1, point_size: 0.12, opacity: 0.85, brightness: 0.04 }},
  liveTimer: null,
  viewMode: 'live', // live | replay
  replayPlaying: false,
  replayIndex: 0,
  replayMeta: null,
  replayAcc: 0,
  replayLastMs: 0,
  points: null,
  boxLine: null,
  lastLiveStep: -1,
  running: false,
  applyTimer: null,
}};

const side = document.getElementById('side');
document.getElementById('collapseBtn').onclick = () => {{
  side.classList.toggle('collapsed');
  document.getElementById('collapseBtn').textContent = side.classList.contains('collapsed') ? '«' : '»';
}};

function fmt(v, step) {{
  if (step >= 1) return String(Math.round(v));
  const d = Math.max(0, Math.min(6, Math.ceil(-Math.log10(step || 1))));
  return Number(v).toFixed(d);
}}

function addSlider(parent, spec, store, opts={{}}) {{
  const row = document.createElement('div');
  row.className = 'row' + (spec.scope === 'spawn' ? ' spawn-only' : '');
  row.dataset.key = spec.key;
  const lab = document.createElement('label');
  const name = document.createElement('span');
  name.textContent = spec.label;
  const val = document.createElement('span');
  val.className = 'val';
  lab.appendChild(name); lab.appendChild(val);
  const input = document.createElement('input');
  input.type = 'range';
  input.min = spec.min; input.max = spec.max; input.step = spec.step;
  input.dataset.key = spec.key;
  input.dataset.scope = spec.scope || 'runtime';
  const cur = store[spec.key] ?? spec.min;
  input.value = cur;
  val.textContent = fmt(cur, spec.step);
  input.addEventListener('input', () => {{
    const v = Number(input.value);
    store[spec.key] = v;
    val.textContent = fmt(v, spec.step);
    if (opts.onInput) opts.onInput(spec.key, v, spec.scope);
  }});
  row.appendChild(lab); row.appendChild(input);
  parent.appendChild(row);
  return input;
}}

function setSpawnEnabled(enabled) {{
  document.querySelectorAll('input[data-scope="spawn"]').forEach(el => {{
    el.disabled = !enabled;
  }});
  const geom = document.getElementById('geomSelect');
  if (geom) geom.disabled = !enabled;
}}

function syncSliderValues() {{
  document.querySelectorAll('#panelRoot input[type=range][data-key]').forEach(inp => {{
    const k = inp.dataset.key;
    if (!(k in state.params)) return;
    inp.value = state.params[k];
    const val = inp.parentElement.querySelector('.val');
    if (val) val.textContent = fmt(Number(inp.value), Number(inp.step));
  }});
  const geom = document.getElementById('geomSelect');
  if (geom) {{
    geom.value = String(state.params.geometry ?? 0);
    const gl = document.getElementById('geomLabel');
    if (gl) gl.textContent = geom.options[geom.selectedIndex]?.text || '';
  }}
}}

function scheduleApply() {{
  if (!BOOT.studioMode) return;
  clearTimeout(state.applyTimer);
  state.applyTimer = setTimeout(() => applyRuntimeParams(false), 280);
}}

async function applyRuntimeParams(manual) {{
  if (!BOOT.studioMode) return;
  try {{
    const r = await fetch('/api/params', {{
      method:'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify(state.params),
    }});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Blad');
    if (j.G != null) document.getElementById('gLabel').textContent = Number(j.G).toFixed(3);
    if (manual || state.running) setStatus(j.message || 'Zastosowano');
  }} catch (e) {{
    if (manual) setStatus(String(e.message || e));
  }}
}}

function buildPanel() {{
  const root = document.getElementById('panelRoot');
  root.innerHTML = '';
  const P = BOOT.panel;

  // --- Sterowanie ---
  const zCtrl = document.createElement('div');
  zCtrl.className = 'zone';
  zCtrl.innerHTML = `
    <div class="zone-head"><p class="zone-title">Sterowanie</p>
      <span class="zone-hint">bieg / stop</span></div>
    <div class="btn-row">
      <button id="runBtn" class="danger">Uruchom od zera</button>
      <button id="contBtn">Kontynuuj</button>
      <button id="stopBtn" class="secondary" disabled>Stop</button>
    </div>
    <div class="btn-row">
      <button id="applyBtn" class="ok">Zastosuj dynamikę</button>
    </div>
    <div class="btn-row" style="margin-top:8px">
      <button id="modeLive" class="ok">LIVE</button>
      <button id="modeReplay" class="secondary">REPLAY</button>
    </div>
    <div id="replayBox" style="display:none;margin-top:6px">
      <div class="btn-row">
        <button id="replayPlay" class="secondary">Play</button>
        <button id="replayPause" class="secondary">Pause</button>
      </div>
      <div class="row"><label><span>Klatka</span><span class="val" id="replayFrameVal">0</span></label>
        <input type="range" id="replayScrub" min="0" max="0" step="1" value="0"/></div>
      <div style="font-size:11px;color:var(--muted)" id="replayInfo">Brak nagrania</div>
    </div>
    <div id="status">Preset lub suwaki → Uruchom. W trakcie: Zastosuj zmienia fizykę na żywo.</div>
    <div class="checks">
      <label><input type="checkbox" id="makeGif"/> GIF na koniec</label>
      <label><input type="checkbox" id="autoRotate"/> Auto-obrót</label>
      <label><input type="checkbox" id="showCube" checked/> Ramka</label>
    </div>`;
  root.appendChild(zCtrl);

  // --- Presety ---
  const zPre = document.createElement('div');
  zPre.className = 'zone';
  zPre.innerHTML = `
    <div class="zone-head"><p class="zone-title">Presety</p>
      <span class="zone-hint">ładują suwaki</span></div>
    <div class="btn-row">
      <button id="presetSphere" class="secondary">Sfera zrównoważona</button>
      <button id="presetDonut" class="secondary">Donut zrównoważony</button>
      <button id="presetSingSphere" class="secondary">Sfera → kolaps społeczny</button>
      <button id="presetSingDonut" class="secondary">Donut → kolaps społeczny</button>
    </div>`;
  root.appendChild(zPre);

  // --- Start / Dynamika / … ---
  P.zoneOrder.forEach(z => {{
    if (z.id === 'control' || z.id === 'presets') return;
    if (z.id === 'view') {{
      const zv = document.createElement('div');
      zv.className = 'zone';
      zv.innerHTML = `<div class="zone-head"><p class="zone-title">${{z.title}}</p>
        <span class="zone-hint">${{z.hint}}</span></div><div id="vizSliders"></div>`;
      root.appendChild(zv);
      return;
    }}
    const specs = P.zones[z.id] || [];
    if (!specs.length) return;
    const scopeClass = specs[0].scope === 'spawn' ? 'spawn' : 'runtime';
    const open = ['start','dynamics','emergence'].includes(z.id);
    const wrap = document.createElement('div');
    wrap.className = 'zone ' + scopeClass;
    const head = document.createElement('div');
    head.className = 'zone-head';
    head.innerHTML = `<p class="zone-title">${{z.title}}</p><span class="zone-hint">${{z.hint}}</span>`;
    wrap.appendChild(head);

    if (z.id === 'start') {{
      const geomRow = document.createElement('div');
      geomRow.className = 'row spawn-only';
      geomRow.innerHTML = `<label><span>Kształt początkowy</span><span class="val" id="geomLabel">—</span></label>
        <select id="geomSelect"></select>`;
      wrap.appendChild(geomRow);
    }}

    const body = document.createElement('div');
    const det = document.createElement('details');
    det.className = 'zone-body';
    det.open = open;
    const sum = document.createElement('summary');
    sum.textContent = open ? 'zwiń listę' : 'rozwiń listę';
    det.appendChild(sum);
    det.appendChild(body);
    wrap.appendChild(det);

    specs.forEach(spec => {{
      if (spec.key === 'geometry') return; // select
      addSlider(body, spec, state.params, {{
        onInput: (key, v, scope) => {{
          if (scope === 'runtime' && state.running) scheduleApply();
        }}
      }});
    }});
    root.appendChild(wrap);
  }});

  // geometria select
  const geomSel = document.getElementById('geomSelect');
  if (geomSel) {{
    (P.geometryOptions || []).forEach(opt => {{
      const o = document.createElement('option');
      o.value = String(opt.value);
      o.textContent = opt.label;
      geomSel.appendChild(o);
    }});
    geomSel.value = String(state.params.geometry ?? 0);
    document.getElementById('geomLabel').textContent = geomSel.options[geomSel.selectedIndex]?.text || '';
    geomSel.onchange = () => {{
      state.params.geometry = Number(geomSel.value);
      document.getElementById('geomLabel').textContent = geomSel.options[geomSel.selectedIndex].text;
    }};
  }}

  // viz
  const vizRoot = document.getElementById('vizSliders');
  (P.viz || []).forEach(spec => addSlider(vizRoot, spec, state.viz, {{
    onInput: (key) => {{
      applyViz();
      if (key === 'live_fps') scheduleLivePoll();
    }}
  }}));

  document.getElementById('runBtn').onclick = () => runSim(false);
  document.getElementById('contBtn').onclick = () => runSim(true);
  document.getElementById('stopBtn').onclick = () => stopSim();
  document.getElementById('applyBtn').onclick = () => applyRuntimeParams(true);
  document.getElementById('presetSphere').onclick = () => loadPreset('stable_sphere');
  document.getElementById('presetDonut').onclick = () => loadPreset('stable_donut');
  document.getElementById('presetSingSphere').onclick = () => loadPreset('singularity_sphere');
  document.getElementById('presetSingDonut').onclick = () => loadPreset('singularity_donut');
  document.getElementById('showCube').onchange = e => {{
    if (state.boxLine) state.boxLine.visible = e.target.checked;
  }};
  document.getElementById('modeLive').onclick = () => setViewMode('live');
  document.getElementById('modeReplay').onclick = () => setViewMode('replay');
  document.getElementById('replayPlay').onclick = () => {{ state.replayPlaying = true; }};
  document.getElementById('replayPause').onclick = () => {{ state.replayPlaying = false; }};
  document.getElementById('replayScrub').oninput = (e) => {{
    state.replayIndex = Number(e.target.value);
    state.replayPlaying = false;
    loadReplayFrame(state.replayIndex);
  }};
}}

function setViewMode(mode) {{
  state.viewMode = mode;
  const box = document.getElementById('replayBox');
  const bL = document.getElementById('modeLive');
  const bR = document.getElementById('modeReplay');
  if (box) box.style.display = mode === 'replay' ? 'block' : 'none';
  if (bL) {{ bL.className = mode === 'live' ? 'ok' : 'secondary'; }}
  if (bR) {{ bR.className = mode === 'replay' ? 'ok' : 'secondary'; }}
  if (mode === 'replay') {{
    state.replayPlaying = false;
    refreshReplayMeta();
  }} else {{
    state.replayPlaying = false;
    pollLive();
  }}
  setStatus(mode === 'live' ? 'Tryb LIVE' : 'Tryb REPLAY — odtwarzanie zapisanych klatek');
}}

async function refreshReplayMeta() {{
  try {{
    let meta;
    if (BOOT.trajectory && BOOT.trajectory.frames && BOOT.trajectory.frames.length) {{
      meta = {{
        n_frames: BOOT.trajectory.frames.length,
        times: BOOT.trajectory.frames.map(f => f.t),
        half: BOOT.trajectory.half,
        offline: true,
      }};
    }} else {{
      meta = await (await fetch('/api/trajectory/meta?' + Date.now())).json();
    }}
    state.replayMeta = meta;
    const n = Number(meta.n_frames || 0);
    const scrub = document.getElementById('replayScrub');
    const info = document.getElementById('replayInfo');
    if (scrub) {{
      scrub.max = String(Math.max(0, n - 1));
      scrub.value = String(Math.min(state.replayIndex, Math.max(0, n - 1)));
    }}
    if (info) info.textContent = n ? (n + ' klatek · prędkość w Widoku') : 'Brak nagrania — uruchom symulację';
    if (n > 0) loadReplayFrame(state.replayIndex);
  }} catch (e) {{
    const info = document.getElementById('replayInfo');
    if (info) info.textContent = 'Brak trajektorii';
  }}
}}

async function loadReplayFrame(i) {{
  const meta = state.replayMeta;
  if (!meta || !meta.n_frames) return;
  i = Math.max(0, Math.min(i, meta.n_frames - 1));
  state.replayIndex = i;
  const scrub = document.getElementById('replayScrub');
  const val = document.getElementById('replayFrameVal');
  if (scrub) scrub.value = String(i);
  if (val) val.textContent = String(i);
  try {{
    let f;
    if (meta.offline && BOOT.trajectory && BOOT.trajectory.frames) {{
      const raw = BOOT.trajectory.frames[i];
      f = {{ t: raw.t, step: i, half: BOOT.trajectory.half, n: raw.x.length, x: raw.x, y: raw.y, z: raw.z, c: raw.c }};
    }} else {{
      f = await (await fetch('/api/trajectory/frame?i=' + i)).json();
    }}
    state.lastLiveStep = -1; // wymuś redraw
    showLive(f);
  }} catch (e) {{}}
}}

function tickReplay(now) {{
  if (state.viewMode !== 'replay' || !state.replayPlaying || !state.replayMeta) return;
  const n = Number(state.replayMeta.n_frames || 0);
  if (n < 2) return;
  if (!state.replayLastMs) state.replayLastMs = now;
  const dt = Math.min(0.1, (now - state.replayLastMs) / 1000);
  state.replayLastMs = now;
  const speed = Math.max(0.25, Number(state.viz.replay_speed) || 1);
  // bazowo ~12 klatek/s przy 1×
  state.replayAcc += dt * 12 * speed;
  while (state.replayAcc >= 1) {{
    state.replayAcc -= 1;
    state.replayIndex += 1;
    if (state.replayIndex >= n) state.replayIndex = 0;
    loadReplayFrame(state.replayIndex);
  }}
}}

function setStatus(msg) {{
  const el = document.getElementById('status');
  if (el) el.textContent = msg;
}}

function setRunButtons(running) {{
  state.running = !!running;
  const runBtn = document.getElementById('runBtn');
  const contBtn = document.getElementById('contBtn');
  const stopBtn = document.getElementById('stopBtn');
  if (runBtn) runBtn.disabled = !!running;
  if (contBtn) contBtn.disabled = !!running;
  if (stopBtn) stopBtn.disabled = !running;
  setSpawnEnabled(!running);
}}

async function loadPreset(kind) {{
  try {{
    const r = await fetch('/api/presets');
    const p = await r.json();
    const preset = p[kind];
    if (!preset) throw new Error('Brak presetu');
    Object.assign(state.params, preset);
    if ('make_gif' in preset) document.getElementById('makeGif').checked = !!preset.make_gif;
    syncSliderValues();
    setStatus('Preset „' + kind + '”. Uruchom od zera albo Zastosuj (gdy bieg).');
  }} catch (e) {{
    setStatus(String(e.message || e));
  }}
}}

async function stopSim() {{
  try {{
    const r = await fetch('/api/stop', {{ method:'POST', headers: {{'Content-Type':'application/json'}}, body: '{{}}' }});
    const j = await r.json();
    setStatus(j.message || 'Stop');
  }} catch (e) {{ setStatus(String(e.message || e)); }}
}}

async function runSim(continueMode) {{
  if (!BOOT.studioMode) {{ setStatus('Uruchom: python -m bone studio'); return; }}
  const body = {{
    ...state.params,
    make_gif: document.getElementById('makeGif').checked,
    open_view: false,
    out_dir: 'out',
    continue: !!continueMode,
  }};
  setRunButtons(true);
  state.lastLiveStep = -1;
  setStatus(continueMode ? 'Kontynuacja LIVE...' : 'Start LIVE...');
  try {{
    const r = await fetch('/api/run', {{
      method:'POST',
      headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify(body),
    }});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Blad');
  }} catch (e) {{
    setRunButtons(false);
    setStatus(String(e.message || e));
  }}
}}

// --- Three.js LIVE ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.1, 800);
const renderer = new THREE.WebGLRenderer({{
  antialias: true,
  powerPreference: 'high-performance',
  failIfMajorPerformanceCaveat: false,
}});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.getElementById('canvas-wrap').appendChild(renderer.domElement);
scene.add(new THREE.AmbientLight(0xffffff, 0.95));

let theta = 0.75, phi = 0.45, radius = 30;
let camUserLocked = false; // po orbicie/zoomie nie ruszamy kamery przy LIVE
let lastBoxHalf = null;
function syncCam() {{
  camera.position.x = radius * Math.sin(theta) * Math.cos(phi);
  camera.position.y = radius * Math.sin(phi);
  camera.position.z = radius * Math.cos(theta) * Math.cos(phi);
  camera.lookAt(0,0,0);
}}
function setBg() {{
  const b = state.viz.brightness;
  scene.background = new THREE.Color(b, b*1.05, b*1.15);
}}
setBg(); syncCam();

function colorRGB(t) {{
  const b = Math.max(0, 1 - 2*t);
  const r = Math.max(0, 2*t - 1);
  const g = 1 - Math.abs(2*t - 1);
  return [r, g*0.55, b];
}}

function ensureBox(half, fitCamera) {{
  // ramka tylko gdy half się zmienił — bez resetu widoku
  const h = Math.max(0.5, Number(half) || 12);
  const changed = lastBoxHalf == null || Math.abs(h - lastBoxHalf) / Math.max(lastBoxHalf, 1e-6) > 0.08;
  if (changed) {{
    if (state.boxLine) scene.remove(state.boxLine);
    const boxGeo = new THREE.BoxGeometry(h*2, h*2, h*2);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    state.boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({{ color: 0x5a6a7a }}));
    const sc = document.getElementById('showCube');
    state.boxLine.visible = sc ? sc.checked : true;
    scene.add(state.boxLine);
    lastBoxHalf = h;
  }}
  // kamera: tylko pierwszy fit albo gdy użytkownik jeszcze nie ruszał widoku
  if (fitCamera && !camUserLocked) {{
    radius = Math.max(h * 2.4, 12);
    syncCam();
  }}
}}

function applyViz() {{
  setBg();
  if (state.points) {{
    state.points.material.size = state.viz.point_size;
    state.points.material.opacity = state.viz.opacity;
  }}
}}

function showLive(f) {{
  if (!f || !f.x || !f.x.length) {{
    document.getElementById('tlabel').textContent = '—';
    document.getElementById('steplabel').textContent = '0';
    document.getElementById('nlabel').textContent = '0';
    return;
  }}
  if (f.step === state.lastLiveStep && state.points) return;
  const firstFrame = state.lastLiveStep < 0 || !state.points;
  state.lastLiveStep = f.step;
  ensureBox(f.half || 12, firstFrame);
  const n = f.x.length;
  const positions = new Float32Array(n*3);
  const cols = new Float32Array(n*3);
  for (let i=0; i<n; i++) {{
    positions[i*3]=f.x[i]; positions[i*3+1]=f.y[i]; positions[i*3+2]=f.z[i];
    const rgb = colorRGB(f.c[i]);
    cols[i*3]=rgb[0]; cols[i*3+1]=rgb[1]; cols[i*3+2]=rgb[2];
  }}
  if (state.points) {{
    const geo = state.points.geometry;
    const sameN = geo.getAttribute('position') && geo.getAttribute('position').count === n;
    if (sameN) {{
      geo.getAttribute('position').array.set(positions);
      geo.getAttribute('position').needsUpdate = true;
      geo.getAttribute('color').array.set(cols);
      geo.getAttribute('color').needsUpdate = true;
    }} else {{
      scene.remove(state.points);
      state.points = null;
    }}
  }}
  if (!state.points) {{
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    state.points = new THREE.Points(geo, new THREE.PointsMaterial({{
      size: state.viz.point_size, vertexColors: true, sizeAttenuation: true,
      transparent: true, opacity: state.viz.opacity,
    }}));
    scene.add(state.points);
  }}
  document.getElementById('tlabel').textContent = Number(f.t).toFixed(3);
  document.getElementById('steplabel').textContent = String(f.step);
  document.getElementById('nlabel').textContent = String(f.n || n);
}}

let dragging=false, prevX=0, prevY=0;
renderer.domElement.addEventListener('pointerdown', e => {{ dragging=true; prevX=e.clientX; prevY=e.clientY; }});
window.addEventListener('pointerup', () => dragging=false);
window.addEventListener('pointermove', e => {{
  if (!dragging) return;
  camUserLocked = true;
  theta -= (e.clientX-prevX)*0.005;
  phi = Math.max(-1.25, Math.min(1.25, phi + (e.clientY-prevY)*0.005));
  prevX=e.clientX; prevY=e.clientY; syncCam();
}});
renderer.domElement.addEventListener('wheel', e => {{
  camUserLocked = true;
  radius = Math.max(6, Math.min(120, radius * (e.deltaY>0 ? 1.08 : 0.92)));
  syncCam();
}});

function loop(now) {{
  requestAnimationFrame(loop);
  const ar = document.getElementById('autoRotate');
  if (ar && ar.checked) {{ theta += 0.003; syncCam(); }}
  tickReplay(now || performance.now());
  renderer.render(scene, camera);
}}
requestAnimationFrame(loop);
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

async function pollLive() {{
  if (!BOOT.studioMode) return;
  if (state.viewMode !== 'live') return;
  try {{
    const f = await (await fetch('/api/live?' + Date.now())).json();
    showLive(f);
  }} catch (e) {{}}
}}

function scheduleLivePoll() {{
  if (state.liveTimer) clearInterval(state.liveTimer);
  if (!BOOT.studioMode) return;
  const hz = Math.max(1, Math.min(30, Number(state.viz.live_fps) || 8));
  state.liveTimer = setInterval(pollLive, Math.round(1000 / hz));
}}

setInterval(async () => {{
  if (!BOOT.studioMode) return;
  try {{
    const s = await (await fetch('/api/status')).json();
    setRunButtons(!!s.running);
    if (s.message) setStatus(s.message);
    if (s.active_G != null) document.getElementById('gLabel').textContent = Number(s.active_G).toFixed(3);
    if (s.gini != null) document.getElementById('giniLabel').textContent = Number(s.gini).toFixed(3);
    if (s.L_mag != null) document.getElementById('lLabel').textContent = Number(s.L_mag).toFixed(2);
    if (s.flattening != null) document.getElementById('flatLabel').textContent = Number(s.flattening).toFixed(3);
    if (s.gpu != null) {{
      const el = document.getElementById('gpuLabel');
      if (el) el.textContent = (s.gpu_enabled ? 'CUDA · ' : 'CPU · ') + s.gpu;
    }}
    if (s.error) setStatus('Blad: ' + s.error);
  }} catch (e) {{}}
}}, 700);

buildPanel();
if (BOOT.studioMode) {{
  setStatus('LIVE / REPLAY · Spawn = od zera · Dynamika = Zastosuj w trakcie.');
  scheduleLivePoll();
  pollLive();
}} else if (BOOT.trajectory && BOOT.trajectory.frames && BOOT.trajectory.frames.length) {{
  setStatus('Replay offline (cube_view.html)');
  setViewMode('replay');
}} else setStatus('Widok — python -m bone studio');
</script>
</body>
</html>
"""
