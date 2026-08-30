import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const MAGIC = 0x424f4e45;
const HEADER_BYTES = 16;

const state = {
  schema: null,
  params: {},
  mode: "live",
  playing: false,
  frame: 0,
  frameCount: 0,
  cameraTouched: false,
  fitted: false,
  inFlight: false,
  drift: [],
};

/* ------------------------------------------------------------------ scena */

const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 100000);
camera.position.set(26, 18, 32);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.addEventListener("start", () => { state.cameraTouched = true; });

// β = |v|/c steruje kolorem w shaderze: chłodny błękit dla wolnych cząstek,
// złoto i biel dla relatywistycznych. Liczenie koloru na GPU pozwala wysyłać
// jedną liczbę na cząstkę zamiast trzech.
const material = new THREE.ShaderMaterial({
  uniforms: { uSize: { value: 2.6 }, uOpacity: { value: 0.92 } },
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  vertexShader: `
    attribute float shade;
    varying float vShade;
    uniform float uSize;
    void main() {
      vShade = shade;
      vec4 mv = modelViewMatrix * vec4(position, 1.0);
      gl_PointSize = uSize * (260.0 / max(-mv.z, 0.001));
      gl_Position = projectionMatrix * mv;
    }`,
  fragmentShader: `
    varying float vShade;
    uniform float uOpacity;
    void main() {
      vec2 d = gl_PointCoord - vec2(0.5);
      float r2 = dot(d, d);
      if (r2 > 0.25) discard;
      float edge = smoothstep(0.25, 0.02, r2);
      float b = clamp(vShade, 0.0, 1.0);
      float k = pow(b, 0.45);
      vec3 cold = vec3(0.22, 0.51, 0.92);
      vec3 mid  = vec3(0.45, 0.85, 0.90);
      vec3 hot  = vec3(1.00, 0.83, 0.42);
      vec3 col = k < 0.5 ? mix(cold, mid, k * 2.0) : mix(mid, hot, (k - 0.5) * 2.0);
      col = mix(col, vec3(1.0), pow(b, 4.0) * 0.8);
      gl_FragColor = vec4(col, uOpacity * edge);
    }`,
});

const geometry = new THREE.BufferGeometry();
let posAttr = null;
let shadeAttr = null;
scene.add(new THREE.Points(geometry, material));

function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas);
resize();

(function loop() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
})();

/* ----------------------------------------------------------- bufor widoku */

function applyView(buffer) {
  if (!buffer || buffer.byteLength < HEADER_BYTES) return;
  const head = new DataView(buffer);
  if (head.getUint32(0, true) !== MAGIC) return;
  const n = head.getUint32(4, true);
  const half = head.getFloat32(8, true);
  if (!n) return;

  const positions = new Float32Array(buffer, HEADER_BYTES, n * 3);
  const shades = new Float32Array(buffer, HEADER_BYTES + n * 12, n);

  // realokuj tylko gdy zmieniła się liczba cząstek — inaczej nadpisz w miejscu
  if (!posAttr || posAttr.count !== n) {
    posAttr = new THREE.BufferAttribute(new Float32Array(n * 3), 3);
    shadeAttr = new THREE.BufferAttribute(new Float32Array(n), 1);
    posAttr.setUsage(THREE.DynamicDrawUsage);
    shadeAttr.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", posAttr);
    geometry.setAttribute("shade", shadeAttr);
  }
  posAttr.array.set(positions);
  shadeAttr.array.set(shades);
  posAttr.needsUpdate = true;
  shadeAttr.needsUpdate = true;
  geometry.setDrawRange(0, n);
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(), half * 2);

  if (!state.cameraTouched && !state.fitted && half > 0) {
    const d = Math.max(half * 2.3, 8);
    camera.position.set(d * 0.8, d * 0.55, d);
    controls.target.set(0, 0, 0);
    controls.update();
    state.fitted = true;
  }
}

/* ------------------------------------------------------------------- HUD */

const el = (id) => document.getElementById(id);
const fmt = (v, d = 3) => (v === undefined || v === null || !isFinite(v) ? "—" : Number(v).toFixed(d));
const sci = (v) => (v === undefined || !isFinite(v) ? "—" : (v >= 0 ? "+" : "") + Number(v).toExponential(2));

function driftClass(node, value) {
  const a = Math.abs(value ?? 0);
  node.classList.toggle("warn", a >= 1e-3 && a < 1e-2);
  node.classList.toggle("bad", a >= 1e-2);
}

function paintSpark() {
  const c = el("spark");
  const ctx = c.getContext("2d");
  const w = c.width;
  const h = c.height;
  ctx.clearRect(0, 0, w, h);
  const data = state.drift;
  if (data.length < 2) return;
  const peak = Math.max(1e-12, ...data.map(Math.abs));
  ctx.strokeStyle = "#1e2634";
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  ctx.strokeStyle = peak >= 1e-2 ? "#e07a7a" : peak >= 1e-3 ? "#d8b66a" : "#74c69d";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h / 2 - (v / peak) * (h / 2 - 3);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
}

function paintDiagnostics(d) {
  if (!d || d.step === undefined) return;
  el("mGamma").textContent = fmt(d.gamma_mean, 5);
  el("mGammaMax").textContent = fmt(d.gamma_max, 4);
  el("mBeta").textContent = fmt(d.beta_max, 4);
  el("mDt").textContent = d.dt ? Number(d.dt).toExponential(2) : "—";
  el("mVirial").textContent = fmt(d.virial, 3);
  el("mRhalf").textContent = fmt(d.r_half_ratio, 3);

  const drift = el("mEdrift");
  drift.textContent = sci(d.E_drift);
  driftClass(drift, d.E_drift);

  el("mLdrift").textContent = sci(d.L_drift);
  el("mPres").textContent = d.P_residual !== undefined ? Number(d.P_residual).toExponential(1) : "—";
  el("mTime").textContent = "t = " + fmt(d.t, 3);
  el("mStep").textContent = "krok " + (d.step | 0);

  const badge = el("mErr");
  if (d.force_err_rms !== undefined) {
    badge.hidden = false;
    badge.textContent = "błąd siły " + (d.force_err_rms * 100).toFixed(2) + "%";
  }

  state.drift.push(d.E_drift ?? 0);
  if (state.drift.length > 220) state.drift.shift();
  paintSpark();
}

/* --------------------------------------------------------------- panel UI */

let applyTimer = null;
function scheduleApply() {
  clearTimeout(applyTimer);
  applyTimer = setTimeout(() => {
    post("/api/apply", { params: state.params });
  }, 140);
}

function buildChoice(key, spec) {
  const wrap = document.createElement("div");
  wrap.className = "choice";
  const label = document.createElement("label");
  label.textContent = spec.label;
  const select = document.createElement("select");
  for (const option of spec.options) {
    const node = document.createElement("option");
    node.value = option;
    node.textContent = option;
    if (state.params[key] === option) node.selected = true;
    select.appendChild(node);
  }
  select.addEventListener("change", () => { state.params[key] = select.value; });
  wrap.append(label, select);
  return wrap;
}

function buildField(control) {
  const wrap = document.createElement("div");
  wrap.className = "field" + (control.live ? "" : " startup");

  const top = document.createElement("div");
  top.className = "top";
  const name = document.createElement("span");
  name.textContent = control.label;
  const value = document.createElement("b");
  value.textContent = String(control.value);
  top.append(name, value);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = control.min;
  slider.max = control.max;
  slider.step = control.step;
  slider.value = control.value;
  slider.title = control.live
    ? "Działa w trakcie symulacji"
    : "Wymaga ponownego uruchomienia";

  slider.addEventListener("input", () => {
    const v = Number(slider.value);
    state.params[control.key] = v;
    value.textContent = slider.value;
    if (control.live) scheduleApply();
  });

  wrap.append(top, slider);
  return wrap;
}

function buildPanel(schema) {
  state.params = { ...schema.defaults };

  const choices = el("choices");
  choices.innerHTML = "";
  for (const [key, spec] of Object.entries(schema.choices)) {
    choices.appendChild(buildChoice(key, spec));
  }

  const root = el("controls");
  root.innerHTML = "";
  schema.groups.forEach((group, i) => {
    const box = document.createElement("details");
    box.className = "group";
    box.open = i < 2;
    const head = document.createElement("summary");
    head.textContent = group.label;
    box.appendChild(head);
    for (const control of group.controls) box.appendChild(buildField(control));
    root.appendChild(box);
  });

  const chips = el("presets");
  chips.innerHTML = "";
  for (const p of schema.presets) {
    const button = document.createElement("button");
    button.textContent = p.label;
    button.addEventListener("click", () => {
      [...chips.children].forEach((c) => c.classList.remove("on"));
      button.classList.add("on");
      start({ preset: p.id });
    });
    chips.appendChild(button);
  }
}

/* -------------------------------------------------------------- transport */

async function get(path) {
  const response = await fetch(path);
  return response.json();
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  return response.json();
}

async function start(extra) {
  state.fitted = false;
  state.cameraTouched = false;
  state.drift = [];
  const answer = await post("/api/start", { params: state.params, ...extra });
  if (answer.error) setStatus(answer.error, true);
}

function setStatus(text, bad) {
  const node = el("status");
  node.textContent = text;
  node.classList.toggle("bad", Boolean(bad));
}

function setHint(text) {
  const node = el("hint");
  node.textContent = text || "";
  node.classList.toggle("visible", Boolean(text));
}

/* ------------------------------------------------------------- odpytywanie */

async function pollStatus() {
  try {
    const s = await get("/api/status");
    setStatus(s.error ? s.error : s.message, Boolean(s.error));
    setHint(s.error ? "" : s.hint);
    el("mBackend").textContent = s.backend + (s.cuda ? " · " + s.cuda_name : " · CPU");
    paintDiagnostics(s.diagnostics);
  } catch (e) {
    setStatus("Brak połączenia z serwerem.", true);
  }
}

async function pollView() {
  if (state.inFlight) return;
  state.inFlight = true;
  try {
    if (state.mode === "live") {
      const response = await fetch("/api/view");
      applyView(await response.arrayBuffer());
    } else if (state.playing && state.frameCount > 0) {
      state.frame = (state.frame + 1) % state.frameCount;
      el("scrub").value = String(state.frame);
      el("frameLabel").textContent = `${state.frame + 1} / ${state.frameCount}`;
      const response = await fetch(`/api/trajectory/frame?i=${state.frame}`);
      applyView(await response.arrayBuffer());
    }
  } catch (e) {
    /* serwer chwilowo zajęty — następna próba za chwilę */
  } finally {
    state.inFlight = false;
  }
}

/* -------------------------------------------------------------- zdarzenia */

el("btnStart").addEventListener("click", () => start({}));
el("btnResume").addEventListener("click", () => start({ resume: true }));
el("btnStop").addEventListener("click", () => post("/api/stop"));

el("modeLive").addEventListener("click", () => {
  state.mode = "live";
  el("modeLive").classList.add("on");
  el("modeReplay").classList.remove("on");
  el("replayBar").hidden = true;
});

el("modeReplay").addEventListener("click", async () => {
  state.mode = "replay";
  el("modeReplay").classList.add("on");
  el("modeLive").classList.remove("on");
  el("replayBar").hidden = false;
  const meta = await get("/api/trajectory");
  state.frameCount = meta.n_frames || 0;
  el("scrub").max = String(Math.max(0, state.frameCount - 1));
  el("frameLabel").textContent = `0 / ${state.frameCount}`;
});

el("btnPlay").addEventListener("click", () => {
  state.playing = !state.playing;
  el("btnPlay").textContent = state.playing ? "❚❚" : "▶";
});

el("scrub").addEventListener("input", async (event) => {
  state.playing = false;
  el("btnPlay").textContent = "▶";
  state.frame = Number(event.target.value);
  el("frameLabel").textContent = `${state.frame + 1} / ${state.frameCount}`;
  const response = await fetch(`/api/trajectory/frame?i=${state.frame}`);
  applyView(await response.arrayBuffer());
});

/* ------------------------------------------------------------------ start */

(async () => {
  state.schema = await get("/api/schema");
  buildPanel(state.schema);
  setInterval(pollStatus, 300);
  setInterval(pollView, 60);
})();
