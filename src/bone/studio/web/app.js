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
  running: false,
  busy: false,
  presetId: null,
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

/* Telemetria poprzedniego biegu musi zniknąć w chwili startu nowego. Plakietka
   błędu siły jest tu najważniejsza: serwer podaje ją tylko w tych rzadkich
   odczytach, w których faktycznie zmierzył błąd, więc gdyby jej nie czyścić,
   wisiałaby na ekranie z liczbą z układu, którego już nie ma. */
function resetHud() {
  for (const id of ["mGamma", "mGammaMax", "mBeta", "mDt", "mVirial", "mRhalf", "mEdrift", "mLdrift", "mPres"]) {
    el(id).textContent = "—";
  }
  el("mEdrift").classList.remove("warn", "bad");
  el("mTime").textContent = "t = 0";
  el("mStep").textContent = "krok 0";
  el("mErr").hidden = true;
  state.drift = [];
  paintSpark();
}

/* --------------------------------------------------------------- panel UI */

let applyTimer = null;
function scheduleApply() {
  clearTimeout(applyTimer);
  applyTimer = setTimeout(async () => {
    try {
      const answer = await post("/api/apply", { params: state.params });
      if (answer.error) notify(answer.error, true);
    } catch (e) {
      notify("Nie udało się przesłać zmiany do serwera.", true);
    }
  }, 140);
}

const LIVE_TITLE = "Działa w trakcie symulacji";
const STARTUP_TITLE = "Wymaga ponownego uruchomienia";

function buildHint(text) {
  if (!text) return null;
  const hint = document.createElement("p");
  hint.className = "hint";
  hint.textContent = text;
  return hint;
}

/* Kontrolki potrafią się odświeżyć z `state.params`, bo wczytanie presetu
   podmienia cały zestaw parametrów i panel musi natychmiast pokazać, co
   naprawdę pojedzie. */
const setters = new Map();

/* Wartość wyświetlana pochodzi z parametru, nie z pozycji suwaka. Presety liczą
   G z zadanej prędkości na brzegu, więc wychodzą liczby w rodzaju 0,140625,
   których siatka kroku suwaka nie potrafi trafić. Suwak jest wtedy przybliżonym
   wskaźnikiem, a autorytetem jest ta etykieta i `state.params`. */
const show = (v) => (typeof v === "number" ? String(Number(v.toPrecision(6))) : String(v));

function buildChoice(control) {
  const wrap = document.createElement("div");
  wrap.className = "field choice" + (control.live ? "" : " startup");

  const top = document.createElement("div");
  top.className = "top";
  const name = document.createElement("span");
  name.textContent = control.label;
  top.appendChild(name);

  const select = document.createElement("select");
  select.title = control.live ? LIVE_TITLE : STARTUP_TITLE;
  for (const option of control.options) {
    const node = document.createElement("option");
    node.value = option.value;
    node.textContent = option.label;
    select.appendChild(node);
  }
  select.value = state.params[control.key];
  select.addEventListener("change", () => {
    state.params[control.key] = select.value;
    clearPresetMark();
    if (control.live) scheduleApply();
  });
  setters.set(control.key, (v) => { select.value = v; });

  wrap.append(top, select);
  const hint = buildHint(control.hint);
  if (hint) wrap.appendChild(hint);
  return wrap;
}

function buildSlider(control) {
  const wrap = document.createElement("div");
  wrap.className = "field" + (control.live ? "" : " startup");

  const top = document.createElement("div");
  top.className = "top";
  const name = document.createElement("span");
  name.textContent = control.label;
  const value = document.createElement("b");
  top.append(name, value);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = control.min;
  slider.max = control.max;
  slider.step = control.step;
  slider.title = control.live ? LIVE_TITLE : STARTUP_TITLE;

  slider.addEventListener("input", () => {
    state.params[control.key] = Number(slider.value);
    value.textContent = show(Number(slider.value));
    clearPresetMark();
    if (control.live) scheduleApply();
  });
  setters.set(control.key, (v) => {
    slider.value = v;
    value.textContent = show(v);
  });
  setters.get(control.key)(state.params[control.key]);

  wrap.append(top, slider);
  const hint = buildHint(control.hint);
  if (hint) wrap.appendChild(hint);
  return wrap;
}

/* Rodzaj kontrolki podaje schemat z config.py, nie ten plik. Dzięki temu dodanie
   pola do konfiguracji nie wymaga tknięcia frontendu. */
function buildControl(control) {
  return control.kind === "choice" ? buildChoice(control) : buildSlider(control);
}

function syncControls() {
  for (const [key, apply] of setters) {
    if (key in state.params) apply(state.params[key]);
  }
}

/* Podświetlony chip znaczy „ten preset stoi w panelu". Każda ręczna zmiana
   pokrętła to unieważnia, bo panel przestaje odpowiadać presetowi — inaczej
   interfejs twierdziłby, że pojedzie coś, co już zostało nadpisane. */
function markPreset(id) {
  state.presetId = id;
  for (const chip of el("presets").children) {
    chip.classList.toggle("on", chip.dataset.preset === id);
  }
}

function clearPresetMark() {
  if (state.presetId !== null) markPreset(null);
}

async function loadPreset(entry) {
  // Start idzie samym identyfikatorem. Pobieranie konfiguracji było tu
  // warunkiem wstępnym — a działający proces studia często jest starszy niż
  // frontend, nie zna /api/preset i nie wkłada configu do schematu. Chip
  // kończył wtedy na „Nie udało się pobrać presetu" i nie wysyłał startu.
  //
  // Params panelu NIE jadą z tym żądaniem: stary serwer nakładał je na preset
  // i wymazywał go do wartości domyślnych. Nowy serwer i tak bierze preset
  // jako źródło prawdy. Panel dopina się z odpowiedzi po starcie.
  if (entry.config) {
    state.params = { ...entry.config };
    syncControls();
  }
  markPreset(entry.id);
  await run({ preset: entry.id, restart: state.running });
}

function buildPanel(schema) {
  state.params = { ...schema.defaults };
  setters.clear();

  const root = el("controls");
  root.innerHTML = "";
  const open = schema.open_sections ?? 2;
  schema.groups.forEach((group, i) => {
    if (!group.controls.length) return;
    const box = document.createElement("details");
    box.className = "group";
    box.open = i < open;
    const head = document.createElement("summary");
    head.textContent = group.label;
    const count = document.createElement("em");
    count.textContent = group.controls.length;
    head.appendChild(count);
    box.appendChild(head);
    for (const control of group.controls) box.appendChild(buildControl(control));
    root.appendChild(box);
  });

  const chips = el("presets");
  chips.innerHTML = "";
  for (const p of schema.presets) {
    const button = document.createElement("button");
    button.textContent = p.label;
    button.dataset.preset = p.id;
    button.title = "Wczytaj konfigurację i uruchom";
    button.addEventListener("click", () => loadPreset(p));
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

/* ------------------------------------------------------- akcje i ich echo */

/* Odpowiedź na kliknięcie musi przeżyć dłużej niż jeden obrót odpytywania.
   Status jest odświeżany trzy razy na sekundę telemetrią biegu, więc komunikat
   wpisany wprost znikał zanim dało się go przeczytać — a to właśnie tam trafiał
   powód, dla którego klik nie dał efektu. */
const NOTICE_MS = 4000;
let noticeUntil = 0;

function notify(text, bad = false) {
  noticeUntil = performance.now() + NOTICE_MS;
  const node = el("status");
  node.textContent = text;
  node.classList.toggle("bad", Boolean(bad));
  node.classList.toggle("notice", !bad);
}

function setStatus(text, bad) {
  if (performance.now() < noticeUntil) return;
  const node = el("status");
  node.textContent = text;
  node.classList.toggle("bad", Boolean(bad));
  node.classList.remove("notice");
}

function syncButtons() {
  const restart = state.running;
  el("btnStart").textContent = restart ? "Restart" : "Uruchom";
  el("btnStart").title = restart
    ? "Zatrzymuje bieżący bieg i startuje z konfiguracją z panelu"
    : "Startuje bieg z konfiguracją z panelu";
  el("btnStart").disabled = state.busy;
  el("btnResume").disabled = state.busy || state.running;
  el("btnStop").disabled = state.busy || !state.running;

  const noFrames = state.frameCount === 0;
  el("btnPlay").disabled = noFrames;
  el("scrub").disabled = noFrames;
}

async function run(extra) {
  if (state.busy) return;
  state.busy = true;
  syncButtons();
  notify(extra.restart ? "Restart biegu…" : "Uruchamianie…");
  try {
    const answer = await post("/api/start", extra.preset
      ? { preset: extra.preset, restart: Boolean(extra.restart) }
      : { params: state.params, ...extra });
    if (answer.error) {
      // odmowa znaczy, że poprzedni bieg trwa dalej — jego telemetria zostaje
      notify(answer.error, true);
    } else {
      // serwer potwierdził start, więc przyciski nie muszą czekać na odpytanie
      state.running = true;
      state.fitted = false;
      state.cameraTouched = false;
      resetHud();
      // Panel staje się odbiciem konfiguracji, którą serwer naprawdę przyjął.
      // Przy wznawianiu pochodzi ona z checkpointu, więc bez tego panel
      // pokazywałby liczby, pod którymi bieg nie idzie.
      if (answer.config) {
        state.params = { ...answer.config };
        syncControls();
      }
      if (answer.resumed) {
        clearPresetMark();
        notify(answer.note || "Wznowiono — konfiguracja wczytana z checkpointu.");
      } else if (answer.note) {
        notify(answer.note);
      }
    }
  } catch (e) {
    notify("Serwer nie odpowiedział na żądanie startu.", true);
  } finally {
    state.busy = false;
    syncButtons();
  }
}

async function stop() {
  notify("Zatrzymywanie…");
  try {
    const answer = await post("/api/stop");
    if (!answer.was_running) notify("Nie było czego zatrzymywać.");
  } catch (e) {
    notify("Serwer nie odpowiedział na żądanie zatrzymania.", true);
  }
}

function setHint(text) {
  const node = el("hint");
  node.textContent = text || "";
  node.classList.toggle("visible", Boolean(text));
}

/* ------------------------------------------------------------- odpytywanie */

let statusTicks = 0;

async function pollStatus() {
  try {
    const s = await get("/api/status");
    if (s.running !== state.running) {
      state.running = Boolean(s.running);
      syncButtons();
    }
    setStatus(s.error ? s.error : s.message, Boolean(s.error));
    setHint(s.error ? "" : s.hint);
    el("mBackend").textContent = s.backend + (s.cuda ? " · " + s.cuda_name : " · CPU");
    paintDiagnostics(s.diagnostics);
  } catch (e) {
    setStatus("Brak połączenia z serwerem.", true);
  }
  // liczba klatek rośnie w trakcie biegu, więc odtwarzacz musi ją odświeżać —
  // inaczej suwak zostaje na stanie z chwili przełączenia trybu
  if (state.mode === "replay" && ++statusTicks % 10 === 0) await refreshFrames();
}

async function refreshFrames() {
  try {
    const meta = await get("/api/trajectory");
    const count = meta.n_frames || 0;
    if (count === state.frameCount) return;
    state.frameCount = count;
    el("scrub").max = String(Math.max(0, count - 1));
    if (state.frame >= count) state.frame = Math.max(0, count - 1);
    paintFrameLabel();
    syncButtons();
  } catch (e) {
    /* następna próba za chwilę */
  }
}

function paintFrameLabel() {
  const shown = state.frameCount ? state.frame + 1 : 0;
  el("frameLabel").textContent = `${shown} / ${state.frameCount}`;
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
      paintFrameLabel();
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

// Uruchom na działającym biegu znaczy „restart", a nie „nic". Serwer zatrzymuje
// wtedy poprzedni bieg i czeka na jego zejście, bo katalog wyjściowy ma jednego
// właściciela.
el("btnStart").addEventListener("click", () => run(state.running ? { restart: true } : {}));
el("btnResume").addEventListener("click", () => run({ resume: true }));
el("btnStop").addEventListener("click", stop);

el("modeLive").addEventListener("click", () => {
  state.mode = "live";
  state.playing = false;
  el("btnPlay").textContent = "▶";
  el("modeLive").classList.add("on");
  el("modeReplay").classList.remove("on");
  el("replayBar").hidden = true;
});

el("modeReplay").addEventListener("click", async () => {
  state.mode = "replay";
  el("modeReplay").classList.add("on");
  el("modeLive").classList.remove("on");
  el("replayBar").hidden = false;
  await refreshFrames();
  if (state.frameCount === 0) {
    notify("Brak zapisanych klatek — uruchom bieg i wróć tu za chwilę.");
  }
});

el("btnPlay").addEventListener("click", () => {
  if (state.frameCount === 0) return;
  state.playing = !state.playing;
  el("btnPlay").textContent = state.playing ? "❚❚" : "▶";
});

el("scrub").addEventListener("input", async (event) => {
  if (state.frameCount === 0) return;
  state.playing = false;
  el("btnPlay").textContent = "▶";
  state.frame = Number(event.target.value);
  paintFrameLabel();
  try {
    const response = await fetch(`/api/trajectory/frame?i=${state.frame}`);
    applyView(await response.arrayBuffer());
  } catch (e) {
    /* następny ruch suwaka spróbuje ponownie */
  }
});

/* ------------------------------------------------------------------ start */

(async () => {
  try {
    state.schema = await get("/api/schema");
  } catch (e) {
    notify("Nie udało się pobrać schematu konfiguracji — serwer nie odpowiada.", true);
    return;
  }
  buildPanel(state.schema);
  resetHud();
  syncButtons();
  setInterval(pollStatus, 300);
  setInterval(pollView, 60);
})();
