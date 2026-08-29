import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const params = {};
let schema = null;
let mode = "live";
let replayPlaying = false;
let replayIndex = 0;
let nFrames = 0;
let camUserLocked = false;
let fitted = false;

const canvas = document.getElementById("c");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a0f);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 2000);
camera.position.set(22, 16, 28);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.addEventListener("start", () => { camUserLocked = true; });

const geom = new THREE.BufferGeometry();
const mat = new THREE.PointsMaterial({ size: 0.11, vertexColors: true, sizeAttenuation: true });
const points = new THREE.Points(geom, mat);
scene.add(points);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function setCloud(frame) {
  if (!frame || !frame.x || !frame.x.length) return;
  const n = frame.x.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = frame.x[i];
    pos[i * 3 + 1] = frame.y[i];
    pos[i * 3 + 2] = frame.z[i];
    const t = frame.c ? frame.c[i] : 0.5;
    col[i * 3] = 0.25 + 0.75 * t;
    col[i * 3 + 1] = 0.55 + 0.2 * (1 - t);
    col[i * 3 + 2] = 0.9 - 0.5 * t;
  }
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
  geom.computeBoundingSphere();
  if (!camUserLocked && !fitted && frame.half) {
    const d = Math.max(frame.half * 2.4, 12);
    camera.position.set(d, d * 0.7, d);
    controls.target.set(0, 0, 0);
    fitted = true;
  }
}

function tick() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
tick();

async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

function buildPanel(sch) {
  const root = document.getElementById("panel");
  root.innerHTML = "";
  for (const z of sch.zones) {
    const wrap = document.createElement(z.id === "advanced" ? "details" : "div");
    if (z.id === "advanced") {
      wrap.className = "adv zone";
      wrap.innerHTML = `<summary>Zaawansowane</summary>`;
    } else {
      wrap.className = "zone";
      const h = document.createElement("h2");
      h.textContent = z.label;
      wrap.appendChild(h);
    }
    for (const s of z.sliders) {
      const def = sch.defaults[s.key] ?? s.min;
      params[s.key] = Number(def);

      if (s.options && s.options.length) {
        const lab = document.createElement("label");
        lab.className = "slider";
        lab.style.display = "grid";
        const name = document.createElement("span");
        name.textContent = s.label;
        const sel = document.createElement("select");
        sel.style.gridColumn = "1 / -1";
        sel.style.width = "100%";
        sel.style.background = "#0e1218";
        sel.style.color = "#e8eef6";
        sel.style.border = "1px solid #2a3444";
        sel.style.borderRadius = "6px";
        sel.style.padding = "0.35rem";
        for (const opt of s.options) {
          const o = document.createElement("option");
          o.value = opt.value;
          o.textContent = `${opt.value}: ${opt.label}`;
          if (Number(opt.value) === Number(def)) o.selected = true;
          sel.appendChild(o);
        }
        sel.addEventListener("change", () => { params[s.key] = Number(sel.value); });
        lab.append(name, sel);
        wrap.appendChild(lab);
        continue;
      }

      const lab = document.createElement("label");
      lab.className = "slider";
      const name = document.createElement("span");
      name.textContent = s.label;
      const val = document.createElement("span");
      val.className = "val";
      const inp = document.createElement("input");
      inp.type = "range";
      inp.min = s.min; inp.max = s.max; inp.step = s.step;
      inp.value = def;
      val.textContent = String(def);
      inp.addEventListener("input", () => {
        params[s.key] = Number(inp.value);
        val.textContent = inp.value;
      });
      lab.append(name, val, inp);
      wrap.appendChild(lab);
    }
    root.appendChild(wrap);
  }
}

function buildPresets(sch) {
  const box = document.getElementById("presets");
  box.innerHTML = "";
  for (const p of sch.presets) {
    const b = document.createElement("button");
    b.textContent = p.label;
    b.onclick = async () => {
      await api("/api/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preset: p.id, params }),
      });
      fitted = false;
      camUserLocked = false;
    };
    box.appendChild(b);
  }
}

document.getElementById("btnStart").onclick = () =>
  api("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume: false, params }),
  }).then(() => { fitted = false; camUserLocked = false; });

document.getElementById("btnResume").onclick = () =>
  api("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resume: true, params }),
  });

document.getElementById("btnStop").onclick = () =>
  api("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });

document.getElementById("btnApply").onclick = () =>
  api("/api/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params }),
  });

document.getElementById("modeLive").onclick = () => {
  mode = "live";
  document.getElementById("modeLive").classList.add("active");
  document.getElementById("modeReplay").classList.remove("active");
  document.getElementById("replayControls").style.display = "none";
};
document.getElementById("modeReplay").onclick = async () => {
  mode = "replay";
  document.getElementById("modeReplay").classList.add("active");
  document.getElementById("modeLive").classList.remove("active");
  document.getElementById("replayControls").style.display = "flex";
  const meta = await api("/api/trajectory/meta");
  nFrames = meta.n_frames || 0;
  document.getElementById("scrub").max = Math.max(0, nFrames - 1);
};
document.getElementById("btnPlay").onclick = () => { replayPlaying = !replayPlaying; };
document.getElementById("scrub").oninput = async (e) => {
  replayIndex = Number(e.target.value);
  setCloud(await api(`/api/trajectory/frame?i=${replayIndex}`));
};

async function poll() {
  const st = await api("/api/status");
  const el = document.getElementById("status");
  el.textContent = (st.gpu_on ? `GPU ${st.gpu}\n` : "CPU\n") + (st.message || "");
  el.className = st.error ? "err" : "";
  if (st.error) el.textContent += "\n" + st.error;
  const h = st.hud || {};
  if (h.gamma != null) document.getElementById("hG").textContent = Number(h.gamma).toFixed(3);
  if (h.gamma_max != null) document.getElementById("hGmax").textContent = Number(h.gamma_max).toFixed(3);
  if (h.r_half != null) document.getElementById("hR").textContent = Number(h.r_half).toFixed(3);
  if (h.L != null) document.getElementById("hL").textContent = Number(h.L).toFixed(2);
  if (h.v_c != null) document.getElementById("hV").textContent = Number(h.v_c).toFixed(3);

  if (mode === "live") {
    setCloud(await api("/api/live"));
  } else if (replayPlaying && nFrames > 0) {
    const spd = Number(document.getElementById("replaySpeed").value) || 1;
    replayIndex = Math.min(nFrames - 1, replayIndex + spd);
    document.getElementById("scrub").value = String(Math.floor(replayIndex));
    setCloud(await api(`/api/trajectory/frame?i=${Math.floor(replayIndex)}`));
    if (replayIndex >= nFrames - 1) replayPlaying = false;
  }
}

(async () => {
  schema = await api("/api/schema");
  if (schema.thesis) document.getElementById("teza").textContent = schema.thesis;
  buildPanel(schema);
  buildPresets(schema);
  setInterval(poll, 200);
})();
