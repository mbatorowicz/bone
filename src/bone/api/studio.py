"""Bone Studio — cienki HTTP + wątek Engine."""

from __future__ import annotations

import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from bone.config.schema import (
    AppConfig,
    merge_runtime,
    preset_burst,
    preset_cluster,
    preset_galaxy,
    schema_for_ui,
)
from bone.engine import Engine
from bone.io.checkpoint import load_checkpoint, save_checkpoint
from bone.io.live import pack_live
from bone.io.trajectory import TrajectoryWriter, load_frame
from bone.physics.gpu import device_label, gpu_enabled

_WEB = Path(__file__).resolve().parents[3] / "web"
_STATE: dict = {
    "lock": threading.Lock(),
    "running": False,
    "stop": False,
    "live": None,
    "message": "Gotowy.",
    "error": "",
    "hud": {},
    "cfg": AppConfig(),
    "out_dir": "out",
    "pending_cfg": None,
}


def _presets() -> dict[str, AppConfig]:
    return {
        "galaxy": preset_galaxy(),
        "cluster": preset_cluster(),
        "burst": preset_burst(),
        # stare aliasy
        "balance": preset_galaxy(),
        "exploit": preset_cluster(),
        "tribes": preset_burst(),
    }


def _run_job(cfg: AppConfig, resume: bool) -> None:
    _STATE["running"] = True
    _STATE["stop"] = False
    _STATE["error"] = ""
    _STATE["message"] = "Start…"
    _STATE["cfg"] = cfg
    out = Path(cfg.io.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = TrajectoryWriter(out, stride=cfg.io.point_stride)
    try:
        if resume and (out / "checkpoint.npz").exists():
            u = load_checkpoint(out, cfg)
            eng = Engine(cfg, universe=u)
            _STATE["message"] = f"Kontynuacja t={u.t:.2f}"
        else:
            eng = Engine(cfg)

        def on_live(universe):
            payload = pack_live(universe)
            with _STATE["lock"]:
                _STATE["live"] = payload

        def on_frame(universe):
            writer.add(universe)

        def on_progress(step, total, universe, row):
            _STATE["hud"] = {
                "gamma": row.get("mean_gamma", 1),
                "gamma_max": row.get("max_gamma", 1),
                "r_half": row.get("collapse_ratio", 1),
                "L": row.get("L_mag", 0),
                "v_c": row.get("v_over_c", 0),
                "t": universe.t,
                "step": universe.step,
            }
            _STATE["message"] = (
                f"t={universe.t:.2f} ⟨γ⟩={row.get('mean_gamma', 1):.3f} "
                f"r½={row.get('collapse_ratio', 1):.3f} |L|={row.get('L_mag', 0):.2f} "
                f"⟨v⟩/c={row.get('v_over_c', 0):.3f}"
            )

        def poll():
            with _STATE["lock"]:
                pending = _STATE.get("pending_cfg")
                _STATE["pending_cfg"] = None
            return pending

        eng.run(
            on_live=on_live,
            on_frame=on_frame,
            on_progress=on_progress,
            should_stop=lambda: bool(_STATE["stop"]),
            poll_config=poll,
        )
        save_checkpoint(eng.universe, out)
        writer.close()
        _STATE["message"] = f"Stop t={eng.universe.t:.2f}. Checkpoint OK."
    except Exception as exc:  # noqa: BLE001
        _STATE["error"] = f"{exc}\n{traceback.format_exc()}"
        _STATE["message"] = "Błąd symulacji."
    finally:
        _STATE["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # noqa: A003
        return

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str) -> None:
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        if path in {"/", "/index.html"}:
            self._file(_WEB / "index.html", "text/html; charset=utf-8")
            return
        if path == "/app.js":
            self._file(_WEB / "app.js", "text/javascript; charset=utf-8")
            return
        if path == "/styles.css":
            self._file(_WEB / "styles.css", "text/css; charset=utf-8")
            return
        if path == "/api/schema":
            self._json(200, schema_for_ui())
            return
        if path == "/api/status":
            self._json(
                200,
                {
                    "running": _STATE["running"],
                    "message": _STATE["message"],
                    "error": _STATE["error"],
                    "hud": _STATE["hud"],
                    "gpu": device_label(),
                    "gpu_on": gpu_enabled(),
                },
            )
            return
        if path == "/api/live":
            with _STATE["lock"]:
                live = _STATE["live"]
            self._json(200, live or {"live": True, "n": 0, "x": [], "y": [], "z": [], "c": [], "half": 12})
            return
        if path == "/api/trajectory/meta":
            meta_p = Path(_STATE["out_dir"]) / "trajectory_meta.json"
            if meta_p.exists():
                self._json(200, json.loads(meta_p.read_text(encoding="utf-8")))
            else:
                self._json(200, {"n_frames": 0})
            return
        if path == "/api/trajectory/frame":
            qs = parse_qs(u.query)
            idx = int(qs.get("i", ["0"])[0])
            frame = load_frame(_STATE["out_dir"], idx)
            self._json(200, frame or {"error": "brak klatki"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            body = {}
        if u.path == "/api/start":
            if _STATE["running"]:
                self._json(409, {"error": "już działa"})
                return
            preset = body.get("preset")
            if preset in _presets():
                cfg = _presets()[preset]
                flat = cfg.to_flat()
                flat.update({k: v for k, v in body.get("params", {}).items()})
                cfg = AppConfig.from_flat(flat)
            else:
                cfg = AppConfig.from_flat(body.get("params", {}))
            _STATE["out_dir"] = cfg.io.out_dir
            _STATE["cfg"] = cfg
            threading.Thread(
                target=_run_job, args=(cfg, bool(body.get("resume"))), daemon=True
            ).start()
            self._json(200, {"ok": True})
            return
        if u.path == "/api/stop":
            _STATE["stop"] = True
            self._json(200, {"ok": True})
            return
        if u.path == "/api/apply":
            base = _STATE.get("cfg") or AppConfig()
            cfg = merge_runtime(base, body.get("params", {}))
            with _STATE["lock"]:
                _STATE["pending_cfg"] = cfg
                _STATE["cfg"] = cfg
            self._json(200, {"ok": True})
            return
        self._json(404, {"error": "not found"})


def main(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    if not _WEB.exists():
        raise SystemExit(f"Brak frontendu: {_WEB}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Bone Studio: {url}", flush=True)
    print(f"GPU: {device_label()} · enabled={gpu_enabled()}", flush=True)
    print("SR N-body: p=γmv · bez ścian · bez warstwy społecznej.", flush=True)
    print("Presety: Galaktyka / Gromada / Burst SR.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("Stop.", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
