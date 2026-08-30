"""Serwer studia: cienkie HTTP nad silnikiem działającym w osobnym wątku.

Zasoby frontendu leżą WEWNĄTRZ pakietu (``bone/studio/web``), więc ścieżka do
nich to ``Path(__file__).parent / "web"`` i działa tak samo z repozytorium jak
z zainstalowanego wheela. Poprzednia wersja szukała katalogu ``web`` trzy
poziomy nad plikiem źródłowym, co działało wyłącznie przy instalacji
edytowalnej i wywracało się przy każdej normalnej instalacji.
"""

from __future__ import annotations

import contextlib
import json
import threading
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from bone.backends import cuda_available, cuda_name
from bone.config import PRESETS, Config, apply_runtime, preset, startup_config, ui_schema
from bone.engine import Callbacks, Engine
from bone.io import checkpoint, trajectory
from bone.io.live import empty_view, pack_view
from bone.state import State

WEB_ROOT = Path(__file__).resolve().parent / "web"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}


@dataclass
class Session:
    """Współdzielony stan między wątkiem HTTP a wątkiem symulacji."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = False
    stop_requested: bool = False
    message: str = "Gotowy."
    hint: str = ""
    error: str = ""
    view: bytes = field(default_factory=empty_view)
    diagnostics: dict[str, float] = field(default_factory=dict)
    config: Config = field(default_factory=Config)
    pending_config: Config | None = None
    backend_label: str = "—"
    out_dir: str = Config().run.out_dir
    worker: threading.Thread | None = None

    def snapshot_status(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "message": self.message,
                "hint": self.hint,
                "error": self.error,
                "diagnostics": self.diagnostics,
                "backend": self.backend_label,
                "cuda": cuda_available(),
                "cuda_name": cuda_name(),
                "out_dir": self.out_dir,
                "config": self.config.to_flat(),
            }


SESSION = Session()


def _format_message(row: dict[str, float]) -> str:
    parts = [
        f"t={row.get('t', 0):.3f}",
        f"krok={int(row.get('step', 0))}",
        f"dt={row.get('dt', 0):.2e}",
        f"⟨γ⟩={row.get('gamma_mean', 1):.4f}",
        f"β_max={row.get('beta_max', 0):.3f}",
        f"dryf E={row.get('E_drift', 0):+.2e}",
        f"wirial={row.get('virial', 0):.2f}",
    ]
    if "force_err_rms" in row:
        parts.append(f"błąd siły={row['force_err_rms']:.2%}")
    return "  ".join(parts)


def _simulation(cfg: Config, resume: bool) -> None:
    session = SESSION
    out = Path(cfg.run.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = trajectory.TrajectoryWriter(out, stride=max(1, cfg.run.point_stride))
    engine: Engine | None = None
    try:
        state: State | None = None
        if resume and checkpoint.exists(out):
            state, _ = checkpoint.load(out)
        engine = Engine(cfg, state=state)
        with session.lock:
            session.backend_label = engine.describe()
            session.message = (
                f"Start: {engine.state.n} cząstek, {engine.describe()}"
                if state is None
                else f"Wznowiono od t={engine.state.time:.3f}"
            )

        def on_view(st: State, current: Config) -> None:
            payload = pack_view(st, current)
            with session.lock:
                session.view = payload

        def on_frame(st: State, current: Config) -> None:
            writer.add(st, current)

        def on_diagnostics(row: dict[str, float]) -> None:
            hint = engine.accuracy_hint()
            with session.lock:
                session.diagnostics = row
                session.message = _format_message(row)
                session.hint = hint

        def poll() -> Config | None:
            with session.lock:
                pending = session.pending_config
                session.pending_config = None
            return pending

        def should_stop() -> bool:
            with session.lock:
                return session.stop_requested

        engine.run(
            Callbacks(
                on_view=on_view,
                on_frame=on_frame,
                on_diagnostics=on_diagnostics,
                should_stop=should_stop,
                poll_config=poll,
            )
        )
        checkpoint.save(engine.state, engine.cfg, out)
        writer.close()
        with session.lock:
            session.message = f"Zatrzymano w t={engine.state.time:.3f}. Checkpoint zapisany."
    except Exception as exc:
        with session.lock:
            session.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            session.message = "Błąd symulacji."
    finally:
        # writer.close() jest idempotentne, ale w ścieżce błędu może być już zamknięty
        with contextlib.suppress(Exception):
            writer.close()
        if engine is not None:
            engine.close()
        with session.lock:
            session.running = False
            session.stop_requested = False


class Handler(BaseHTTPRequestHandler):
    server_version = "BoneStudio/1.0"

    def log_message(self, fmt, *args):
        return

    # ------------------------------------------------------------ odpowiedzi

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # przeglądarka rozłącza się w trakcie transferu przy szybkim odpytywaniu
        with contextlib.suppress(BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _static(self, name: str) -> None:
        path = (WEB_ROOT / name).resolve()
        if not path.is_file() or WEB_ROOT.resolve() not in path.parents:
            self._json(404, {"error": f"brak zasobu: {name}"})
            return
        ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes(), ctype)

    # ------------------------------------------------------------------ GET

    def do_GET(self) -> None:
        url = urlparse(self.path)
        route = url.path

        if route in {"/", "/index.html"}:
            self._static("index.html")
        elif route in {"/app.js", "/styles.css"}:
            self._static(route.lstrip("/"))
        elif route == "/api/schema":
            self._json(200, ui_schema())
        elif route == "/api/status":
            self._json(200, SESSION.snapshot_status())
        elif route == "/api/view":
            with SESSION.lock:
                payload = SESSION.view
            self._send(200, payload, "application/octet-stream")
        elif route == "/api/trajectory":
            with SESSION.lock:
                out = SESSION.out_dir
            meta = trajectory.read_meta(out)
            self._json(200, {"n_frames": meta.get("n_frames", 0)})
        elif route == "/api/trajectory/frame":
            self._frame(url.query)
        else:
            self._json(404, {"error": "nieznana ścieżka"})

    def _frame(self, query: str) -> None:
        with SESSION.lock:
            out = SESSION.out_dir
        index = int((parse_qs(query).get("i", ["0"])[0]) or 0)
        frame = trajectory.load_frame(out, index)
        if frame is None:
            self._send(200, empty_view(), "application/octet-stream")
            return
        positions, shades, time_value = frame
        from bone.io.live import HEADER, MAGIC

        half = float(np.abs(positions).max() * 1.05 + 1.0) if positions.size else 1.0
        body = b"".join(
            (
                HEADER.pack(MAGIC, int(positions.shape[0]), half, time_value),
                np.ascontiguousarray(positions, dtype=np.float32).tobytes(),
                np.ascontiguousarray(shades, dtype=np.float32).tobytes(),
            )
        )
        self._send(200, body, "application/octet-stream")

    # ----------------------------------------------------------------- POST

    def do_POST(self) -> None:
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._json(400, {"error": "ciało żądania nie jest poprawnym JSON-em"})
            return

        if url.path == "/api/start":
            self._start(body)
        elif url.path == "/api/stop":
            with SESSION.lock:
                SESSION.stop_requested = True
            self._json(200, {"ok": True})
        elif url.path == "/api/apply":
            self._apply(body)
        else:
            self._json(404, {"error": "nieznana ścieżka"})

    def _start(self, body: dict) -> None:
        params = body.get("params") or {}
        name = body.get("preset")
        base = preset(name) if name in PRESETS else Config()
        try:
            cfg = startup_config(base, params)
        except (TypeError, ValueError) as exc:
            self._json(400, {"error": f"zła konfiguracja: {exc}"})
            return

        # rezerwacja slotu i start wątku pod tym samym zamkiem — bez tego dwa
        # szybkie żądania potrafiły uruchomić dwie symulacje na jeden katalog
        with SESSION.lock:
            if SESSION.running:
                self._json(409, {"error": "symulacja już działa"})
                return
            SESSION.running = True
            SESSION.stop_requested = False
            SESSION.error = ""
            SESSION.message = "Przygotowanie…"
            SESSION.hint = ""
            SESSION.config = cfg
            SESSION.out_dir = cfg.run.out_dir
            SESSION.diagnostics = {}
            SESSION.view = empty_view()
            worker = threading.Thread(
                target=_simulation, args=(cfg, bool(body.get("resume"))), daemon=True
            )
            SESSION.worker = worker
        worker.start()
        self._json(200, {"ok": True, "config": cfg.to_flat()})

    def _apply(self, body: dict) -> None:
        params = body.get("params") or {}
        with SESSION.lock:
            base = SESSION.pending_config or SESSION.config
            try:
                cfg = apply_runtime(base, params)
            except (TypeError, ValueError) as exc:
                self._json(400, {"error": f"zła wartość: {exc}"})
                return
            SESSION.pending_config = cfg
            SESSION.config = cfg
        self._json(200, {"ok": True, "config": cfg.to_flat()})


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"Brak zasobów frontendu w {WEB_ROOT}")
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"Bone Studio → {url}", flush=True)
    print(f"CUDA: {cuda_name() if cuda_available() else 'niedostępna'}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nZatrzymano.", flush=True)
    finally:
        with SESSION.lock:
            SESSION.stop_requested = True
        httpd.server_close()
