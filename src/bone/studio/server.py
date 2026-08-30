"""Serwer studia: cienkie HTTP nad silnikiem działającym w osobnym wątku.

Zasoby frontendu leżą WEWNĄTRZ pakietu (``bone/studio/web``), więc ścieżka do
nich to ``Path(__file__).parent / "web"`` i działa tak samo z repozytorium jak
z zainstalowanego wheela. Poprzednia wersja szukała katalogu ``web`` trzy
poziomy nad plikiem źródłowym, co działało wyłącznie przy instalacji
edytowalnej i wywracało się przy każdej normalnej instalacji.

Warstwa ``dispatch_get`` / ``dispatch_post`` jest wspólna dla lokalnego
``ThreadingHTTPServer`` i ASGI na Vercelu — ten sam kontrakt HTTP, dwa hosty.
"""

from __future__ import annotations

import contextlib
import json
import os
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

#: Na Hobby/Fluid funkcja ma budżet rzędu minut; 120k cząstek O(N²) go zjada.
_SERVERLESS_MAX_N = 8_000
_SERVERLESS_EXACT_MAX_N = 2_000


@dataclass(frozen=True)
class ApiResponse:
    status: int
    body: bytes
    content_type: str

    @classmethod
    def json(cls, status: int, payload: dict) -> ApiResponse:
        return cls(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    @classmethod
    def raw(cls, status: int, body: bytes, content_type: str) -> ApiResponse:
        return cls(status, body, content_type)


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
                "host": "vercel" if on_serverless() else "local",
            }


SESSION = Session()

#: Ile czekać na zejście poprzedniego biegu przy restarcie. Jeden krok przy
#: 200 tys. cząstek na procesorze trwa rzędu sekundy, więc próg musi być
#: wielokrotnością tego, a nie ułamkiem. Na Vercelu budżet requestu jest
#: krótszy — nie wolno zjadać go całym join'em.
_STOP_TIMEOUT = 20.0
_STOP_TIMEOUT_SERVERLESS = 5.0


def on_serverless() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))


def stop_timeout() -> float:
    return _STOP_TIMEOUT_SERVERLESS if on_serverless() else _STOP_TIMEOUT


def apply_host_limits(cfg: Config) -> Config:
    """Dostosuj bieg do ograniczeń hosta (filesystem, czas, pamięć)."""
    if not on_serverless():
        return cfg
    n = min(int(cfg.spawn.n_particles), _SERVERLESS_MAX_N)
    updates: dict = {
        "out_dir": "/tmp/bone-runs",
        "device": "cpu",
        "n_particles": n,
    }
    if cfg.solver.backend == "exact" and n > _SERVERLESS_EXACT_MAX_N:
        updates["backend"] = "mesh"
        updates["n_particles"] = min(n, _SERVERLESS_EXACT_MAX_N * 4)
    return cfg.replace_flat(updates)


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
            # bierzemy wyłącznie stan: konfigurację wznawianego biegu rozstrzygnął
            # już `_resume_config`, żeby dało się ją pokazać w panelu
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


def _static(name: str) -> ApiResponse:
    path = (WEB_ROOT / name).resolve()
    if not path.is_file() or WEB_ROOT.resolve() not in path.parents:
        return ApiResponse.json(404, {"error": f"brak zasobu: {name}"})
    ctype = _CONTENT_TYPES.get(path.suffix, "application/octet-stream")
    return ApiResponse.raw(200, path.read_bytes(), ctype)


def _preset(query: str) -> ApiResponse:
    """Zwróć konfigurację presetu, nie uruchamiając niczego.

    Preset jest wczytywany do panelu, a bieg startuje osobnym kliknięciem.
    Dzięki temu widać, co się właściwie odpali, a jedynym źródłem prawdy o
    konfiguracji pozostaje panel — nie ma dwóch kanałów, które mogą się
    rozjechać.
    """
    name = parse_qs(query).get("name", [""])[0]
    if name not in PRESETS:
        return ApiResponse.json(404, {"error": f"nieznany preset: {name!r}"})
    cfg = apply_host_limits(preset(name))
    return ApiResponse.json(
        200, {"name": name, "label": PRESETS[name][0], "config": cfg.to_flat()}
    )


def _frame(query: str) -> ApiResponse:
    with SESSION.lock:
        out = SESSION.out_dir
    index = int((parse_qs(query).get("i", ["0"])[0]) or 0)
    frame = trajectory.load_frame(out, index)
    if frame is None:
        return ApiResponse.raw(200, empty_view(), "application/octet-stream")
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
    return ApiResponse.raw(200, body, "application/octet-stream")


def stop_and_wait(timeout: float | None = None) -> bool:
    """Poproś bieżący bieg o zatrzymanie i poczekaj, aż wątek naprawdę zejdzie.

    Czekanie jest tu konieczne, bo katalog wyjściowy ma jednego właściciela:
    dwa wątki piszące tę samą trajektorię i checkpoint dałyby ciche
    przemieszanie danych. Zamek jest zwolniony na czas `join`, inaczej wątek
    symulacji nie mógłby zaktualizować statusu i doszłoby do zakleszczenia.
    """
    limit = stop_timeout() if timeout is None else timeout
    with SESSION.lock:
        if not SESSION.running:
            return True
        SESSION.stop_requested = True
        worker = SESSION.worker
    if worker is not None:
        worker.join(limit)
    with SESSION.lock:
        return not SESSION.running


def _resume_config(requested: Config) -> tuple[Config | None, str]:
    """Konfiguracja wznawianego biegu pochodzi z checkpointu, nie z panelu.

    Zapisany stan powstał pod konkretnymi prawami fizyki. Branie parametrów
    z panelu znaczyło, że wznowiony układ dostawał po cichu inne G i c niż
    te, z którymi był zapisany: bieg wyglądał na kontynuację, a był innym
    doświadczeniem — i nic tego nie zdradzało, bo panel pokazywał swoje
    wartości, a wykres energii liczy dryf od nowego punktu odniesienia.

    Katalog wyjściowy zostaje ten, w którym checkpoint naprawdę leży: ścieżka
    zapisana w środku mogła przestać się zgadzać, gdy katalog przeniesiono.

    Zwraca ``(None, "")``, gdy nie ma czego wznawiać, oraz konfigurację
    z panelu wraz z uwagą, gdy stan jest, ale nie zapisano do niego parametrów.
    """
    out_dir = requested.run.out_dir
    if not checkpoint.exists(out_dir):
        return None, ""
    saved = checkpoint.load_config(out_dir)
    if saved is None:
        return requested, (
            f"Checkpoint w {out_dir} nie ma zapisanej konfiguracji "
            "— wznowiono z parametrami z panelu."
        )
    return saved.replace_flat({"out_dir": out_dir}), ""


def _start(body: dict) -> ApiResponse:
    params = body.get("params") or {}
    name = body.get("preset")

    # Preset i params to dwa opisy tej samej rzeczy, więc jeden musi wygrać.
    # Wygrywa preset. Wcześniej params nakładały się na preset, a ponieważ
    # frontend wysyłał KOMPLET kluczy z wartościami domyślnymi, każdy preset
    # był wymazywany do domyślnej konfiguracji — klik działał, efekt był
    # zerowy. Kto chce preset z poprawką, wczytuje go przez /api/preset,
    # zmienia w panelu i startuje bez pola `preset`.
    if name in PRESETS:
        cfg = preset(name)
    else:
        try:
            cfg = startup_config(Config(), params)
        except (TypeError, ValueError) as exc:
            return ApiResponse.json(400, {"error": f"zła konfiguracja: {exc}"})

    cfg = apply_host_limits(cfg)

    if body.get("restart") and not stop_and_wait():
        return ApiResponse.json(
            409, {"error": "poprzedni bieg nie zatrzymał się w czasie — spróbuj ponownie"}
        )

    # Kolejność jest tu istotna: checkpoint czytamy PO zatrzymaniu poprzedniego
    # biegu, bo to on zapisuje go na wyjściu. Odwrotna kolejność dałaby stan
    # nowszy niż wczytana do niego konfiguracja.
    note = ""
    resume = bool(body.get("resume"))
    if resume:
        requested_dir = cfg.run.out_dir
        resumed, note = _resume_config(cfg)
        if resumed is None:
            return ApiResponse.json(
                409,
                {"error": f"brak checkpointu w {requested_dir} — nie ma czego wznawiać"},
            )
        cfg = apply_host_limits(resumed)

    # rezerwacja slotu i start wątku pod tym samym zamkiem — bez tego dwa
    # szybkie żądania potrafiły uruchomić dwie symulacje na jeden katalog
    with SESSION.lock:
        if SESSION.running:
            return ApiResponse.json(
                409, {"error": "symulacja już działa — użyj Restart albo Stop"}
            )
        SESSION.running = True
        SESSION.stop_requested = False
        SESSION.error = ""
        SESSION.message = "Przygotowanie…"
        SESSION.hint = ""
        SESSION.config = cfg
        SESSION.out_dir = cfg.run.out_dir
        SESSION.diagnostics = {}
        SESSION.view = empty_view()
        worker = threading.Thread(target=_simulation, args=(cfg, resume), daemon=True)
        SESSION.worker = worker
    worker.start()
    return ApiResponse.json(
        200, {"ok": True, "config": cfg.to_flat(), "resumed": resume, "note": note}
    )


def _apply(body: dict) -> ApiResponse:
    params = body.get("params") or {}
    with SESSION.lock:
        base = SESSION.pending_config or SESSION.config
        try:
            cfg = apply_runtime(base, params)
        except (TypeError, ValueError) as exc:
            return ApiResponse.json(400, {"error": f"zła wartość: {exc}"})
        if on_serverless():
            cfg = apply_host_limits(cfg)
        SESSION.pending_config = cfg
        SESSION.config = cfg
    return ApiResponse.json(200, {"ok": True, "config": cfg.to_flat()})


def dispatch_get(path: str, query: str = "") -> ApiResponse:
    if path in {"/", "/index.html"}:
        return _static("index.html")
    if path in {"/app.js", "/styles.css"}:
        return _static(path.lstrip("/"))
    if path == "/api/schema":
        return ApiResponse.json(200, ui_schema())
    if path == "/api/preset":
        return _preset(query)
    if path == "/api/status":
        return ApiResponse.json(200, SESSION.snapshot_status())
    if path == "/api/view":
        with SESSION.lock:
            payload = SESSION.view
        return ApiResponse.raw(200, payload, "application/octet-stream")
    if path == "/api/trajectory":
        with SESSION.lock:
            out = SESSION.out_dir
        meta = trajectory.read_meta(out)
        return ApiResponse.json(200, {"n_frames": meta.get("n_frames", 0)})
    if path == "/api/trajectory/frame":
        return _frame(query)
    return ApiResponse.json(404, {"error": "nieznana ścieżka"})


def dispatch_post(path: str, body: dict) -> ApiResponse:
    if path == "/api/start":
        return _start(body)
    if path == "/api/stop":
        # Odpowiedź mówi, czy było co zatrzymywać. Bez tego klik w Stop na
        # zatrzymanym układzie wygląda dokładnie jak klik zignorowany.
        with SESSION.lock:
            was_running = SESSION.running
            SESSION.stop_requested = was_running
        return ApiResponse.json(200, {"ok": True, "was_running": was_running})
    if path == "/api/apply":
        return _apply(body)
    return ApiResponse.json(404, {"error": "nieznana ścieżka"})


class Handler(BaseHTTPRequestHandler):
    server_version = "BoneStudio/1.0"

    def log_message(self, fmt, *args):
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # przeglądarka rozłącza się w trakcie transferu przy szybkim odpytywaniu
        with contextlib.suppress(BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            self.wfile.write(body)

    def _reply(self, response: ApiResponse) -> None:
        self._send(response.status, response.body, response.content_type)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        self._reply(dispatch_get(url.path, url.query))

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._reply(ApiResponse.json(400, {"error": "ciało żądania nie jest poprawnym JSON-em"}))
            return
        url = urlparse(self.path)
        self._reply(dispatch_post(url.path, body))

    # Zachowane pod testy i ewentualne bezpośrednie wywołania.
    _stop_and_wait = staticmethod(stop_and_wait)
    _resume_config = staticmethod(_resume_config)


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
