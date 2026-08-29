"""Bone Studio — LIVE + panel stref + hot-apply parametrów runtime."""

from __future__ import annotations

import json
import os
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# matplotlib w wątku symulacji nie może używać Tk (crash Tcl_AsyncDelete)
os.environ.setdefault("MPLBACKEND", "Agg")

from bone.studio_page import build_studio_html
from bone.trajectory import load_trajectory_frame, load_trajectory_meta
from bone.ui_schema import (
    config_from_params,
    default_param_values,
    merge_runtime_params,
    orbit_then_singularity_preset,
    panel_boot_schema,
    stable_orbit_preset,
)
from bone.viewer import load_live_frame, pack_live_frame

_STATE: dict = {
    "running": False,
    "stop_requested": False,
    "message": "Gotowy.",
    "error": None,
    "out_dir": Path("out"),
    "step": 0,
    "total": 0,
    "t": 0.0,
    "abs_step": 0,
    "live": None,
    "live_lock": threading.Lock(),
    "active_cfg": None,
    "pending_cfg": None,
    "cfg_lock": threading.Lock(),
    "applied_note": "",
}


def _set_live_from_universe(universe, point_stride: int = 4) -> None:
    colors = None
    neigh = getattr(universe, "neighbors", None)
    if neigh is not None and getattr(neigh, "pairs", None) is not None:
        try:
            colors = neigh.pair_degree(
                universe.n, universe.config.spacing * 1.6, universe.positions
            )
        except Exception:  # noqa: BLE001
            colors = None
    payload = pack_live_frame(
        universe.positions,
        universe.velocities,
        universe.t,
        universe.step,
        point_stride=point_stride,
        colors=colors,
    )
    with _STATE["live_lock"]:
        _STATE["live"] = payload
        _STATE["t"] = payload["t"]
        _STATE["abs_step"] = payload["step"]


def _live_payload() -> dict:
    with _STATE["live_lock"]:
        live = _STATE["live"]
    if live is not None:
        return live
    disk = load_live_frame(Path(_STATE["out_dir"]), point_stride=2)
    if disk is not None:
        return disk
    return {
        "live": True,
        "t": 0.0,
        "step": 0,
        "half": 12.0,
        "n": 0,
        "x": [],
        "y": [],
        "z": [],
        "c": [],
    }


def _poll_config():
    with _STATE["cfg_lock"]:
        pending = _STATE["pending_cfg"]
        if pending is None:
            return None
        _STATE["pending_cfg"] = None
        _STATE["active_cfg"] = pending
        return pending


def _run_job(params: dict) -> None:
    from bone.simulate import run_simulation

    resume = bool(params.pop("continue", False) or params.pop("resume", False))
    _STATE["running"] = True
    _STATE["stop_requested"] = False
    _STATE["error"] = None
    _STATE["step"] = 0
    _STATE["t"] = 0.0
    _STATE["applied_note"] = ""
    with _STATE["live_lock"]:
        _STATE["live"] = None
    with _STATE["cfg_lock"]:
        _STATE["pending_cfg"] = None
    _STATE["message"] = "Kontynuacja..." if resume else "LIVE startuje..."
    try:
        cfg = config_from_params(params)
        with _STATE["cfg_lock"]:
            _STATE["active_cfg"] = cfg
        _STATE["out_dir"] = Path(cfg.out_dir)
        _STATE["total"] = cfg.steps
        limit = "∞" if cfg.steps <= 0 else str(cfg.steps)
        _STATE["message"] = (
            f"{'Kontynuacja' if resume else 'LIVE'}: steps={limit} G={cfg.G}"
        )

        def on_live(universe) -> None:
            _set_live_from_universe(universe, point_stride=4)

        def on_progress(step: int, total: int, universe, row: dict) -> None:
            _STATE["step"] = step
            _STATE["total"] = total
            tot = "∞" if cfg.steps <= 0 else str(total)
            note = _STATE.get("applied_note") or ""
            _STATE["message"] = (
                f"LIVE t={universe.t:.2f} krok {step}/{tot} "
                f"G={universe.config.G:.3f} "
                f"Gini={row.get('gini', 0):.3f} "
                f"|L|={row.get('L_mag', 0):.2f} "
                f"flat={row.get('flattening', 1):.3f} "
                f"r½={row.get('collapse_ratio', 1):.3f}"
                + (f" · {note}" if note else "")
            )
            _STATE["live_gini"] = float(row.get("gini", 0.0))
            _STATE["live_top_w"] = float(row.get("top_wealth_frac", 0.0))
            _STATE["live_top_m"] = float(row.get("top_mass_frac", 0.0))
            _STATE["live_L"] = float(row.get("L_mag", 0.0))
            _STATE["live_flat"] = float(row.get("flattening", 1.0))

        run_simulation(
            cfg,
            progress_every=max(10, (cfg.steps // 80) if cfg.steps > 0 else 40),
            resume=resume,
            on_progress=on_progress,
            on_live=on_live,
            should_stop=lambda: bool(_STATE["stop_requested"]),
            poll_config=_poll_config,
        )
        if _STATE["stop_requested"]:
            _STATE["message"] = f"Zatrzymano t={_STATE['t']:.2f}. Checkpoint OK."
        else:
            _STATE["message"] = f"Gotowe t={_STATE['t']:.2f}. Mozesz Kontynuowac."
    except Exception as exc:  # noqa: BLE001
        _STATE["error"] = f"{exc}\n{traceback.format_exc()}"
        _STATE["message"] = "Blad symulacji."
    finally:
        _STATE["running"] = False
        _STATE["stop_requested"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html", "/cube_view.html"):
            html = build_studio_html(
                params=default_param_values(),
                studio_mode=True,
            )
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/live":
            self._json(200, _live_payload())
            return
        if path == "/api/trajectory/meta":
            out = Path(_STATE.get("out_dir") or "out")
            meta = load_trajectory_meta(out)
            # ogranicz rozmiar odpowiedzi
            times = meta.get("times") or []
            if len(times) > 5000:
                meta = dict(meta)
                meta["times"] = times[:: max(1, len(times) // 2000)]
                meta["times_downsampled"] = True
            self._json(200, meta)
            return
        if path == "/api/trajectory/frame":
            qs = parse_qs(urlparse(self.path).query)
            try:
                idx = int(qs.get("i", ["0"])[0])
            except ValueError:
                self._json(400, {"error": "zly indeks"})
                return
            out = Path(_STATE.get("out_dir") or "out")
            frame = load_trajectory_frame(out, idx, point_stride=2)
            if frame is None:
                self._json(404, {"error": "brak klatki"})
                return
            self._json(200, frame)
            return
        if path == "/api/status":
            from bone.gpu import gpu_info

            with _STATE["cfg_lock"]:
                acfg = _STATE["active_cfg"]
            gi = gpu_info()
            self._json(
                200,
                {
                    "running": _STATE["running"],
                    "stop_requested": _STATE["stop_requested"],
                    "message": _STATE["message"],
                    "error": _STATE["error"],
                    "step": _STATE.get("step", 0),
                    "total": _STATE.get("total", 0),
                    "t": _STATE.get("t", 0.0),
                    "abs_step": _STATE.get("abs_step", 0),
                    "applied": _STATE.get("applied_note", ""),
                    "active_G": float(acfg.G) if acfg else None,
                    "active_circularize": float(acfg.circularize_rate) if acfg else None,
                    "gini": _STATE.get("live_gini"),
                    "top_wealth": _STATE.get("live_top_w"),
                    "top_mass": _STATE.get("live_top_m"),
                    "L_mag": _STATE.get("live_L"),
                    "flattening": _STATE.get("live_flat"),
                    "gpu": gi.get("device"),
                    "gpu_enabled": gi.get("enabled"),
                    "live": True,
                },
            )
            return
        if path == "/api/schema":
            from bone.gpu import gpu_info

            self._json(
                200,
                {
                    "params": default_param_values(),
                    "panel": panel_boot_schema(),
                    "gpu": gpu_info(),
                },
            )
            return
        if path == "/api/presets":
            self._json(
                200,
                {
                    "stable_sphere": stable_orbit_preset(1),
                    "stable_donut": stable_orbit_preset(8),
                    "singularity_sphere": orbit_then_singularity_preset(1),
                    "singularity_donut": orbit_then_singularity_preset(8),
                },
            )
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            params = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError:
            self._json(400, {"error": "Zly JSON"})
            return

        if path == "/api/stop":
            if not _STATE["running"]:
                self._json(200, {"ok": True, "message": "Nic nie biegnie"})
                return
            _STATE["stop_requested"] = True
            _STATE["message"] = "Stop zlecony..."
            self._json(200, {"ok": True, "message": "Stop zlecony"})
            return

        if path == "/api/params":
            # hot-apply runtime podczas biegu; poza biegiem tylko aktualizuje active_cfg
            with _STATE["cfg_lock"]:
                base = _STATE["active_cfg"] or config_from_params(params)
                merged = merge_runtime_params(base, params)
                if _STATE["running"]:
                    _STATE["pending_cfg"] = merged
                    _STATE["applied_note"] = (
                        f"Zastosowano G={merged.G:.3f} "
                        f"×{merged.sim_speed} "
                        f"ineq={merged.inequality_drive:.2f}"
                    )
                    msg = "Parametry runtime w kolejce — wejdą od następnego kroku"
                else:
                    _STATE["active_cfg"] = merged
                    _STATE["applied_note"] = "Zapisano (bieg nieaktywny — użyj Uruchom)"
                    msg = _STATE["applied_note"]
            self._json(
                200,
                {
                    "ok": True,
                    "message": msg,
                    "G": merged.G,
                    "circularize_rate": merged.circularize_rate,
                    "inequality_drive": merged.inequality_drive,
                    "greed_bias": merged.greed_bias,
                    "running": _STATE["running"],
                },
            )
            return

        if path != "/api/run":
            self._json(404, {"error": "not found"})
            return
        if _STATE["running"]:
            self._json(409, {"error": "Symulacja juz trwa — Stop albo Zastosuj parametry"})
            return
        thread = threading.Thread(target=_run_job, args=(params,), daemon=True)
        thread.start()
        self._json(200, {"ok": True})


def main(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    from bone.gpu import gpu_info

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    gi = gpu_info()
    print(f"Bone Studio: {url}", flush=True)
    print(
        f"GPU: {gi.get('device')} · enabled={gi.get('enabled')} "
        f"· torch={gi.get('torch')}",
        flush=True,
    )
    print("Strefy · Emergencja · LIVE · Zastosuj = hot-apply runtime.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStop.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
