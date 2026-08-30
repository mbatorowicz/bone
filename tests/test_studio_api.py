"""Kontrakt warstwy HTTP studia i reakcji silnika na kliknięcia.

Te testy pilnują rzeczy, które psuły się cicho: przycisk wysyłał żądanie, serwer
odpowiadał 200, a efekt był zerowy albo odwrotny do zamierzonego. Dlatego
sprawdzana jest KONFIGURACJA, z którą bieg faktycznie startuje, a nie samo to,
że żądanie przeszło.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from bone.config import PRESETS, Config, preset, startup_config
from bone.engine import Callbacks, Engine
from bone.io import checkpoint
from bone.io.trajectory import TrajectoryWriter, load_frame, read_meta
from bone.spawn import make_state
from bone.studio.server import SESSION, Handler

#: Konfiguracja startująca w ułamku sekundy — testy dotyczą przepływu sterowania,
#: nie fizyki, więc liczba cząstek ma być najmniejsza z możliwych.
LIGHT = {"n_particles": 200, "backend": "exact", "device": "cpu", "time_scale": 1}


def _light_params(**overrides: object) -> dict:
    params = Config().to_flat()
    params.update(LIGHT)
    params.update(overrides)
    return params


# --------------------------------------------------------------------- osprzęt


def _reset_session() -> None:
    Handler._stop_and_wait(30.0)
    with SESSION.lock:
        SESSION.running = False
        SESSION.stop_requested = False
        SESSION.error = ""
        SESSION.message = "Gotowy."
        SESSION.hint = ""
        SESSION.diagnostics = {}
        SESSION.config = Config()
        SESSION.pending_config = None
        SESSION.worker = None


@pytest.fixture
def studio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # katalog wyjściowy jest ścieżką względną z konfiguracji, więc bieg testowy
    # musi mieć własny katalog roboczy — inaczej pisałby do runs/ w repozytorium
    monkeypatch.chdir(tmp_path)
    _reset_session()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        _reset_session()
        server.shutdown()
        server.server_close()
        thread.join(5.0)


def _request(url: str, payload: dict | None = None) -> tuple[int, dict]:
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ----------------------------------------------------------------- presety


def test_preset_endpoint_returns_config_without_starting_anything(studio: str):
    status, body = _request(f"{studio}/api/preset?name=merger")
    assert status == 200
    assert body["config"] == preset("merger").to_flat()
    assert body["label"] == PRESETS["merger"][0]
    with SESSION.lock:
        assert not SESSION.running


def test_unknown_preset_is_rejected(studio: str):
    status, body = _request(f"{studio}/api/preset?name=nie_ma_takiego")
    assert status == 404
    assert "error" in body


def test_schema_embeds_full_preset_config(studio: str):
    """Chip startuje z konfiguracją ze schematu, bez drugiego żądania.

    Gdyby w schemacie było tylko id i etykieta, klik wymagałby /api/preset.
    Stary serwer tego endpointu nie ma, więc przycisk wyglądałby na martwy
    dopóki ktoś nie zrestartuje studia — a to dokładnie zgłoszony objaw.
    """
    _, schema = _request(f"{studio}/api/schema")
    by_id = {p["id"]: p for p in schema["presets"]}
    assert set(by_id) == set(PRESETS)
    for name, entry in by_id.items():
        assert entry["config"] == preset(name).to_flat()


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_panel_can_reproduce_every_preset_exactly(studio: str, name: str):
    """Warunek poprawności dwuetapowego wczytywania presetu.

    Chip wkłada konfigurację do panelu, a start wysyła ją z powrotem jako zwykłe
    parametry. Jeżeli ta droga w tę i z powrotem czegokolwiek nie przenosi, preset
    startuje inny, niż pokazuje panel — czyli dokładnie ta klasa błędu, którą
    naprawiamy, tylko przeniesiona w inne miejsce.
    """
    _, body = _request(f"{studio}/api/preset?name={name}")
    assert startup_config(Config(), body["config"]) == preset(name)


def test_start_with_only_preset_id_uses_that_preset(studio: str):
    """Chip wysyła samo id — bez params z panelu.

    To jest jedyna droga, która działa i na nowym, i na starym serwerze:
    stary nakładał params na preset, więc komplet kluczy z panelu wymazywał
    każdy chip do domyślnej kuli.
    """
    status, body = _request(f"{studio}/api/start", {"preset": "precision"})
    assert status == 200
    assert body["config"] == preset("precision").to_flat()


def test_params_cannot_silently_wipe_a_preset(studio: str):
    """Regresja: preset i params to dwa opisy tej samej rzeczy, wygrywa preset.

    Wcześniej params nakładały się na preset, a frontend wysyłał komplet kluczy
    z wartościami domyślnymi — każdy preset był więc wymazywany do domyślnej
    konfiguracji. Klik działał, odpowiedź była 200, a startowała domyślna kula.
    """
    wiping = _light_params(geometry="ball", n_particles=200, c=99.0)
    status, body = _request(f"{studio}/api/start", {"preset": "precision", "params": wiping})
    assert status == 200
    assert body["config"] == preset("precision").to_flat()
    assert body["config"]["geometry"] == "plummer"
    assert body["config"]["n_particles"] == 2_000


def test_two_step_flow_starts_exactly_the_loaded_preset(studio: str):
    """Pełna droga, którą chodzi panel: schemat → preset → start bez pola `preset`.

    Tu właśnie mieszkał błąd. Frontend wysyłał komplet parametrów razem z nazwą
    presetu i preset przegrywał, więc każdy chip startował domyślną konfigurację.
    Teraz chip wypełnia panel, a start wysyła już tylko parametry.
    """
    _, schema = _request(f"{studio}/api/schema")
    params = dict(schema["defaults"])

    _, loaded = _request(f"{studio}/api/preset?name=precision")
    params = dict(loaded["config"])

    status, body = _request(f"{studio}/api/start", {"params": params})
    assert status == 200
    assert body["config"] == preset("precision").to_flat()
    assert _wait_for_running(True)
    with SESSION.lock:
        assert SESSION.config == preset("precision")


# ------------------------------------------------------- start, stop, restart


def _wait_for_running(expected: bool, timeout: float = 20.0) -> bool:
    tick = threading.Event()
    for _ in range(int(timeout / 0.05)):
        with SESSION.lock:
            if SESSION.running is expected:
                return True
        tick.wait(0.05)
    return False


def test_second_start_without_restart_is_refused_with_a_reason(studio: str):
    status, _ = _request(f"{studio}/api/start", {"params": _light_params()})
    assert status == 200
    assert _wait_for_running(True)

    status, body = _request(f"{studio}/api/start", {"params": _light_params()})
    assert status == 409
    assert "Restart" in body["error"] or "Stop" in body["error"]


def test_restart_replaces_the_running_simulation(studio: str):
    _request(f"{studio}/api/start", {"params": _light_params()})
    assert _wait_for_running(True)
    with SESSION.lock:
        first = SESSION.worker

    status, _ = _request(
        f"{studio}/api/start", {"params": _light_params(c=41.0), "restart": True}
    )
    assert status == 200
    with SESSION.lock:
        assert SESSION.running
        assert SESSION.worker is not first
        assert SESSION.config.physics.c == 41.0
    assert first is not None and not first.is_alive()


# ---------------------------------------------------------------- wznawianie


def _make_checkpoint(out_dir: Path, **overrides: object) -> Config:
    cfg = Config().replace_flat({**LIGHT, "out_dir": str(out_dir), **overrides})
    engine = Engine(cfg)
    engine.advance(2)
    checkpoint.save(engine.state, cfg, out_dir)
    engine.close()
    return cfg


def test_resume_takes_configuration_from_the_checkpoint_not_the_panel(
    studio: str, tmp_path: Path
):
    """Wznowienie musi być kontynuacją zapisanego biegu, nie nowym doświadczeniem.

    Stan na dysku powstał pod konkretnymi G i c. Wcześniej wznawianie łączyło ten
    stan z parametrami z panelu, więc układ dostawał po cichu inną fizykę, a nic
    tego nie zdradzało — panel pokazywał swoje liczby i wyglądało to jak
    kontynuacja.
    """
    out = tmp_path / "saved_run"
    saved = _make_checkpoint(out, c=41.0, G=0.33)

    panel = _light_params(out_dir=str(out), c=7.0, G=1.5)
    status, body = _request(f"{studio}/api/start", {"params": panel, "resume": True})

    assert status == 200
    assert body["resumed"] is True
    assert body["config"]["c"] == 41.0
    assert body["config"]["G"] == 0.33
    assert Config.from_flat(body["config"]) == saved
    assert _wait_for_running(True)
    with SESSION.lock:
        assert SESSION.config.physics.c == 41.0


def test_resume_without_checkpoint_says_so_instead_of_starting_fresh(
    studio: str, tmp_path: Path
):
    empty_dir = tmp_path / "nothing_here"
    status, body = _request(
        f"{studio}/api/start", {"params": _light_params(out_dir=str(empty_dir)), "resume": True}
    )
    assert status == 409
    assert "nie ma czego wznawiać" in body["error"]
    with SESSION.lock:
        assert not SESSION.running


def test_resume_keeps_the_directory_it_was_found_in(studio: str, tmp_path: Path):
    """Ścieżka zapisana w checkpoincie mogła przestać się zgadzać."""
    origin = tmp_path / "where_it_was_written"
    _make_checkpoint(origin)
    moved = tmp_path / "where_it_lies_now"
    origin.rename(moved)

    _, body = _request(
        f"{studio}/api/start",
        {"params": _light_params(out_dir=str(moved)), "resume": True},
    )
    assert body["config"]["out_dir"] == str(moved)


def test_resume_reports_a_checkpoint_without_saved_configuration(studio: str, tmp_path: Path):
    out = tmp_path / "state_without_config"
    _make_checkpoint(out)
    (out / "config.json").unlink()

    _, body = _request(
        f"{studio}/api/start", {"params": _light_params(out_dir=str(out), c=7.0), "resume": True}
    )
    assert body["resumed"] is True
    assert "nie ma zapisanej konfiguracji" in body["note"]
    # skoro nie ma nic lepszego, jawnie użyto panelu — i to jest powiedziane
    assert body["config"]["c"] == 7.0


def test_stop_reports_whether_there_was_anything_to_stop(studio: str):
    status, body = _request(f"{studio}/api/stop", {})
    assert status == 200
    assert body["was_running"] is False

    _request(f"{studio}/api/start", {"params": _light_params()})
    assert _wait_for_running(True)
    status, body = _request(f"{studio}/api/stop", {})
    assert status == 200
    assert body["was_running"] is True
    assert _wait_for_running(False)


# ------------------------------------------------- reakcja silnika na Stop


def test_advance_checks_stop_after_every_step():
    engine = Engine(Config().replace_flat(LIGHT))
    checks = 0

    def should_stop() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    engine.advance(40, should_stop)
    engine.close()
    assert engine.state.step == 2


def test_stop_is_honoured_within_one_step_at_maximum_time_scale():
    """Bez tego Stop czekał na całą paczkę `time_scale` kroków.

    Przy suwaku tempa na 40 i dużym układzie to kilka sekund ciszy po kliknięciu,
    czyli przycisk nie do odróżnienia od zepsutego.
    """
    engine = Engine(Config().replace_flat({**LIGHT, "time_scale": 40}))
    checks = 0

    def should_stop() -> bool:
        nonlocal checks
        checks += 1
        return checks > 3

    engine.run(Callbacks(should_stop=should_stop))
    engine.close()
    assert engine.state.step == 3


# ---------------------------------------------------- klatki dla odtwarzacza


def test_frames_become_readable_before_the_chunk_fills(tmp_path: Path):
    """Odtwarzacz pokazywał 0 / 0 przez 1280 kroków biegu.

    Indeks trafiał na dysk wyłącznie przy zapełnieniu 64-klatkowej paczki, a
    klatki powstają co kilkadziesiąt kroków. Zapis wyzwalany też czasem sprawia,
    że suwak ma czym się zapełniać od pierwszych sekund.
    """
    cfg = Config().replace_flat(LIGHT)
    state = make_state(cfg)
    writer = TrajectoryWriter(tmp_path, chunk_size=64, flush_interval=0.0)
    writer.add(state, cfg)

    assert read_meta(tmp_path)["n_frames"] == 1
    frame = load_frame(tmp_path, 0)
    assert frame is not None
    positions, _, _ = frame
    assert positions.shape == (state.n, 3)
    writer.close()


def test_full_chunk_still_flushes_when_time_has_not_passed(tmp_path: Path):
    cfg = Config().replace_flat(LIGHT)
    state = make_state(cfg)
    writer = TrajectoryWriter(tmp_path, chunk_size=2, flush_interval=1e6)
    writer.add(state, cfg)
    assert read_meta(tmp_path)["n_frames"] == 0
    writer.add(state, cfg)
    assert read_meta(tmp_path)["n_frames"] == 2
    writer.close()


# ------------------------------------------------------------ kaskada CSS


def test_hidden_attribute_beats_layout_rules():
    """Reguła autorska z `display` unieważniała atrybut `hidden`.

    Pasek odtwarzania był przez to widoczny również w trybie na żywo, gdzie jego
    przyciski nic nie robią. Kaskady nie da się sprawdzić bez przeglądarki, więc
    zostaje pilnowanie samej reguły — tanie i celne dla tej jednej pułapki.
    """
    css = (Path(__file__).resolve().parents[1] / "src/bone/studio/web/styles.css").read_text(
        encoding="utf-8"
    )
    assert "[hidden]" in css
    normalised = css[css.index("[hidden]") :].replace(" ", "")
    assert normalised.startswith("[hidden]{display:none!important;}")
