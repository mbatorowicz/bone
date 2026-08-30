"""Konfiguracja, granica runtime/startup oraz zapis i odczyt danych."""

from __future__ import annotations

import numpy as np
import pytest

from bone.config import (
    GEOMETRIES,
    PRESETS,
    Config,
    apply_runtime,
    preset,
    startup_config,
    ui_schema,
)
from bone.engine import Engine
from bone.io import checkpoint, live, trajectory
from bone.spawn import make_state


def test_flat_roundtrip_is_lossless():
    cfg = preset("galaxy")
    assert Config.from_flat(cfg.to_flat()) == cfg


def test_every_ui_control_maps_to_a_real_field():
    """Suwak bez pola w configu to martwa kontrolka — dokładnie ten błąd,
    przez który 'Zasięg grawitacji' w poprzedniej wersji nic nie robił."""
    flat = Config().to_flat()
    schema = ui_schema()
    for group in schema["groups"]:
        for control in group["controls"]:
            assert control["key"] in flat, f"suwak {control['key']} nie ma pola w configu"


def test_runtime_changes_apply_and_startup_changes_do_not():
    base = Config()
    changed = apply_runtime(base, {"G": 3.0, "n_particles": 999_999})
    assert changed.physics.G == 3.0
    assert changed.spawn.n_particles == base.spawn.n_particles


def test_startup_config_accepts_text_choices_and_rejects_nonsense():
    cfg = startup_config(Config(), {"geometry": "torus", "backend": "mesh", "device": "cpu"})
    assert cfg.spawn.geometry == "torus"
    assert cfg.solver.backend == "mesh"
    unchanged = startup_config(Config(), {"geometry": "nie_istnieje"})
    assert unchanged.spawn.geometry == Config().spawn.geometry


def test_runtime_apply_ignores_text_fields_without_crashing():
    """Klient może przysłać cały stan panelu; serwer nie ma prawa się wywrócić
    na polu tekstowym, jak robił to stary `merge_runtime` na `out_dir`."""
    cfg = apply_runtime(Config(), {"out_dir": "runs/gdzies", "geometry": "ball", "G": 2.0})
    assert cfg.physics.G == 2.0
    assert cfg.run.out_dir == Config().run.out_dir


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_builds_and_runs(name):
    cfg = preset(name).replace_flat({"n_particles": 300, "backend": "exact", "device": "cpu"})
    engine = Engine(cfg)
    engine.advance(3)
    assert engine.state.is_finite()
    engine.close()


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_every_geometry_produces_finite_state(geometry):
    cfg = Config().replace_flat({"geometry": geometry, "n_particles": 200, "seed": 2})
    state = make_state(cfg)
    assert state.n == 200
    assert np.isfinite(state.positions).all()
    assert np.isfinite(state.momenta).all()
    assert np.all(state.masses > 0)
    assert np.all(state.speed_over_c(cfg.physics.c) < 1.0)


def test_particle_count_changes_resolution_not_physics():
    """Suwak liczby cząstek ma zmieniać rozdzielczość, nie badany układ.

    Dlatego konfiguracja mówi o masie CAŁEGO układu. Gdyby parametrem była masa
    pojedynczej cząstki, przejście z 4 na 100 tysięcy cząstek zwiększyłoby masę
    układu 25-krotnie: prędkość okrężna wzrosłaby 5-krotnie, przekroczyłaby c
    i warunek startowy przestałby być spełnialny.
    """
    import warnings

    reference = None
    for n in (1_000, 5_000, 25_000):
        cfg = Config().replace_flat({"n_particles": n, "backend": "exact", "device": "cpu"})
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # przycięcie prędkości = błąd doboru
            state = make_state(cfg)

        assert state.masses.sum() == pytest.approx(cfg.spawn.total_mass, rel=1e-12)

        beta_max = float(state.speed_over_c(cfg.physics.c).max())
        if reference is None:
            reference = beta_max
        else:
            # ta sama fizyka, inna liczba próbek: różnice tylko z losowania
            assert beta_max == pytest.approx(reference, rel=0.05)


def test_unreachable_initial_condition_is_reported_not_silently_clipped():
    """Za duże G przy danym c czyni orbitę kołową niespełnialną — musi być słychać.

    Bez tego ostrzeżenia użytkownik dostaje układ o dodatniej energii, który się
    rozlatuje, i wygląda to na błąd fizyki zamiast na błąd doboru parametrów.
    """
    cfg = Config().replace_flat({"n_particles": 400, "G": 50.0, "c": 5.0, "rotation": 1.0})
    with pytest.warns(RuntimeWarning, match="0,95 c"):
        state = make_state(cfg)
    # nawet po przycięciu stan zostaje fizyczny
    assert np.all(state.speed_over_c(cfg.physics.c) < 1.0)


def test_gravity_for_beta_hits_the_requested_edge_speed():
    from bone.config import gravity_for_beta

    mass, radius, c, beta = 4_000.0, 10.0, 30.0, 0.25
    G = gravity_for_beta(mass, radius, c, beta)
    assert np.sqrt(G * mass / radius) == pytest.approx(beta * c, rel=1e-12)


def test_initial_momentum_sums_to_zero():
    state = make_state(Config().replace_flat({"n_particles": 500, "rotation": 0.8, "temperature": 0.05}))
    scale = np.linalg.norm(state.momenta, axis=1).sum()
    assert np.linalg.norm(state.momenta.sum(axis=0)) / scale < 1e-14


def test_checkpoint_roundtrip(tmp_path):
    cfg = Config().replace_flat({"n_particles": 250, "backend": "exact", "device": "cpu"})
    engine = Engine(cfg)
    engine.advance(5)
    checkpoint.save(engine.state, engine.cfg, tmp_path)

    restored, restored_cfg = checkpoint.load(tmp_path)
    assert np.array_equal(restored.positions, engine.state.positions)
    assert np.array_equal(restored.momenta, engine.state.momenta)
    assert restored.step == engine.state.step
    assert restored_cfg == engine.cfg


def test_trajectory_index_gives_direct_frame_access(tmp_path):
    cfg = Config().replace_flat({"n_particles": 120, "backend": "exact", "device": "cpu"})
    engine = Engine(cfg)
    writer = trajectory.TrajectoryWriter(tmp_path, chunk_size=4)
    times = []
    for _ in range(11):
        engine.advance(1)
        writer.add(engine.state, cfg)
        times.append(engine.state.time)
    writer.close()

    assert trajectory.read_meta(tmp_path)["n_frames"] == 11
    for wanted in (0, 5, 10):
        frame = trajectory.load_frame(tmp_path, wanted)
        assert frame is not None
        positions, shades, when = frame
        assert positions.shape == (120, 3)
        assert shades.shape == (120,)
        assert when == pytest.approx(times[wanted])
    assert trajectory.load_frame(tmp_path, 99) is None


def test_live_buffer_layout():
    cfg = Config().replace_flat({"n_particles": 64, "backend": "exact", "device": "cpu"})
    engine = Engine(cfg)
    payload = live.pack_view(engine.state, cfg)
    magic, n, half, when = live.HEADER.unpack_from(payload, 0)
    assert magic == live.MAGIC
    assert n == 64
    assert half > 0
    expected = live.HEADER.size + n * 3 * 4 + n * 4
    assert len(payload) == expected
    positions = np.frombuffer(payload, dtype=np.float32, count=n * 3, offset=live.HEADER.size)
    assert np.allclose(positions.reshape(n, 3), engine.state.positions, atol=1e-4)


def test_runtime_config_change_reaches_the_engine():
    """Regresja: parametry runtime muszą działać w trakcie biegu."""
    cfg = Config().replace_flat({"n_particles": 200, "backend": "exact", "device": "cpu"})
    engine = Engine(cfg)
    engine.advance(2)
    engine.apply_config(cfg.replace_flat({"G": 4.0}))
    assert engine.cfg.physics.G == 4.0
    force_after = np.linalg.norm(engine.state.forces, axis=1).mean()
    engine.apply_config(cfg.replace_flat({"G": 1.0}))
    force_weaker = np.linalg.norm(engine.state.forces, axis=1).mean()
    assert force_after > force_weaker
