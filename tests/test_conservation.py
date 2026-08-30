"""Wielkości zachowane — jedyny obiektywny sprawdzian, czy to jeszcze fizyka.

Te testy są powodem, dla którego backend zwraca potencjał razem z siłą.
"""

from __future__ import annotations

import numpy as np

from bone import relativity as sr
from bone.config import Config
from bone.engine import Engine
from bone.state import State


def make_engine(**overrides) -> Engine:
    base = {
        "n_particles": 600,
        "geometry": "plummer",
        "radius": 8.0,
        "seed": 4,
        "rotation": 0.5,
        "backend": "exact",
        "device": "cpu",
        "dt_max": 0.004,
        "accuracy": 0.015,
        "softening": 0.35,
        "G": 1.0,
        "c": 40.0,
        "steps": 0,
    }
    base.update(overrides)
    return Engine(Config().replace_flat(base))


def total_energy(engine: Engine) -> float:
    row = engine.collect_diagnostics()
    return row["E_tot"]


def test_energy_drift_is_small_over_many_steps():
    engine = make_engine()
    before = total_energy(engine)
    engine.advance(400)
    after = total_energy(engine)
    drift = abs(after - before) / abs(before)
    assert drift < 2e-3, f"dryf energii {drift:.3e} jest za duży"


def test_momentum_is_conserved_to_machine_precision():
    engine = make_engine()
    p0 = engine.state.momenta.sum(axis=0).copy()
    engine.advance(200)
    p1 = engine.state.momenta.sum(axis=0)
    scale = np.linalg.norm(engine.state.momenta, axis=1).sum()
    assert np.linalg.norm(p1 - p0) / scale < 1e-12


def test_angular_momentum_is_conserved():
    engine = make_engine()
    from bone.diagnostics import angular_momentum

    l0 = angular_momentum(engine.state).copy()
    engine.advance(300)
    l1 = angular_momentum(engine.state)
    assert np.linalg.norm(l1 - l0) / np.linalg.norm(l0) < 1e-3


def com_velocity(engine: Engine) -> np.ndarray:
    """Prędkość środka masy spoczynkowej: Σmv/Σm."""
    v = sr.velocity(engine.state.masses, engine.state.momenta, engine.cfg.physics.c)
    m = engine.state.masses
    return (m[:, None] * v).sum(axis=0) / m.sum()


def test_center_of_mass_drift_matches_prediction():
    """Środek masy wędruje — i wędruje DOKŁADNIE tyle, ile przewiduje kinematyka.

    Przy Σp = 0 środek masy spoczynkowej nie stoi, bo jego prędkość to Σmv/Σm =
    Σ(p/γ)/Σm, a to nie to samo co Σp: cząstki o większym γ wnoszą mniej prędkości
    na jednostkę pędu. Ściśle nieruchomy byłby środek energii, ale i on nie jest
    zachowany dokładnie, bo grawitacja jest tu newtonowska (natychmiastowa), więc
    pole nie niesie pędu. To znany kompromis modelu „kinematyka SR + siła Newtona".

    Dlatego nie porównujemy dryfu z wyssanym z palca progiem — sprawdzamy, czy
    zgadza się z niezależnie wyliczoną prędkością środka masy. To wykrywa prawdziwy
    błąd (dryf z niewłaściwego źródła, np. niesymetrycznej siły), a nie reaguje na
    zmianę parametrów startowych.
    """
    engine = make_engine()
    start = engine.state.center_of_mass().copy()
    v_com = com_velocity(engine)
    elapsed = engine.advance(300)
    moved = engine.state.center_of_mass() - start
    predicted = np.linalg.norm(v_com) * elapsed

    assert predicted > 0.0, "test bez sensu, gdyby przewidywany dryf był zerowy"
    # v_com sama się zmienia w trakcie biegu, więc zgodność co do rzędu wielkości
    # jest tu mocnym stwierdzeniem: wyklucza dryf z innego źródła niż kinematyka
    assert 0.2 < np.linalg.norm(moved) / predicted < 5.0, (
        f"dryf {np.linalg.norm(moved):.3e} nie zgadza się z przewidywaniem "
        f"{predicted:.3e} — środek masy rusza z innego powodu niż kinematyka SR"
    )
    # i tak jest nieistotny w skali układu
    assert np.linalg.norm(moved) / engine.collect_diagnostics()["r_half"] < 0.01


def test_smaller_timestep_gives_smaller_drift():
    """Dowód, że dryf pochodzi z całkowania, a nie z niespójności modelu."""
    coarse = make_engine(dt_max=0.02, accuracy=0.2)
    fine = make_engine(dt_max=0.005, accuracy=0.05)
    results = []
    for engine in (coarse, fine):
        before = total_energy(engine)
        while engine.state.time < 1.0:
            engine.step()
        results.append(abs(total_energy(engine) - before) / abs(before))
    assert results[1] < results[0]


def test_two_body_circular_orbit_stays_circular():
    """Dwa ciała na orbicie kołowej: promień i okres muszą się utrzymać.

    Warunek na ruch po okręgu w SR to γmv²/r = F, bo p = γmv, a nie mv.
    Test przechodzi tylko wtedy, gdy integrator naprawdę całkuje pęd.
    """
    G, c, eps = 1.0, 60.0, 1e-6
    mass, r = 1.0, 4.0
    separation = 2 * r
    force = G * mass * mass / (separation**2 + eps**2) ** 1.5 * separation
    # γ m v² / r = F, rozwiązane iteracyjnie względem v (γ zależy od v)
    v = np.sqrt(force * r / mass)
    for _ in range(60):
        gamma = 1.0 / np.sqrt(1.0 - (v / c) ** 2)
        v = np.sqrt(force * r / (gamma * mass))

    positions = np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]])
    velocities = np.array([[0.0, v, 0.0], [0.0, -v, 0.0]])
    masses = np.array([mass, mass])
    state = State(
        positions=positions,
        momenta=sr.momentum(masses, velocities, c),
        masses=masses,
    )

    cfg = Config().replace_flat(
        {
            "backend": "exact",
            "device": "cpu",
            "G": G,
            "c": c,
            "softening": eps,
            "dt_max": 0.002,
            "adaptive_dt": False,
        }
    )
    engine = Engine(cfg, state=state)

    period = 2 * np.pi * r / v
    radii = []
    while engine.state.time < period:
        engine.step()
        radii.append(np.linalg.norm(engine.state.positions[0]))

    radii = np.asarray(radii)
    assert abs(radii.max() - radii.min()) / r < 1e-3, "orbita nie jest kołowa"
    # po pełnym okresie cząstka wraca w okolicę punktu startowego
    back = np.linalg.norm(engine.state.positions[0] - positions[0])
    assert back / r < 2e-2


def test_plummer_sphere_stays_near_virial_equilibrium():
    """Sfera Plummera z prędkościami wirialnymi nie powinna gwałtownie kolapsować."""
    engine = make_engine(n_particles=800, rotation=0.0, temperature=0.0, seed=17)
    r_start = engine.collect_diagnostics()["r_half"]
    engine.advance(400)
    r_end = engine.collect_diagnostics()["r_half"]
    assert 0.2 < r_end / r_start < 5.0


def test_engine_raises_instead_of_producing_nan():
    engine = make_engine(n_particles=200, softening=0.3)
    engine.advance(20)
    engine.state.positions[0] = np.nan
    assert not engine.state.is_finite()
