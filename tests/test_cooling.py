"""Dyssypacja: co chłodzenie MA robić i czego nie ma prawa ruszyć.

Kluczowy test to ``test_bulk_flow_survives_cooling``. Bez niego nie ma podstaw
twierdzić, że to chłodzenie, a nie zwykłe tarcie — a te dwie rzeczy wyglądają na
obrazku identycznie (obie zagęszczają układ) i różnią się dopiero tym, że tarcie
kasuje moment pędu i sprowadza wszystko do środka niezależnie od fizyki.
"""

from __future__ import annotations

import numpy as np
import pytest

from bone import relativity as sr
from bone.config import Config
from bone.cooling import Cooling
from bone.engine import Engine
from bone.state import State


def _uniform_cloud(n: int, seed: int, sigma: float, bulk: np.ndarray) -> State:
    """Chmura o zadanym ruchu masowym i zadanej dyspersji wokół niego."""
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-4.0, 4.0, size=(n, 3))
    masses = np.full(n, 1.0)
    velocities = bulk + rng.normal(0.0, sigma, size=(n, 3))
    return State(
        positions=positions,
        momenta=sr.momentum(masses, velocities, 30.0),
        masses=masses,
    )


def _physics(**overrides) -> object:
    """Domyślnie BEZ podłogi Jeansa, żeby testy jednostkowe badały po jednym
    mechanizmie. Podłoga Jeansa zależy od gęstości i sama z siebie blokowałaby
    chłodzenie w gęstych chmurach testowych, maskując to, co jest sprawdzane."""
    return Config().replace_flat({"c": 30.0, "cooling_min_particles": 0.0, **overrides}).physics


def _dispersion(state: State, c: float) -> float:
    v = sr.velocity(state.masses, state.momenta, c)
    return float(np.sqrt(np.mean((v - v.mean(axis=0)) ** 2) * 3.0))


def test_cooling_reduces_velocity_dispersion():
    state = _uniform_cloud(4000, seed=1, sigma=1.0, bulk=np.zeros(3))
    phys = _physics(cooling_rate=4.0, cooling_density_power=0.0, cooling_floor=0.0)
    before = _dispersion(state, phys.c)

    removed = Cooling().apply(state, phys, dt=0.1)

    assert _dispersion(state, phys.c) < 0.9 * before
    assert removed > 0.0


def test_bulk_flow_survives_cooling():
    """Ruch masowy musi przejść nietknięty — inaczej to tarcie, nie chłodzenie.

    Tarcie ``v ← v(1−λΔt)`` przeszłoby poprzedni test równie dobrze, a tutaj
    ścięłoby prędkość masową o dokładnie ten sam czynnik, którym tłumi dyspersję.
    """
    bulk = np.array([2.5, -1.0, 0.5])
    state = _uniform_cloud(4000, seed=2, sigma=1.0, bulk=bulk)
    phys = _physics(cooling_rate=8.0, cooling_density_power=0.0, cooling_floor=0.0)

    Cooling().apply(state, phys, dt=0.25)

    velocities = sr.velocity(state.masses, state.momenta, phys.c)
    assert np.allclose(velocities.mean(axis=0), bulk, atol=0.05)
    # a jednocześnie dyspersja faktycznie spadła — obie rzeczy naraz
    assert _dispersion(state, phys.c) < 0.5


def test_rotation_is_not_damped():
    """Wirujący dysk ma po chłodzeniu zachować moment pędu.

    To ten sam niezmiennik co powyżej, ale w geometrii, w której naprawdę o niego
    chodzi: chłodzony dysk ma stawać się cieńszy, a nie zwalniać.
    """
    cfg = Config().replace_flat({
        "n_particles": 3000, "geometry": "disk", "rotation": 1.0, "temperature": 0.02,
        "backend": "exact", "device": "cpu",
        "cooling_rate": 5.0, "cooling_density_power": 0.0, "cooling_floor": 0.0,
    })
    engine = Engine(cfg)
    rel = engine.state.positions - engine.state.center_of_mass()
    before = float(np.linalg.norm(np.cross(rel, engine.state.momenta).sum(axis=0)))
    thickness_before = float(np.std(engine.state.positions[:, 2]))

    for _ in range(40):
        engine.step()

    rel = engine.state.positions - engine.state.center_of_mass()
    after = float(np.linalg.norm(np.cross(rel, engine.state.momenta).sum(axis=0)))
    assert after == pytest.approx(before, rel=0.05), "chłodzenie skasowało moment pędu"
    # bez tych dwóch warunków test przechodziłby także wtedy, gdyby chłodzenie
    # w ogóle nie zadziałało — a wtedy nie dowodziłby niczego
    assert engine.energy_removed > 0.0
    assert float(np.std(engine.state.positions[:, 2])) < thickness_before, "dysk nie zcieńczał"
    engine.close()


def test_total_momentum_is_conserved_exactly():
    """Interpolacja CIC nie jest tożsamościowa, więc pęd trzeba przywracać jawnie.

    Bez tego resztka narastałaby przez tysiące kroków i układ odpłynąłby z kadru.
    """
    state = _uniform_cloud(3000, seed=3, sigma=1.2, bulk=np.array([0.7, 0.0, -0.3]))
    phys = _physics(cooling_rate=6.0, cooling_floor=0.0)
    reference = state.momenta.sum(axis=0).copy()
    scale = float(np.linalg.norm(state.momenta, axis=1).sum())

    cooling = Cooling()
    for _ in range(30):
        cooling.apply(state, phys, dt=0.05)

    assert np.linalg.norm(state.momenta.sum(axis=0) - reference) / scale < 1e-13


def test_energy_budget_closes_so_drift_still_measures_integration():
    """E_tot + E_wypromieniowana ma być zachowane — inaczej E_drift traci sens.

    To jest cena wejścia dyssypacji do tego kodu: główny wskaźnik jakości nie
    może zamienić się w licznik energii, którą celowo wyrzuciliśmy.
    """
    cfg = Config().replace_flat({
        # rotacja zero i niezerowa temperatura, żeby energia kinetyczna była
        # głównie ruchem nieuporządkowanym — czyli tym, co chłodzenie zabiera
        "n_particles": 1200, "backend": "exact", "device": "cpu",
        "rotation": 0.0, "temperature": 0.05,
        "softening": 0.3, "dt_max": 0.002, "accuracy": 0.01,
        # bez podłóg: test dotyczy księgowania energii, a nie progów. Przy 1200
        # cząstkach domyślny próg 1000 na masę Jeansa obejmuje niemal cały układ
        # i chłodzenie stanęłoby, zanim cokolwiek zdążyłoby się rozliczyć.
        "cooling_rate": 3.0, "cooling_floor": 0.0, "cooling_min_particles": 0.0,
    })
    engine = Engine(cfg)
    first = engine.collect_diagnostics()
    for _ in range(60):
        engine.step()
    last = engine.collect_diagnostics()

    assert last["E_cooled"] > 0.2 * first["E_kin"], "chłodzenie zabrało zbyt mało"
    assert abs(last["E_drift"]) < 5e-3, f"bilans się nie domyka: {last['E_drift']:.2e}"
    engine.close()


def test_mass_floor_stops_cooling_when_fragments_would_be_unresolved():
    """Chłodzenie ma się zatrzymać, gdy masa Jeansa zjeżdża do zadanej liczby cząstek.

    To jest właściwy niezmiennik chłodzenia w kodzie cząstkowym i on odróżnia je
    od stałego progu na dyspersję: masa Jeansa zależy również od gęstości, więc
    stały próg σ przestaje chronić rozdzielczość, gdy ρ rośnie. Test chłodzi
    bardzo mocno i sprawdza, gdzie proces stanął.
    """
    min_particles = 500.0
    rng = np.random.default_rng(11)
    n, particle_mass, G = 8000, 0.5, 0.05
    positions = rng.normal(0.0, 1.5, size=(n, 3))
    masses = np.full(n, particle_mass)
    state = State(
        positions=positions,
        momenta=sr.momentum(masses, rng.normal(0.0, 3.0, size=(n, 3)), 30.0),
        masses=masses,
    )
    phys = (
        Config()
        .replace_flat({
            "c": 30.0, "G": G, "cooling_rate": 40.0, "cooling_density_power": 0.0,
            "cooling_floor": 0.0, "cooling_min_particles": min_particles,
        })
        .physics
    )

    cooling = Cooling()
    for _ in range(600):
        cooling.apply(state, phys, dt=0.01)

    # zmierz masę Jeansa tam, gdzie masa faktycznie jest: w rdzeniu chmury
    radius = 1.5
    core = np.linalg.norm(positions, axis=1) < radius
    velocities = sr.velocity(masses, state.momenta, phys.c)
    sigma_1d = float(np.sqrt(np.mean(np.var(velocities[core], axis=0))))
    density = float(masses[core].sum() / (4.0 / 3.0 * np.pi * radius**3))
    jeans_length = sigma_1d * np.sqrt(np.pi / (G * density))
    particles_in_jeans_mass = density * jeans_length**3 / particle_mass

    assert particles_in_jeans_mass > 0.2 * min_particles, (
        f"zjechało pod podłogę: {particles_in_jeans_mass:.0f} cząstek na masę Jeansa"
    )
    assert particles_in_jeans_mass < 20.0 * min_particles, (
        f"podłoga zablokowała chłodzenie za wcześnie: {particles_in_jeans_mass:.0f}"
    )


def test_undersampled_grid_is_reported_not_silently_ineffective():
    """Za gęsta siatka musi być słychać.

    Bez ostrzeżenia użytkownik dostaje bieg, w którym chłodzenie nic nie robi,
    kończący się bez błędu — i fałszywy wniosek, że dyssypacja nic nie zmienia.
    """
    state = _uniform_cloud(200, seed=6, sigma=1.0, bulk=np.zeros(3))
    with pytest.warns(RuntimeWarning, match="cząstek na zajętą komórkę"):
        Cooling(grid=64).apply(state, _physics(cooling_rate=5.0), dt=0.05)


def test_zero_rate_is_a_true_no_op():
    state = _uniform_cloud(500, seed=4, sigma=1.0, bulk=np.zeros(3))
    momenta = state.momenta.copy()
    removed = Cooling().apply(state, _physics(cooling_rate=0.0), dt=0.1)
    assert removed == 0.0
    assert np.array_equal(state.momenta, momenta)


def test_floor_stops_cooling_at_the_requested_dispersion():
    """Poniżej podłogi chłodzenie musi wygasnąć, nie tylko zwolnić."""
    sigma_floor = 0.05
    state = _uniform_cloud(4000, seed=5, sigma=0.6, bulk=np.zeros(3))
    phys = _physics(
        cooling_rate=30.0, cooling_density_power=0.0, cooling_floor=sigma_floor / 30.0
    )
    cooling = Cooling()
    for _ in range(400):
        cooling.apply(state, phys, dt=0.02)

    remaining = _dispersion(state, phys.c)
    assert remaining > 0.4 * sigma_floor, "zjechało poniżej podłogi"
    assert remaining < 4.0 * sigma_floor, "podłoga zablokowała chłodzenie za wcześnie"


def test_density_power_makes_cooling_selective():
    """Przy wykładniku > 0 gęste obszary mają chłodzić się szybciej niż rzadkie.

    To jest ta własność, która pozwala fragmentować zamiast opadać równomiernie.
    """
    rng = np.random.default_rng(7)
    n = 6000
    # dwie chmury o tej samej dyspersji i bardzo różnej gęstości
    dense = rng.normal([-6.0, 0.0, 0.0], 0.5, size=(n // 2, 3))
    sparse = rng.normal([6.0, 0.0, 0.0], 2.5, size=(n // 2, 3))
    positions = np.vstack([dense, sparse])
    masses = np.full(n, 1.0)
    velocities = rng.normal(0.0, 1.0, size=(n, 3))
    state = State(
        positions=positions, momenta=sr.momentum(masses, velocities, 30.0), masses=masses
    )
    phys = _physics(cooling_rate=2.0, cooling_density_power=1.0, cooling_floor=0.0)

    Cooling().apply(state, phys, dt=0.2)

    v = sr.velocity(masses, state.momenta, phys.c)
    half = n // 2
    spread_dense = float(np.std(v[:half]))
    spread_sparse = float(np.std(v[half:]))
    assert spread_dense < 0.8 * spread_sparse
