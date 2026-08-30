"""Kształty startowe i ich dwa parametry proporcji.

Cel tych testów jest węższy niż „czy się nie wywala" — to sprawdza już
`test_every_geometry_produces_finite_state`. Tutaj chodzi o to, żeby pokrętła
FAKTYCZNIE zmieniały geometrię. Kontrolka, która nic nie robi, jest gorsza od
braku kontrolki, bo prowadzi do wniosków wyciągniętych z niezmienionego układu.
"""

from __future__ import annotations

import numpy as np
import pytest

from bone.config import GEOMETRIES, GEOMETRY_LABELS, Config
from bone.engine import Engine
from bone.spawn import _SHAPES_WITH_THICKNESS, make_state, sample_positions


def _positions(geometry: str, n: int = 4000, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(7)
    return sample_positions(geometry, rng, n, kwargs.pop("radius", 8.0), **kwargs)


def _extent(positions: np.ndarray) -> np.ndarray:
    return positions.max(axis=0) - positions.min(axis=0)


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_every_geometry_has_a_label_for_the_panel(geometry):
    """Nowy kształt bez etykiety pokazałby się w panelu jako goły identyfikator."""
    assert geometry in GEOMETRY_LABELS, f"kształt {geometry} nie ma etykiety w interfejsie"


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_flatten_squashes_z_and_leaves_the_plane_alone(geometry):
    """Spłaszczenie ma działać na każdy kształt i tylko na wysokość.

    To jest sens rozdzielenia rozmiaru od proporcji: zmiana kształtu nie może
    przy okazji zmieniać skali, bo wtedy porównania między biegami są nieuczciwe.
    """
    plain = _positions(geometry)
    flat = _positions(geometry, flatten=0.2)

    assert _extent(flat)[2] < 0.5 * _extent(plain)[2], "wysokość się nie zmieniła"
    assert np.allclose(flat[:, :2], plain[:, :2]), "spłaszczenie ruszyło płaszczyznę xy"


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_radius_sets_the_scale_for_every_geometry(geometry):
    """Podwojenie promienia ma podwoić rozciągłość — dla każdego kształtu.

    Bez tego niektóre kształty miałyby rozmiar wpisany na sztywno i suwak
    promienia działałby na część z nich, co jest najgorszym wariantem: pokrętło
    działa, ale nie zawsze.
    """
    small = _extent(_positions(geometry, radius=5.0))
    large = _extent(_positions(geometry, radius=10.0))
    assert np.allclose(large, 2.0 * small, rtol=0.15)


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
def test_thickness_acts_exactly_on_the_shapes_that_declare_it(geometry):
    """Grubość musi zmieniać przekrój tam, gdzie kształt ją zapowiada — i nigdzie
    indziej. Milcząco ignorowany parametr byłby kontrolką bez skutku."""
    thin = _positions(geometry, thickness=0.02)
    thick = _positions(geometry, thickness=0.4)

    if geometry in _SHAPES_WITH_THICKNESS:
        assert not np.allclose(thin, thick), f"{geometry} ignoruje grubość"
    else:
        assert np.array_equal(thin, thick), f"{geometry} niespodziewanie czyta grubość"


def test_cube_fills_corners_that_a_ball_cannot_reach():
    """Kostka i kula o tym samym promieniu różnią się właśnie narożnikami.

    Test rozstrzyga to bez odwoływania się do implementacji: narożnik kostki leży
    na odległości r√3 od środka, a kula nigdy nie wychodzi poza r.
    """
    radius = 8.0
    cube = _positions("cube", radius=radius)
    ball = _positions("ball", radius=radius)

    assert np.linalg.norm(cube, axis=1).max() > 1.3 * radius
    assert np.linalg.norm(ball, axis=1).max() <= radius + 1e-9
    # kostka jest izotropowa w sensie rozciągłości, choć nie w sensie promienia
    assert np.allclose(_extent(cube), 2.0 * radius, rtol=0.05)


def test_cylinder_keeps_its_width_up_to_the_ends_unlike_a_ball():
    """Walec ma stały przekrój wzdłuż osi — kula zwęża się ku biegunom."""
    radius = 8.0
    near_end = 0.8 * radius

    def width_near_end(geometry: str) -> float:
        positions = _positions(geometry, radius=radius, n=8000)
        ends = np.abs(positions[:, 2]) > near_end
        assert ends.any(), f"{geometry} nie sięga tak wysoko"
        return float(np.linalg.norm(positions[ends][:, :2], axis=1).max())

    assert width_near_end("cylinder") > 0.8 * radius
    assert width_near_end("ball") < 0.7 * radius


def test_torus_has_a_hole_and_a_ball_does_not():
    """Cecha definiująca torusa: brak masy przy osi."""
    radius = 8.0
    torus = _positions("torus", radius=radius, thickness=0.15)
    plane_radius = np.linalg.norm(torus[:, :2], axis=1)
    assert plane_radius.min() > 0.5 * radius, "torus nie ma dziury"


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("target", [0.5, 1.0])
def test_virial_target_is_hit_for_every_shape(geometry, target):
    """Zadany wirial musi wyjść dla KAŻDEGO kształtu — to cała jego wartość.

    Gdyby trafiał tylko dla kuli, presety porównawcze dalej mieszałyby wpływ
    geometrii z odległością od równowagi i nie dałoby się rozstrzygnąć, co
    spowodowało wynik.
    """
    cfg = Config().replace_flat({
        "geometry": geometry, "n_particles": 2500, "rotation": 0.0, "virial": target,
        "softening": 0.15, "backend": "exact", "device": "cpu",
    })
    engine = Engine(cfg)
    measured = engine.collect_diagnostics()["virial"]
    engine.close()
    assert measured == pytest.approx(target, rel=0.05)


def test_virial_overrides_temperature_and_zero_gives_it_back():
    """Dwa pokrętła na tę samą wielkość muszą mieć jednoznaczne pierwszeństwo."""
    base = {
        "geometry": "cube", "n_particles": 2000, "rotation": 0.0,
        "temperature": 0.4, "backend": "exact", "device": "cpu",
    }
    with_virial = Engine(Config().replace_flat({**base, "virial": 0.6}))
    assert with_virial.collect_diagnostics()["virial"] == pytest.approx(0.6, rel=0.05)
    with_virial.close()

    by_hand = Engine(Config().replace_flat({**base, "virial": 0.0}))
    # sama temperatura 0,4·c daje wirial daleki od 0,6 — czyli naprawdę wróciła
    assert by_hand.collect_diagnostics()["virial"] > 1.5
    by_hand.close()


def test_virial_target_of_zero_kinetic_energy_is_reachable():
    """Wirial 0 to zimny start — nie może wymagać ujemnej dyspersji."""
    cfg = Config().replace_flat({
        "geometry": "ball", "n_particles": 800, "rotation": 0.0, "virial": 0.0,
        "temperature": 0.0, "backend": "exact", "device": "cpu",
    })
    state = make_state(cfg)
    assert np.allclose(state.momenta, 0.0)


def test_subsampled_potential_matches_the_full_sum():
    """|U| zależy od rozkładu masy, nie od liczby próbek.

    Na tym opiera się cała wydajność wirializacji: podpróbka niosąca całą masę
    układu daje tę samą energię, więc koszt O(m²) nie rośnie ze rozdzielczością.
    """
    from bone.spawn import _VIRIAL_SAMPLE, potential_energy

    rng = np.random.default_rng(3)
    small = sample_positions("ball", rng, _VIRIAL_SAMPLE, 8.0)
    large = sample_positions("ball", np.random.default_rng(4), 8 * _VIRIAL_SAMPLE, 8.0)

    total_mass, G, eps = 4000.0, 0.1, 0.2
    direct = potential_energy(small, np.full(len(small), total_mass / len(small)), G, eps)
    sampled = potential_energy(large, np.full(len(large), total_mass / len(large)), G, eps)
    assert sampled == pytest.approx(direct, rel=0.05)


def test_flatten_reaches_the_engine_through_the_config():
    """Pole musi przejść całą drogę config → spawn, nie tylko istnieć w schemacie."""
    base = Config().replace_flat({"n_particles": 800, "geometry": "ball", "rotation": 0.0})
    round_shape = make_state(base)
    squashed = make_state(base.replace_flat({"flatten": 0.1}))

    height = lambda s: float(s.positions[:, 2].max() - s.positions[:, 2].min())  # noqa: E731
    assert height(squashed) < 0.3 * height(round_shape)
