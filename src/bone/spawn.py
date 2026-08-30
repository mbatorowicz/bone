"""Warunki początkowe.

Prędkość początkowa jest skalowana do lokalnej prędkości okrężnej wyliczonej
z masy zamkniętej wewnątrz promienia, a nie do ułamka ``c``. Dzięki temu
``rotation = 1`` naprawdę oznacza orbitę kołową dla dowolnego G, N i promienia,
zamiast liczby, którą trzeba dobierać metodą prób.
"""

from __future__ import annotations

import warnings

import numpy as np

from bone import relativity as sr
from bone.config import Config
from bone.state import State


def _ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * (rng.random(n) ** (1 / 3) * radius)[:, None]


def _sphere_shell(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * radius


def _disk(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    r = radius * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0.0, 0.04 * radius, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a), z])


def _torus(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    big, small = radius, 0.3 * radius
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    rr = small * np.sqrt(rng.random(n))
    return np.column_stack(
        [
            (big + rr * np.cos(phi)) * np.cos(theta),
            (big + rr * np.cos(phi)) * np.sin(theta),
            rr * np.sin(phi),
        ]
    )


def _gaussian(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    return rng.normal(0.0, radius / 2.5, size=(n, 3))


def _filament(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    t = rng.uniform(-radius, radius, n)
    return np.column_stack(
        [t, rng.normal(0, 0.08 * radius, n), rng.normal(0, 0.08 * radius, n)]
    )


def _two_clumps(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    half = n // 2
    sep, sigma = 0.55 * radius, 0.16 * radius
    a = rng.normal([-sep, 0.0, 0.0], sigma, size=(half, 3))
    b = rng.normal([sep, 0.0, 0.0], sigma, size=(n - half, 3))
    return np.vstack([a, b])


def _plummer(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Sfera Plummera — jedyny z tych rozkładów, który jest samouzgodnionym
    rozwiązaniem równania Poissona, więc nadaje się na test równowagi."""
    a = radius / 3.0
    x = rng.random(n)
    r = a / np.sqrt(np.maximum(x ** (-2 / 3) - 1.0, 1e-12))
    # ogon ucięty na 6a (96% masy). Solver siatkowy dopasowuje pudło do PEŁNEJ
    # rozciągłości chmury, więc kilka cząstek na r = 30a obniżyłoby rozdzielczość
    # wszystkim pozostałym.
    r = np.minimum(r, 6.0 * a)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * r[:, None]


_SHAPES = {
    "ball": _ball,
    "sphere_shell": _sphere_shell,
    "disk": _disk,
    "torus": _torus,
    "gaussian": _gaussian,
    "filament": _filament,
    "two_clumps": _two_clumps,
    "plummer": _plummer,
}


def sample_positions(name: str, rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    try:
        shape = _SHAPES[name]
    except KeyError:
        raise KeyError(f"nieznany kształt: {name!r}; dostępne: {sorted(_SHAPES)}") from None
    return np.ascontiguousarray(shape(rng, n, radius), dtype=np.float64)


def circular_speed(positions: np.ndarray, masses: np.ndarray, G: float, softening: float) -> np.ndarray:
    """v_okrężna(r) = √(G·M(<r)/√(r²+ε²)) wokół środka masy.

    M(<r) liczone z posortowanych promieni, więc koszt to O(N log N) i nie
    zależy od backendu — to tylko warunek początkowy.
    """
    com = np.average(positions, axis=0, weights=masses)
    rel = positions - com
    r = np.linalg.norm(rel, axis=1)
    order = np.argsort(r)
    enclosed = np.empty_like(r)
    enclosed[order] = np.cumsum(masses[order])
    return np.sqrt(G * enclosed / np.sqrt(r * r + softening * softening))


def make_state(cfg: Config) -> State:
    sp, ph = cfg.spawn, cfg.physics
    rng = np.random.default_rng(sp.seed)
    n = int(sp.n_particles)

    positions = sample_positions(sp.geometry, rng, n, sp.radius)
    # rozrzut mas jest tylko kosmetyczny; suma jest przeskalowana dokładnie do
    # zamawianej masy układu, żeby liczba cząstek nie wpływała na dynamikę
    masses = np.maximum(rng.normal(1.0, sp.mass_spread, size=n), 0.05)
    masses *= sp.total_mass / masses.sum()

    velocities = np.zeros((n, 3), dtype=np.float64)

    if abs(sp.rotation) > 1e-9:
        com = np.average(positions, axis=0, weights=masses)
        rel = positions - com
        axis = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(axis, rel)
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        # cząstki na osi obrotu nie mają zdefiniowanej stycznej — zostają w spoczynku
        unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-12)
        v_circ = circular_speed(positions, masses, ph.G, ph.softening)
        velocities += (sp.rotation * v_circ)[:, None] * unit

    if abs(sp.temperature) > 1e-9:
        velocities += rng.normal(0.0, sp.temperature * ph.c / np.sqrt(3.0), size=(n, 3))

    # warunek początkowy musi być fizyczny: |v| < c z zapasem
    speed = np.linalg.norm(velocities, axis=1)
    limit = 0.95 * ph.c
    hot = speed > limit
    if np.any(hot):
        velocities[hot] *= (limit / speed[hot])[:, None]
        # Przycięcie ratuje całkowanie, ale niszczy zamawiany warunek początkowy:
        # obcięta orbita kołowa nie jest już kołowa, a układ może wystartować z
        # energią dodatnią i po prostu się rozlecieć. Lepiej powiedzieć to wprost
        # niż pozwolić użytkownikowi zobaczyć „zepsutą fizykę".
        share = float(np.count_nonzero(hot)) / n
        warnings.warn(
            f"{share:.1%} cząstek chciało lecieć szybciej niż 0,95 c i zostało "
            f"przyciętych — zamawiany warunek początkowy jest nieosiągalny przy "
            f"G={ph.G:g} i c={ph.c:g}. Zmniejsz G lub rotation, albo podnieś c "
            f"(patrz bone.config.gravity_for_beta).",
            RuntimeWarning,
            stacklevel=2,
        )

    # usuń pęd środka masy, żeby układ nie odpływał z kadru
    momenta = sr.momentum(masses, velocities, ph.c)
    momenta -= momenta.sum(axis=0) / n

    return State(positions=positions, momenta=momenta, masses=masses)
