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


def _cube(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Kostka jednorodna o połowie boku ``radius``, wypełniona losowo.

    Losowo, a nie na siatce, i to jest istotne: rozkład Poissona ma wbudowane
    fluktuacje gęstości ~1/√N na każdej skali, czyli gotowe zarodki niestabilności
    grawitacyjnej. Siatka regularna ich nie ma i musiałaby dostać zaburzenie
    z zewnątrz, żeby cokolwiek się na niej zaczęło.
    """
    return rng.uniform(-radius, radius, size=(n, 3))


def _cylinder(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Walec pełny o osi z, promieniu ``radius`` i wysokości równej średnicy.

    Proporcje ustawia potem spłaszczenie (patrz `sample_positions`), więc walec
    nie potrzebuje własnego parametru długości: 0,2 daje krążek, 3 daje pręt.
    Pierwiastek z liczby losowej jest konieczny — bez niego cząstki skupiłyby się
    przy osi, bo pole pierścienia rośnie liniowo z promieniem.
    """
    r = radius * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a), rng.uniform(-radius, radius, n)])


def _disk(rng: np.random.Generator, n: int, radius: float, thickness: float) -> np.ndarray:
    r = radius * np.sqrt(rng.random(n))
    a = rng.uniform(0, 2 * np.pi, n)
    z = rng.normal(0.0, max(thickness, 1e-6) * radius, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a), z])


def _torus(rng: np.random.Generator, n: int, radius: float, thickness: float) -> np.ndarray:
    """Torus o promieniu wiodącym ``radius`` i przekroju ``thickness``·radius.

    Przy małej grubości jest to zamknięte włókno bez końców i właśnie dlatego
    jest to jedyny kształt w tym zestawie, który potrafi fragmentować bez
    dyssypacji. Włókno o swobodnych końcach zapada się od nich do środka, a to
    zapadanie skaluje się jak 1/grubość — czyli dokładnie tak samo jak wzrost
    zgęstek, więc pocienianie nie rozdziela tych dwóch modów i globalny zawsze
    wygrywa (zmierzone dwukrotnie, dla grubości 0,64 i 0,28). Torus nie ma końców,
    więc ten mod po prostu nie istnieje, a przy nadanej rotacji nie ma też
    globalnego zapadania promieniowego. Zostaje wyłącznie fragmentacja podłużna.
    """
    minor = max(thickness, 1e-6) * radius
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    rr = minor * np.sqrt(rng.random(n))
    return np.column_stack(
        [
            (radius + rr * np.cos(phi)) * np.cos(theta),
            (radius + rr * np.cos(phi)) * np.sin(theta),
            rr * np.sin(phi),
        ]
    )


def _gaussian(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    return rng.normal(0.0, radius / 2.5, size=(n, 3))


def _filament(rng: np.random.Generator, n: int, radius: float, thickness: float) -> np.ndarray:
    """Walec o długości 2·radius i gaussowskim przekroju o σ = thickness·radius.

    Grubość jest osobnym parametrem, bo to ONA, a nie długość, wyznacza długość
    fali fragmentacji (λ ≈ 3,6·σ). Przy sztywno wpisanym σ = 0,08·radius kształt
    był samopodobny: na całą długość wypadało zawsze ~7 długości Jeansa, czyli
    za mało, żeby fragmentacja wygrała z globalnym zapadaniem się włókna do
    środka — i żadna zmiana promienia tego nie ruszała.
    """
    sigma = max(thickness, 1e-6) * radius
    t = rng.uniform(-radius, radius, n)
    return np.column_stack([t, rng.normal(0, sigma, n), rng.normal(0, sigma, n)])


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
    "cube": _cube,
    "cylinder": _cylinder,
    "disk": _disk,
    "torus": _torus,
    "sphere_shell": _sphere_shell,
    "gaussian": _gaussian,
    "filament": _filament,
    "two_clumps": _two_clumps,
    "plummer": _plummer,
}


#: kształty, które mają własny przekrój poprzeczny i przyjmują ``thickness``
_SHAPES_WITH_THICKNESS = frozenset({"filament", "torus", "disk"})


def sample_positions(
    name: str,
    rng: np.random.Generator,
    n: int,
    radius: float,
    thickness: float = 0.08,
    flatten: float = 1.0,
) -> np.ndarray:
    """Wylosuj położenia startowe zadanego kształtu.

    ``flatten`` skaluje wyłącznie współrzędną z i działa na KAŻDY kształt, już po
    jego wylosowaniu. Jeden mnożnik zamiast parametru proporcji w każdej funkcji
    osobno: kula staje się plackiem albo cygarem, kostka płytą albo słupem, walec
    krążkiem albo prętem. Rozdzielenie rozmiaru (``radius``) od proporcji
    (``flatten``) ma tę zaletę, że zmiana kształtu nie zmienia przy okazji skali,
    więc porównania między biegami pozostają uczciwe.
    """
    try:
        shape = _SHAPES[name]
    except KeyError:
        raise KeyError(f"nieznany kształt: {name!r}; dostępne: {sorted(_SHAPES)}") from None
    args = (rng, n, radius, thickness) if name in _SHAPES_WITH_THICKNESS else (rng, n, radius)
    positions = np.array(shape(*args), dtype=np.float64)
    if abs(flatten - 1.0) > 1e-12:
        positions[:, 2] *= flatten
    return np.ascontiguousarray(positions)


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


#: Ile cząstek wystarcza do oszacowania energii potencjalnej kształtu. |U| zależy
#: od ROZKŁADU masy, nie od liczby próbek, więc podpróbka nosząca całą masę układu
#: daje tę samą wartość — a koszt O(m²) trzyma się wtedy w ułamku sekundy zamiast
#: rosnąć do minut przy stu tysiącach cząstek.
_VIRIAL_SAMPLE = 3000


def potential_energy(
    positions: np.ndarray, masses: np.ndarray, G: float, softening: float
) -> float:
    """|U| kształtu, oszacowane na podpróbce dokładnym sumowaniem po parach.

    Używa tego samego backendu i tej samej konwencji softeningu, którą liczy
    symulacja, więc wirial zadany na starcie zgadza się z wiriałem, który potem
    pokazuje diagnostyka. Własna implementacja sumy po parach rozjechałaby się
    z nią przy pierwszej zmianie konwencji ε.
    """
    from bone.backends.exact import ExactNumpy

    n = positions.shape[0]
    if n > _VIRIAL_SAMPLE:
        rng = np.random.default_rng(0)
        pick = rng.choice(n, size=_VIRIAL_SAMPLE, replace=False)
        positions = positions[pick]
        # podpróbka musi nieść CAŁĄ masę układu, inaczej oszacowałaby potencjał
        # układu o masie m/N razy mniejszej, czyli za mały (N/m)² razy
        masses = np.full(_VIRIAL_SAMPLE, masses.sum() / _VIRIAL_SAMPLE)

    backend = ExactNumpy()
    field = backend.compute(np.ascontiguousarray(positions), masses, G, softening)
    backend.close()
    return abs(field.energy(masses))


def _scale_to_virial(
    rotation_part: np.ndarray,
    dispersion_part: np.ndarray,
    masses: np.ndarray,
    c: float,
    target_kinetic: float,
) -> np.ndarray:
    """Dobierz mnożnik dyspersji tak, by energia kinetyczna trafiła w cel.

    Rozwiązywane numerycznie, a nie wzorem K = 3/2·Mσ², z dwóch powodów: energia
    kinetyczna jest relatywistyczna ((γ−1)mc², nie ½mv²), a rotacja wnosi część
    energii, której skalowanie dyspersji nie dotyczy. K(s) jest rosnąca, więc
    bisekcja jest tu i wystarczająca, i odporna — w przeciwieństwie do wzoru
    nierelatywistycznego, który przy β ≈ 0,5 myli się o kilkanaście procent.
    """
    def kinetic(scale: float) -> float:
        velocities = rotation_part + scale * dispersion_part
        speed = np.linalg.norm(velocities, axis=1)
        limit = 0.95 * c
        hot = speed > limit
        if np.any(hot):
            velocities = velocities.copy()
            velocities[hot] *= (limit / speed[hot])[:, None]
        return float(sr.kinetic_energy(masses, sr.momentum(masses, velocities, c), c).sum())

    if kinetic(0.0) >= target_kinetic:
        return np.zeros_like(dispersion_part)

    lo, hi = 0.0, 1.0
    while kinetic(hi) < target_kinetic and hi < 1e6:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kinetic(mid) < target_kinetic:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * dispersion_part


def make_state(cfg: Config) -> State:
    sp, ph = cfg.spawn, cfg.physics
    rng = np.random.default_rng(sp.seed)
    n = int(sp.n_particles)

    positions = sample_positions(sp.geometry, rng, n, sp.radius, sp.thickness, sp.flatten)
    # rozrzut mas jest tylko kosmetyczny; suma jest przeskalowana dokładnie do
    # zamawianej masy układu, żeby liczba cząstek nie wpływała na dynamikę
    masses = np.maximum(rng.normal(1.0, sp.mass_spread, size=n), 0.05)
    masses *= sp.total_mass / masses.sum()

    rotation_part = np.zeros((n, 3), dtype=np.float64)
    if abs(sp.rotation) > 1e-9:
        com = np.average(positions, axis=0, weights=masses)
        rel = positions - com
        axis = np.array([0.0, 0.0, 1.0])
        tangent = np.cross(axis, rel)
        norm = np.linalg.norm(tangent, axis=1, keepdims=True)
        # cząstki na osi obrotu nie mają zdefiniowanej stycznej — zostają w spoczynku
        unit = np.divide(tangent, norm, out=np.zeros_like(tangent), where=norm > 1e-12)
        v_circ = circular_speed(positions, masses, ph.G, ph.softening)
        rotation_part = (sp.rotation * v_circ)[:, None] * unit

    if sp.virial > 0.0:
        # dyspersja dobrana do energii potencjalnej TEGO kształtu, nie do wpisanej
        # liczby — dopiero wtedy dwa różne kształty można ze sobą porównywać
        target = 0.5 * sp.virial * potential_energy(positions, masses, ph.G, ph.softening)
        shape_noise = rng.normal(0.0, 1.0, size=(n, 3))
        dispersion_part = _scale_to_virial(rotation_part, shape_noise, masses, ph.c, target)
    elif abs(sp.temperature) > 1e-9:
        dispersion_part = rng.normal(0.0, sp.temperature * ph.c / np.sqrt(3.0), size=(n, 3))
    else:
        dispersion_part = np.zeros((n, 3), dtype=np.float64)

    velocities = rotation_part + dispersion_part

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
