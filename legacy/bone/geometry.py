"""Geometrie początkowe rozmieszczenia punktów."""

from __future__ import annotations

import numpy as np

GEOMETRY_NAMES: tuple[str, ...] = (
    "cube",           # 0 — siatka sześcienna
    "sphere",         # 1 — kula wypełniona
    "shell",          # 2 — powłoka sferyczna
    "disk",           # 3 — dysk / galaktyka
    "spiral",         # 4 — spiralne ramię
    "filament",       # 5 — włókno / filament kosmiczny
    "twins",          # 6 — dwa skupiska (zderzenie)
    "gaussian",       # 7 — chmura Gaussa
    "donut",          # 8 — torus / donut
    "sphere_band",    # 9 — pas równikowy na sferze
)

GEOMETRY_LABELS: dict[str, str] = {
    "cube": "Szescian (siatka)",
    "sphere": "Kula wypelniona",
    "shell": "Powloka sferyczna",
    "disk": "Dysk galaktyczny",
    "spiral": "Spirala",
    "filament": "Filament",
    "twins": "Dwa skupiska",
    "gaussian": "Chmura Gaussa",
    "donut": "Donut (torus)",
    "sphere_band": "Pas na sferze",
}


def resolve_geometry(geom: str | int) -> str:
    if isinstance(geom, (int, float, np.integer, np.floating)):
        idx = int(round(float(geom))) % len(GEOMETRY_NAMES)
        return GEOMETRY_NAMES[idx]
    name = str(geom).strip().lower()
    if name.isdigit():
        return GEOMETRY_NAMES[int(name) % len(GEOMETRY_NAMES)]
    if name not in GEOMETRY_LABELS:
        return "cube"
    return name


def _scale_to_box(pos: np.ndarray, half: float) -> np.ndarray:
    """Przeskaluj chmurę tak, by mieściła się w [-half, half]."""
    if pos.size == 0:
        return pos
    m = np.max(np.abs(pos))
    if m < 1e-12:
        return pos
    return pos * (0.92 * half / m)


def generate_positions(
    geom: str | int,
    *,
    grid_n: int,
    spacing: float,
    n_particles: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Zwróć (N, 3) pozycji.
    cube → N = grid_n^3; pozostałe → N = n_particles.
    """
    name = resolve_geometry(geom)
    half = 0.5 * (grid_n - 1) * spacing
    half = max(half, 4.0)
    n = int(max(8, n_particles))

    if name == "cube":
        coords = np.arange(grid_n, dtype=np.float64) * spacing
        coords -= coords.mean()
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    if name == "sphere":
        # losowe w kuli (odrzucanie)
        pts = []
        while len(pts) < n:
            batch = rng.uniform(-1, 1, size=(n, 3))
            mask = np.sum(batch**2, axis=1) <= 1.0
            pts.append(batch[mask])
        pos = np.vstack(pts)[:n] * half
        return pos

    if name == "shell":
        vec = rng.normal(size=(n, 3))
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-12)
        thickness = 0.08 * half
        rad = half * (0.85 + 0.1 * rng.random(n))
        return vec * rad[:, None] + rng.normal(0, thickness * 0.15, size=(n, 3))

    if name == "disk":
        r = half * np.sqrt(rng.random(n))
        phi = rng.uniform(0, 2 * np.pi, size=n)
        z = rng.normal(0, 0.08 * half, size=n)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    if name == "spiral":
        # 2-ramienna spirala w dysku
        t = rng.random(n)
        arm = rng.integers(0, 2, size=n)
        r = half * (0.15 + 0.85 * t)
        phi = 2.8 * t * 2 * np.pi + arm * np.pi + rng.normal(0, 0.15, size=n)
        z = rng.normal(0, 0.06 * half, size=n)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    if name == "filament":
        # długie włókno wzdłuż x + cienkie zagęszczenia
        x = rng.uniform(-half, half, size=n)
        y = rng.normal(0, 0.12 * half, size=n)
        z = rng.normal(0, 0.12 * half, size=n)
        # lokalne zgęstnienia (proto-klastry)
        knots = rng.uniform(-half * 0.8, half * 0.8, size=5)
        for k in knots:
            near = np.abs(x - k) < 0.25 * half
            y[near] *= 0.4
            z[near] *= 0.4
        return np.column_stack([x, y, z])

    if name == "twins":
        n1 = n // 2
        n2 = n - n1
        c1 = np.array([-0.45 * half, 0.0, 0.0])
        c2 = np.array([0.45 * half, 0.0, 0.0])
        s = 0.22 * half
        a = rng.normal(0, s, size=(n1, 3)) + c1
        b = rng.normal(0, s, size=(n2, 3)) + c2
        return np.vstack([a, b])

    if name == "donut":
        # torus: R = promień dużego pierścienia, a = grubość rurki
        R = 0.58 * half
        a = 0.24 * half
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
        rho = a * np.sqrt(rng.random(n))  # wypełniony przekrój
        x = (R + rho * np.cos(phi)) * np.cos(theta)
        y = (R + rho * np.cos(phi)) * np.sin(theta)
        z = rho * np.sin(phi)
        return np.column_stack([x, y, z])

    if name == "sphere_band":
        # pas równikowy na sferze (start „ze sfery”, ale w toroidalnym pasie)
        lat = rng.normal(0.0, 0.22, size=n)  # radians around equator
        lon = rng.uniform(0.0, 2.0 * np.pi, size=n)
        rad = half * (0.72 + 0.18 * rng.random(n))
        cl = np.cos(lat)
        return np.column_stack(
            [
                rad * cl * np.cos(lon),
                rad * cl * np.sin(lon),
                rad * np.sin(lat),
            ]
        )

    # gaussian
    pos = rng.normal(0, 0.35 * half, size=(n, 3))
    return _scale_to_box(pos, half)
