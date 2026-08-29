"""Bryły startowe — otwarta przestrzeń, bez klatki."""

from __future__ import annotations

import numpy as np

from bone.config.schema import SpawnConfig

GEOMETRY_NAMES = (
    "cube_cloud",
    "sphere_shell",
    "ball",
    "disk",
    "filament",
    "two_clumps",
    "lattice",
    "gaussian",
    "donut",
    "sphere_band",
)


def sample_positions(sp: SpawnConfig, rng: np.random.Generator) -> np.ndarray:
    n = int(sp.n_particles)
    g = int(np.clip(sp.geometry, 0, len(GEOMETRY_NAMES) - 1))
    s = float(sp.spacing)

    if g == 0:  # cube_cloud — chmura w sześcianie (nie ściany symulacji)
        return rng.uniform(-5 * s, 5 * s, size=(n, 3))

    if g == 1:  # sphere shell
        vec = rng.normal(0, 1, size=(n, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        return vec * (6.0 * s)

    if g == 2:  # ball
        vec = rng.normal(0, 1, size=(n, 3))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        rad = (rng.random(n) ** (1 / 3)) * 5.0 * s
        return vec * rad[:, None]

    if g == 3:  # disk
        rad = np.sqrt(rng.random(n)) * 8.0 * s
        ang = rng.uniform(0, 2 * np.pi, n)
        z = rng.normal(0, 0.25 * s, n)
        return np.column_stack([rad * np.cos(ang), rad * np.sin(ang), z])

    if g == 4:  # filament
        t = rng.uniform(-8 * s, 8 * s, n)
        return np.column_stack(
            [t, rng.normal(0, 0.8 * s, n), rng.normal(0, 0.8 * s, n)]
        )

    if g == 5:  # two clumps
        a = rng.normal([-5 * s, 0, 0], 1.4 * s, size=(n // 2, 3))
        b = rng.normal([5 * s, 0, 0], 1.4 * s, size=(n - n // 2, 3))
        return np.vstack([a, b])

    if g == 6:  # lattice with jitter
        side = int(np.ceil(n ** (1 / 3)))
        xs = np.linspace(-4 * s, 4 * s, side)
        grid = np.array(np.meshgrid(xs, xs, xs, indexing="ij")).reshape(3, -1).T
        if grid.shape[0] < n:
            extra = rng.uniform(-4 * s, 4 * s, size=(n - grid.shape[0], 3))
            grid = np.vstack([grid, extra])
        pos = grid[:n] + rng.normal(0, 0.15 * s, size=(n, 3))
        return pos

    if g == 7:  # gaussian
        return rng.normal(0.0, 4.0 * s, size=(n, 3))

    if g == 8:  # donut / torus
        R, r = 7.0 * s, 2.2 * s
        theta = rng.uniform(0, 2 * np.pi, n)
        phi = rng.uniform(0, 2 * np.pi, n)
        rr = r * (0.55 + 0.45 * rng.random(n))
        x = (R + rr * np.cos(phi)) * np.cos(theta)
        y = (R + rr * np.cos(phi)) * np.sin(theta)
        z = rr * np.sin(phi) * 0.4
        return np.column_stack([x, y, z])

    # sphere_band
    vec = rng.normal(0, 1, size=(n, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
    lat = np.abs(vec[:, 2])
    keep = lat < 0.45
    if keep.sum() < n // 3:
        keep = np.argsort(lat)[: max(n // 2, 1)]
        mask = np.zeros(n, dtype=bool)
        mask[keep] = True
        keep = mask
    band = vec[keep]
    if band.shape[0] < n:
        extra = rng.normal(0, 1, size=(n - band.shape[0], 3))
        extra /= np.linalg.norm(extra, axis=1, keepdims=True) + 1e-12
        band = np.vstack([band, extra])
    return band[:n] * (6.5 * s)
