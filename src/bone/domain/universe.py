"""Universe — cząstki: masa spoczynkowa, r, v (pęd wyliczany)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bone.config.schema import AppConfig, SpawnConfig
from bone.domain.geometry import sample_positions


@dataclass
class Universe:
    positions: np.ndarray  # (N,3)
    velocities: np.ndarray  # (N,3) — |v| < c
    mass: np.ndarray  # (N,) masa spoczynkowa
    alive: np.ndarray  # (N,) bool
    config: AppConfig
    t: float = 0.0
    step: int = 0
    neighbors: object | None = None

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def alive_mask(self) -> np.ndarray:
        return self.alive


def spawn(cfg: AppConfig | None = None, rng: np.random.Generator | None = None) -> Universe:
    cfg = cfg or AppConfig()
    sp: SpawnConfig = cfg.spawn
    phys = cfg.physics
    rng = rng or np.random.default_rng(sp.seed)
    pos = sample_positions(sp, rng)
    n = pos.shape[0]
    mass = np.maximum(
        rng.normal(sp.mass_mean, sp.mass_sigma * sp.mass_mean, size=n), 0.05
    )
    vel = np.zeros((n, 3), dtype=np.float64)
    c = float(phys.c)

    if sp.orbital_seed > 1e-9:
        com = pos.mean(axis=0)
        r = pos - com
        axis = np.array([0.0, 0.0, 1.0])
        tang = np.cross(np.broadcast_to(axis, r.shape), r)
        tn = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        # Kepler-ish: v ~ sqrt(GM/r) capped as fraction of c
        rad = np.linalg.norm(r, axis=1) + 1e-6
        v_circ = np.sqrt(np.maximum(phys.G * mass.mean() * n * 0.15 / rad, 0.0))
        v_orb = np.minimum(sp.orbital_seed * c, v_circ + sp.orbital_seed * c * 0.5)
        vel = (v_orb[:, None]) * (tang / tn)

    if sp.thermal_seed > 1e-9:
        kick = rng.normal(0.0, sp.thermal_seed * c / np.sqrt(3.0), size=(n, 3))
        vel += kick

    # twardy limit |v| < 0.99 c
    sp_ = np.linalg.norm(vel, axis=1)
    lim = 0.99 * c
    too = sp_ > lim
    if np.any(too):
        vel[too] *= (lim / sp_[too])[:, None]

    return Universe(
        positions=pos.astype(np.float64),
        velocities=vel,
        mass=mass.astype(np.float64),
        alive=np.ones(n, dtype=bool),
        config=cfg,
    )
