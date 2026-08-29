"""Metryki kinematyki SR — bez Gini/wealth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bone.domain.mapping import gamma_from_v, rest_mass
from bone.domain.universe import Universe


def half_mass_radius(pos: np.ndarray, mass: np.ndarray) -> float:
    if pos.size == 0:
        return 0.0
    w = np.maximum(mass, 1e-9)
    com = np.average(pos, axis=0, weights=w)
    r = np.linalg.norm(pos - com, axis=1)
    order = np.argsort(r)
    cum = np.cumsum(mass[order])
    total = cum[-1]
    if total <= 0:
        return 0.0
    k = min(int(np.searchsorted(cum, 0.5 * total)), order.size - 1)
    return float(r[order[k]])


def angular_L(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> float:
    w = np.maximum(mass, 1e-9)
    com = np.average(pos, axis=0, weights=w)
    r = pos - com
    L = np.sum(mass[:, None] * np.cross(r, vel), axis=0)
    return float(np.linalg.norm(L))


def observe(universe: Universe) -> dict[str, float]:
    idx = np.flatnonzero(universe.alive_mask)
    n_alive = int(idx.size)
    c = universe.config.physics.c
    if n_alive == 0:
        return {
            "t": universe.t,
            "step": float(universe.step),
            "n_alive": 0.0,
            "r_half": 0.0,
            "collapse_ratio": 1.0,
            "L_mag": 0.0,
            "mean_speed": 0.0,
            "mean_gamma": 1.0,
            "max_gamma": 1.0,
            "v_over_c": 0.0,
        }
    pos = universe.positions[idx]
    vel = universe.velocities[idx]
    m = rest_mass(universe.mass[idx])
    r_half = half_mass_radius(pos, m)
    if getattr(universe, "_r_half0", None) in (None, 0):
        universe._r_half0 = r_half  # type: ignore[attr-defined]
    r0 = float(getattr(universe, "_r_half0", r_half) or r_half)
    speed = np.linalg.norm(vel, axis=1)
    g = gamma_from_v(vel, c)
    return {
        "t": universe.t,
        "step": float(universe.step),
        "n_alive": float(n_alive),
        "r_half": r_half,
        "collapse_ratio": float(r_half / (r0 + 1e-9)),
        "L_mag": angular_L(pos, vel, m),
        "mean_speed": float(speed.mean()),
        "mean_gamma": float(g.mean()),
        "max_gamma": float(g.max()),
        "v_over_c": float(speed.mean() / (c + 1e-12)),
    }


@dataclass
class MetricsTracker:
    history: list[dict[str, float]] = field(default_factory=list)

    def observe(self, universe: Universe) -> dict[str, float]:
        row = observe(universe)
        self.history.append(row)
        return row
