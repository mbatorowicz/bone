"""Relatywistyczny leapfrog na pędzie: dp/dt = F, v = p c² / E."""

from __future__ import annotations

import numpy as np

from bone.domain.mapping import momentum_from_v, rest_mass, velocity_from_p
from bone.domain.universe import Universe
from bone.physics.forces import compute_forces
from bone.physics.neighbors import NeighborSet, get_or_build_neighbors


def _sanitize(u: Universe) -> None:
    bad_p = ~np.isfinite(u.positions).all(axis=1)
    bad_v = ~np.isfinite(u.velocities).all(axis=1)
    bad = bad_p | bad_v
    if np.any(bad):
        u.positions[bad] = np.nan_to_num(u.positions[bad], nan=0.0)
        u.velocities[bad] = 0.0


def leapfrog_step(
    universe: Universe, forces: np.ndarray | None = None
) -> tuple[np.ndarray, NeighborSet]:
    phys = universe.config.physics
    dt = phys.dt
    c = phys.c
    alive = universe.alive_mask
    _sanitize(universe)
    m = rest_mass(universe.mass)

    if forces is None:
        forces = compute_forces(universe, neighbors=get_or_build_neighbors(universe))
    forces = np.nan_to_num(forces, nan=0.0, posinf=phys.force_cap, neginf=-phys.force_cap)
    forces[~alive] = 0.0

    # p = γ m v
    p = momentum_from_v(m, universe.velocities, c)
    # kick 1/2
    p[alive] += 0.5 * dt * forces[alive]
    universe.velocities[alive] = velocity_from_p(m[alive], p[alive], c)

    # drift
    universe.positions[alive] += dt * universe.velocities[alive]
    _sanitize(universe)

    neighbors = get_or_build_neighbors(universe)
    forces_new = compute_forces(universe, neighbors=neighbors)
    forces_new = np.nan_to_num(
        forces_new, nan=0.0, posinf=phys.force_cap, neginf=-phys.force_cap
    )
    forces_new[~alive] = 0.0

    # kick 1/2 — p zaktualizowane z nowym v po dryfie
    p = momentum_from_v(m, universe.velocities, c)
    p[alive] += 0.5 * dt * forces_new[alive]
    universe.velocities[alive] = velocity_from_p(m[alive], p[alive], c)
    universe.velocities[~alive] = 0.0

    universe.t += dt
    universe.step += 1
    universe.neighbors = neighbors
    return forces_new, neighbors
