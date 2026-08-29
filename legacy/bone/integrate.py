"""Integracja ruchu: leapfrog / velocity Verlet z limitem v < c."""

from __future__ import annotations

import numpy as np

from bone.gravity import compute_forces
from bone.mapping import gravitational_mass
from bone.neighbors import NeighborSet, get_or_build_neighbors
from bone.orbits import stabilize_orbits
from bone.traits import Universe


def _clamp_velocities(velocities: np.ndarray, c: float) -> None:
    """Ogranicz |v| do 0.99 c; wyzeruj NaN/Inf (in-place)."""
    bad = ~np.isfinite(velocities)
    if np.any(bad):
        velocities[bad] = 0.0
    speeds = np.linalg.norm(velocities, axis=1)
    limit = 0.99 * c
    too_fast = np.isfinite(speeds) & (speeds > limit)
    if np.any(too_fast):
        velocities[too_fast] *= (limit / speeds[too_fast])[:, None]


def _sanitize_state(universe: Universe, half_extent: float) -> None:
    """Usuń NaN/Inf z pozycji/prędkości; przytnij ucieczkę poza domenę."""
    pos = universe.positions
    vel = universe.velocities
    bad_p = ~np.isfinite(pos).all(axis=1)
    bad_v = ~np.isfinite(vel).all(axis=1)
    bad = bad_p | bad_v
    if np.any(bad):
        pos[bad] = np.nan_to_num(pos[bad], nan=0.0, posinf=half_extent, neginf=-half_extent)
        vel[bad] = 0.0
    np.clip(pos, -half_extent * 4.0, half_extent * 4.0, out=pos)


def _finite_forces(forces: np.ndarray, max_f: float = 80.0) -> np.ndarray:
    out = np.nan_to_num(forces, nan=0.0, posinf=max_f, neginf=-max_f)
    np.clip(out, -max_f, max_f, out=out)
    return out


def leapfrog_step(
    universe: Universe,
    forces: np.ndarray | None = None,
) -> tuple[np.ndarray, NeighborSet]:
    """
    Jeden krok velocity Verlet + opcjonalna stabilizacja orbit (orbit_every).
    NeighborSet: lista Verlet (rebuild co neighbor_rebuild_every / drift).
    """
    cfg = universe.config
    dt = cfg.dt
    alive = universe.alive_mask
    half = 0.5 * (cfg.grid_n - 1) * cfg.spacing + cfg.wall_margin + 8.0
    _sanitize_state(universe, half)

    m = np.maximum(gravitational_mass(universe.traits, cfg), 1e-6)
    m = np.nan_to_num(m, nan=1.0, posinf=1e6, neginf=1.0)

    if forces is None:
        neigh0 = get_or_build_neighbors(universe)
        forces = compute_forces(universe, neighbors=neigh0)
    forces = _finite_forces(forces)

    acc = forces / m[:, None]
    acc[~alive] = 0.0
    # cap przyspieszenia — chroni przed wybuchem przy dużym sim_speed×dt
    max_a = float(max(cfg.c / max(dt, 1e-9) * 0.5, 20.0))
    np.clip(acc, -max_a, max_a, out=acc)

    # v(t + dt/2)
    universe.velocities[alive] += 0.5 * dt * acc[alive]
    _clamp_velocities(universe.velocities, cfg.c)

    # x(t + dt)
    universe.positions[alive] += dt * universe.velocities[alive]
    _sanitize_state(universe, half)

    neighbors = get_or_build_neighbors(universe)
    forces_new = _finite_forces(compute_forces(universe, neighbors=neighbors))
    acc_new = forces_new / m[:, None]
    acc_new[~alive] = 0.0
    np.clip(acc_new, -max_a, max_a, out=acc_new)

    # v(t + dt)
    universe.velocities[alive] += 0.5 * dt * acc_new[alive]
    _clamp_velocities(universe.velocities, cfg.c)

    orbit_every = int(max(1, getattr(cfg, "orbit_every", 2)))
    if (int(universe.step) + 1) % orbit_every == 0:
        stabilize_orbits(universe, neighbors=neighbors)
        _clamp_velocities(universe.velocities, cfg.c)
        _sanitize_state(universe, half)

    universe.velocities[~alive] = 0.0

    universe.t += dt
    universe.step += 1
    universe.neighbors = neighbors  # type: ignore[attr-defined]
    return forces_new, neighbors
