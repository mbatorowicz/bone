"""Relatywistyczny leapfrog KDK na pędzie.

    p ← p + ½Δt F(x)
    x ← x + Δt v(p)
    p ← p + ½Δt F(x)

Siła z drugiego kopnięcia jest przenoszona do pierwszego kopnięcia następnego
kroku, więc na krok przypada dokładnie JEDNO liczenie sił — przy dokładnym
O(N²) to różnica dwukrotna w czasie działania.

Krok czasowy jest dobierany adaptacyjnie z kryterium przyspieszenia
Δt ≤ η√(ε/a_max), standardowego w kodach N-ciałowych. Poprzednia wersja mnożyła
Δt przez suwak „szybkość" bez żadnego kryterium: układ nie wybuchał tylko
dlatego, że relatywistyka nie pozwala przekroczyć c, ale trajektorie przestawały
cokolwiek znaczyć. Tutaj przyspieszenie symulacji zwiększa liczbę kroków na
klatkę, a nie błąd całkowania.
"""

from __future__ import annotations

import numpy as np

from bone import relativity as sr
from bone.backends.base import Backend, Field
from bone.config import PhysicsConfig
from bone.state import State


def choose_dt(state: State, forces: np.ndarray, phys: PhysicsConfig) -> float:
    """Δt = min(Δt_max, η√(ε/a_max))."""
    if not phys.adaptive_dt:
        return float(phys.dt_max)
    accel = np.linalg.norm(forces, axis=1) / sr.rest_mass(state.masses)
    a_max = float(accel.max()) if accel.size else 0.0
    if not np.isfinite(a_max) or a_max <= 0.0:
        return float(phys.dt_max)
    limit = phys.accuracy * np.sqrt(phys.softening / a_max)
    return float(min(phys.dt_max, limit))


def _field(backend: Backend, state: State, phys: PhysicsConfig) -> Field:
    return backend.compute(state.positions, state.masses, phys.G, phys.softening)


def ensure_field(backend: Backend, state: State, phys: PhysicsConfig) -> Field:
    """Zapewnij aktualną siłę w stanie (potrzebne po wczytaniu checkpointu)."""
    if state.forces is None or state.potential is None:
        field = _field(backend, state, phys)
        state.forces = field.force
        state.potential = field.potential
    return Field(force=state.forces, potential=state.potential)


def step(backend: Backend, state: State, phys: PhysicsConfig) -> float:
    """Wykonaj jeden krok. Zwraca użyty Δt."""
    field = ensure_field(backend, state, phys)
    dt = choose_dt(state, field.force, phys)
    half = 0.5 * dt

    state.momenta += half * field.force
    state.positions += dt * sr.velocity(state.masses, state.momenta, phys.c)

    field = _field(backend, state, phys)
    state.momenta += half * field.force

    state.forces = field.force
    state.potential = field.potential
    state.time += dt
    state.step += 1
    return dt
