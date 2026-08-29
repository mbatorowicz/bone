"""Fizyczny moment pędu i metryka spłaszczenia (tensor bezwładności)."""

from __future__ import annotations

import numpy as np

from bone.mapping import gravitational_mass
from bone.traits import Universe


def angular_momentum(universe: Universe) -> tuple[np.ndarray, np.ndarray, float]:
    """
    L = Σ m (r − COM) × v  względem środka masy żywych.
    Zwraca (L_vec, L_hat, |L|). Fallback osi gdy |L|≈0: średnia predyspozycja, potem ẑ.
    """
    alive = universe.alive_mask
    if not np.any(alive):
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64), z, 0.0

    m = gravitational_mass(universe.traits, universe.config)[alive]
    pos = universe.positions[alive]
    vel = universe.velocities[alive]
    mtot = float(m.sum()) + 1e-18
    com = (m[:, None] * pos).sum(axis=0) / mtot
    r = pos - com
    L = np.sum(m[:, None] * np.cross(r, vel), axis=0)
    Lmag = float(np.linalg.norm(L))
    if Lmag > 1e-12:
        return L, L / Lmag, Lmag

    pred = universe.traits["predisposition"][alive]
    axis = pred.mean(axis=0)
    an = float(np.linalg.norm(axis))
    if an > 1e-9:
        return L, axis / an, Lmag
    return L, np.array([0.0, 0.0, 1.0], dtype=np.float64), Lmag


def global_spin_axis(universe: Universe) -> np.ndarray:
    """Jednostkowa oś do kicka IC / dyssypacji — L_hat z ruchu lub predyspozycji."""
    _L, L_hat, _mag = angular_momentum(universe)
    return L_hat


def mean_predisposition_axis(traits: np.ndarray, alive: np.ndarray | None = None) -> np.ndarray:
    """Średnia predyspozycja jako spójna oś L0 przy starcie."""
    pred = traits["predisposition"]
    if alive is not None:
        pred = pred[alive]
    axis = pred.mean(axis=0)
    n = float(np.linalg.norm(axis))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / n


def flattening_ratio(universe: Universe) -> float:
    """
    c/a z wartości własnych tensora bezwładności względem COM (masy).
    1 ≈ kula, →0 silnie spłaszczony dysk. Przy N<3 zwraca 1.
    """
    alive = universe.alive_mask
    if int(alive.sum()) < 3:
        return 1.0
    m = gravitational_mass(universe.traits, universe.config)[alive]
    pos = universe.positions[alive]
    mtot = float(m.sum()) + 1e-18
    com = (m[:, None] * pos).sum(axis=0) / mtot
    r = pos - com
    # I_ij = Σ m (r² δ_ij − r_i r_j)
    r2 = np.sum(r * r, axis=1)
    I = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            I[a, b] = float(np.sum(m * ((a == b) * r2 - r[:, a] * r[:, b])))
    try:
        eig = np.linalg.eigvalsh(I)
    except np.linalg.LinAlgError:
        return 1.0
    eig = np.sort(np.maximum(eig, 0.0))
    # dla cienkiego dysku I_max ≈ I_mid >> I_min; c/a ~ sqrt(I_min/I_max) w przybliżeniu elipsoidy
    a = float(eig[2])
    c = float(eig[0])
    if a < 1e-18:
        return 1.0
    return float(np.clip(np.sqrt(c / a), 0.0, 1.0))
