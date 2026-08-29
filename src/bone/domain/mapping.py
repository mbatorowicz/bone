"""Masa i kinematyka SR."""

from __future__ import annotations

import numpy as np

from bone.config.schema import PhysicsConfig


def rest_mass(mass: np.ndarray) -> np.ndarray:
    return np.maximum(mass, 1e-6)


def gamma_from_v(velocities: np.ndarray, c: float) -> np.ndarray:
    v2 = np.sum(velocities * velocities, axis=1)
    beta2 = np.clip(v2 / (c * c), 0.0, 0.999999)
    return 1.0 / np.sqrt(1.0 - beta2)


def momentum_from_v(mass: np.ndarray, velocities: np.ndarray, c: float) -> np.ndarray:
    g = gamma_from_v(velocities, c)
    return (rest_mass(mass) * g)[:, None] * velocities


def velocity_from_p(mass: np.ndarray, p: np.ndarray, c: float) -> np.ndarray:
    """v = p c² / E, E = sqrt((pc)² + (mc²)²)."""
    m = rest_mass(mass)
    p2 = np.sum(p * p, axis=1)
    e = np.sqrt(p2 * c * c + (m * c * c) ** 2)
    return (p * (c * c)) / e[:, None]


def gravitational_mass(mass: np.ndarray, _phys: PhysicsConfig | None = None) -> np.ndarray:
    """W tej wersji masa graw = masa spoczynkowa."""
    return rest_mass(mass)
