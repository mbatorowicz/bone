"""Kinematyka szczególnej teorii względności.

Stanem cząstki jest pęd ``p``, nie prędkość. Dzięki temu

    γ = E/(mc²) = √(1 + (|p|/mc)²)

jest zawsze skończone i ≥ 1, niezależnie od tego jak duża jest siła. Żaden
clamp prędkości nie jest potrzebny — |v| → c asymptotycznie z samej definicji.
Wersja trzymająca ``v`` wymagała przycinania β² poniżej 1, co maskowało błędy
całkowania zamiast im zapobiegać.
"""

from __future__ import annotations

import numpy as np

MIN_MASS = 1e-12


def rest_mass(mass: np.ndarray) -> np.ndarray:
    return np.maximum(mass, MIN_MASS)


def _norm(vectors: np.ndarray) -> np.ndarray:
    """Długość wektorów odporna na przepełnienie.

    ``sqrt(sum(x²))`` przepełnia się już dla składowych rzędu 1e200, a wtedy
    wynik to ``inf`` i cała dalsza arytmetyka jest bezużyteczna. Skalowanie
    przez największą składową kosztuje jeden dodatkowy przebieg i usuwa problem.
    """
    scale = np.max(np.abs(vectors), axis=-1)
    safe = np.where(scale > 0.0, scale, 1.0)
    return scale * np.sqrt(np.sum((vectors / safe[..., None]) ** 2, axis=-1))


def _energy_over_c(mass: np.ndarray, momentum: np.ndarray, c: float) -> np.ndarray:
    """E/c = √(|p|² + (mc)²), liczone przez ``hypot`` — bez pośredniego kwadratu."""
    return np.hypot(_norm(momentum), rest_mass(mass) * c)


def gamma(mass: np.ndarray, momentum: np.ndarray, c: float) -> np.ndarray:
    """γ = E/(mc²) = √(|p|² + (mc)²)/(mc). Zawsze ≥ 1, nigdy NaN."""
    mc = rest_mass(mass) * c
    return _energy_over_c(mass, momentum, c) / mc


def velocity(mass: np.ndarray, momentum: np.ndarray, c: float) -> np.ndarray:
    """v = p c²/E = p c/√(|p|² + (mc)²).

    Liczone bez pośrednictwa γ, więc pozostaje poprawne nawet wtedy, gdy samo γ
    wykracza poza zakres liczb zmiennoprzecinkowych. Gwarancja: |v| ≤ c zawsze,
    a |v| < c dla każdego fizycznie sensownego pędu. Równość może wystąpić
    dopiero, gdy |p| przewyższa mc o tyle rzędów wielkości, że mc znika przy
    zaokrągleniu — i wtedy jest to najlepsze możliwe przybliżenie, a nie błąd.
    """
    return momentum * (c / _energy_over_c(mass, momentum, c))[..., None]


def momentum(mass: np.ndarray, vel: np.ndarray, c: float) -> np.ndarray:
    """p = γmv. Prędkości ≥ c są odrzucane — to błąd wywołującego, nie stan fizyczny."""
    m = rest_mass(mass)
    beta2 = np.sum(vel * vel, axis=-1) / (c * c)
    if np.any(beta2 >= 1.0):
        raise ValueError("prędkość początkowa ≥ c — pęd byłby nieskończony")
    return (m / np.sqrt(1.0 - beta2))[..., None] * vel


def kinetic_energy(mass: np.ndarray, momentum_: np.ndarray, c: float) -> np.ndarray:
    """T = (γ−1)mc². Energia spoczynkowa jest pominięta, bo jest stałą ruchu
    i zagłuszyłaby dryf o interesującej nas skali."""
    return (gamma(mass, momentum_, c) - 1.0) * rest_mass(mass) * c * c


def speed_over_c(mass: np.ndarray, momentum_: np.ndarray, c: float) -> np.ndarray:
    """β = |p|/√(|p|² + (mc)²)."""
    return _norm(momentum_) / _energy_over_c(mass, momentum_, c)
