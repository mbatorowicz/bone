"""Kinematyka SR: konwersje, granice i odporność na skrajne pędy."""

from __future__ import annotations

import numpy as np
import pytest

from bone import relativity as sr


def test_momentum_velocity_roundtrip():
    rng = np.random.default_rng(0)
    c = 10.0
    m = rng.uniform(0.5, 2.0, size=500)
    direction = rng.normal(size=(500, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    speed = rng.uniform(0.0, 0.98, size=500) * c
    v = direction * speed[:, None]

    p = sr.momentum(m, v, c)
    assert np.allclose(sr.velocity(m, p, c), v, rtol=1e-12, atol=1e-12)


def test_gamma_matches_definition():
    c = 3.0
    m = np.array([1.0, 2.5])
    v = np.array([[0.0, 0.0, 0.9 * c], [0.6 * c, 0.0, 0.0]])
    p = sr.momentum(m, v, c)
    beta2 = np.sum(v * v, axis=1) / c**2
    assert np.allclose(sr.gamma(m, p, c), 1.0 / np.sqrt(1.0 - beta2), rtol=1e-12)


def test_absurd_momentum_never_produces_nan_or_superluminal_speed():
    """Kluczowa własność formułowania na pędzie: żaden clamp nie jest potrzebny.

    Nawet pęd 1e300 — który w naiwnym wzorze przepełnia kwadrat i daje NaN —
    zwraca skończoną prędkość nieprzekraczającą c.
    """
    c = 1.0
    m = np.array([1e-3, 1.0, 1.0])
    p = np.array([[1e12, 0.0, 0.0], [0.0, 1e30, 0.0], [0.0, 0.0, 1e300]])
    v = sr.velocity(m, p, c)
    speed = np.linalg.norm(v, axis=1)
    assert np.all(np.isfinite(speed))
    assert np.all(speed <= c)
    assert np.all(sr.gamma(m, p, c) >= 1.0)
    assert np.all(sr.speed_over_c(m, p, c) <= 1.0)


def test_speed_stays_strictly_below_c_across_realistic_range():
    """Dla każdego pędu mieszczącego się w sensownym zakresie mamy ostre |v| < c."""
    c = 12.0
    m = np.full(9, 0.7)
    magnitudes = 10.0 ** np.arange(-3, 6)
    p = np.zeros((9, 3))
    p[:, 0] = magnitudes
    speed = np.linalg.norm(sr.velocity(m, p, c), axis=1)
    assert np.all(speed < c)
    assert np.all(np.diff(speed) > 0)  # monotonicznie rośnie, nasyca się przy c


def test_speed_over_c_agrees_with_velocity():
    rng = np.random.default_rng(3)
    c = 7.0
    m = rng.uniform(0.2, 3.0, size=200)
    p = rng.normal(scale=5.0, size=(200, 3))
    direct = np.linalg.norm(sr.velocity(m, p, c), axis=1) / c
    assert np.allclose(sr.speed_over_c(m, p, c), direct, rtol=1e-12)


def test_superluminal_initial_condition_is_rejected():
    with pytest.raises(ValueError):
        sr.momentum(np.array([1.0]), np.array([[2.0, 0.0, 0.0]]), 1.0)


def test_kinetic_energy_reduces_to_classical_at_low_speed():
    c = 1000.0
    m = np.array([2.0])
    v = np.array([[1.0, 0.0, 0.0]])  # β = 1e-3
    p = sr.momentum(m, v, c)
    classical = 0.5 * m * 1.0
    assert sr.kinetic_energy(m, p, c) == pytest.approx(classical, rel=1e-5)
