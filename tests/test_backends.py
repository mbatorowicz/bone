"""Poprawność backendów: zgodność z definicją, trzecia zasada dynamiki,
izolowane brzegi i zgodność CPU↔GPU."""

from __future__ import annotations

import numpy as np
import pytest

from bone.backends import cuda_available
from bone.backends.exact import ExactNumpy, ExactTorch, exact_forces_for
from bone.backends.mesh import MeshBackend

G = 1.3
EPS = 0.4


def brute_force(x, m, G=G, eps=EPS):
    """Definicja wprost, bez żadnych sztuczek — wzorzec dla wszystkiego innego."""
    n = x.shape[0]
    force = np.zeros((n, 3))
    phi = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = x[i] - x[j]
            r = np.sqrt(d @ d + eps * eps)
            force[i] -= G * m[i] * m[j] * d / r**3
            phi[i] -= G * m[j] / r
    return force, phi


@pytest.fixture
def cloud():
    rng = np.random.default_rng(11)
    x = rng.normal(scale=3.0, size=(120, 3))
    m = rng.uniform(0.4, 2.2, size=120)
    return x, m


def test_exact_matches_definition(cloud):
    x, m = cloud
    field = ExactNumpy().compute(x, m, G, EPS)
    ref_force, ref_phi = brute_force(x, m)
    assert np.allclose(field.force, ref_force, rtol=1e-10, atol=1e-12)
    assert np.allclose(field.potential, ref_phi, rtol=1e-10, atol=1e-12)


def test_exact_obeys_newtons_third_law(cloud):
    """Suma sił wewnętrznych musi być zerem — inaczej pęd nie ma prawa się zachować."""
    x, m = cloud
    field = ExactNumpy().compute(x, m, G, EPS)
    scale = np.linalg.norm(field.force, axis=1).sum()
    assert np.linalg.norm(field.force.sum(axis=0)) / scale < 1e-13


def test_exact_energy_matches_pair_sum(cloud):
    x, m = cloud
    field = ExactNumpy().compute(x, m, G, EPS)
    pair_sum = 0.0
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            d = x[i] - x[j]
            pair_sum -= G * m[i] * m[j] / np.sqrt(d @ d + EPS * EPS)
    assert field.energy(m) == pytest.approx(pair_sum, rel=1e-10)


def test_subset_forces_match_full(cloud):
    x, m = cloud
    full = ExactNumpy().compute(x, m, G, EPS).force
    rows = np.array([0, 7, 33, 119])
    assert np.allclose(exact_forces_for(x, m, G, EPS, rows), full[rows], rtol=1e-12)


def test_mesh_is_not_periodic():
    """Serce izolowanych brzegów.

    Dwie cząstki przy przeciwległych krawędziach pudła. Przy periodycznym FFT
    krótszą drogą jest przejście przez ścianę i siła wskazywałaby NA ZEWNĄTRZ.
    Metoda Hockneya z zerowym paddingiem musi dać przyciąganie do wnętrza.
    """
    x = np.array([[-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    m = np.array([1.0, 1.0])
    mesh = MeshBackend(grid=64, device="cpu", dtype="float64")
    force = mesh.compute(x, m, G, EPS).force
    # lewa cząstka ciągnięta w prawo, prawa w lewo
    assert force[0, 0] > 0.0
    assert force[1, 0] < 0.0
    exact = ExactNumpy().compute(x, m, G, EPS).force
    assert np.sign(force[0, 0]) == np.sign(exact[0, 0])
    assert force[0, 0] == pytest.approx(exact[0, 0], rel=0.25)


def test_mesh_conserves_momentum():
    rng = np.random.default_rng(5)
    x = rng.normal(scale=4.0, size=(4000, 3))
    m = rng.uniform(0.5, 1.5, size=4000)
    force = MeshBackend(grid=64, device="cpu", dtype="float64").compute(x, m, G, EPS).force
    scale = np.linalg.norm(force, axis=1).sum()
    assert np.linalg.norm(force.sum(axis=0)) / scale < 1e-6


def test_mesh_error_is_small_and_measurable():
    """PM jest przybliżeniem — wymagamy, żeby błąd był mały ORAZ mierzalny."""
    rng = np.random.default_rng(7)
    x = rng.normal(scale=4.0, size=(6000, 3))
    m = rng.uniform(0.5, 1.5, size=6000)
    mesh_force = MeshBackend(grid=96, device="cpu", dtype="float64").compute(x, m, G, 0.6).force
    rows = rng.choice(x.shape[0], size=400, replace=False)
    reference = exact_forces_for(x, m, G, 0.6, rows)
    typical = np.sqrt(np.mean(np.linalg.norm(reference, axis=1) ** 2))
    rms = np.sqrt(np.mean(np.linalg.norm(mesh_force[rows] - reference, axis=1) ** 2)) / typical
    assert rms < 0.10, f"błąd siły PM = {rms:.2%}"


def test_mesh_refits_box_when_cloud_expands():
    mesh = MeshBackend(grid=32, device="cpu", dtype="float64")
    m = np.ones(64)
    rng = np.random.default_rng(1)
    small = rng.normal(scale=1.0, size=(64, 3))
    mesh.compute(small, m, G, EPS)
    first = mesh.refits
    mesh.compute(small * 40.0, m, G, EPS)
    assert mesh.refits > first


@pytest.mark.skipif(not cuda_available(), reason="CUDA niedostępna")
def test_exact_cuda_matches_cpu(cloud):
    x, m = cloud
    cpu = ExactNumpy().compute(x, m, G, EPS)
    gpu = ExactTorch(device="cuda", dtype="float32").compute(x, m, G, EPS)
    scale = np.linalg.norm(cpu.force, axis=1).max()
    assert np.max(np.linalg.norm(gpu.force - cpu.force, axis=1)) / scale < 2e-4


@pytest.mark.skipif(not cuda_available(), reason="CUDA niedostępna")
def test_mesh_cuda_matches_cpu():
    rng = np.random.default_rng(9)
    x = rng.normal(scale=3.0, size=(2000, 3))
    m = rng.uniform(0.5, 1.5, size=2000)
    cpu = MeshBackend(grid=48, device="cpu", dtype="float64").compute(x, m, G, EPS).force
    gpu = MeshBackend(grid=48, device="cuda", dtype="float32").compute(x, m, G, EPS).force
    scale = np.linalg.norm(cpu, axis=1).max()
    assert np.max(np.linalg.norm(gpu - cpu, axis=1)) / scale < 5e-3
