"""Spawn + kinematyka SR."""

from __future__ import annotations

import numpy as np

from bone.config.schema import AppConfig
from bone.domain.mapping import gamma_from_v, momentum_from_v, velocity_from_p
from bone.domain.universe import spawn


def test_spawn_finite_all_geoms():
    for g in range(10):
        cfg = AppConfig.from_flat({"n_particles": 200, "geometry": g, "seed": g + 1})
        u = spawn(cfg)
        assert u.n == 200
        assert np.isfinite(u.positions).all()
        assert np.all(u.mass > 0)


def test_momentum_roundtrip():
    m = np.array([1.0, 2.0])
    c = 10.0
    v = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    p = momentum_from_v(m, v, c)
    v2 = velocity_from_p(m, p, c)
    assert np.allclose(v, v2, rtol=1e-5)
    g = gamma_from_v(v, c)
    assert np.all(g >= 1.0)
