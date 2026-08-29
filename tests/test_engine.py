"""Kroki SR bez NaN, bez ścian."""

from __future__ import annotations

import numpy as np

from bone.config.schema import AppConfig
from bone.engine import Engine


def test_steps_finite_no_walls():
    cfg = AppConfig.from_flat(
        {
            "n_particles": 300,
            "geometry": 8,
            "seed": 3,
            "steps": 40,
            "orbital_seed": 0.3,
            "snapshot_every": 20,
            "traj_every": 100,
            "live_every": 100,
            "max_neighbors": 20,
            "G": 0.14,
        }
    )
    eng = Engine(cfg)
    r0 = np.linalg.norm(eng.universe.positions, axis=1).mean()
    eng.run()
    u = eng.universe
    assert np.isfinite(u.positions).all()
    assert np.isfinite(u.velocities).all()
    assert u.step >= 40
    # bez ścian pozycje mogą wyjść poza „kostkę” startową
    extent = np.max(np.abs(u.positions))
    assert extent > 0.0
    # układ nie powinien eksplodować do 100× skali w 40 krokach
    r1 = np.linalg.norm(u.positions, axis=1).mean()
    assert r1 < r0 * 20
