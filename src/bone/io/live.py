"""Pack LIVE JSON (subsample)."""

from __future__ import annotations

import numpy as np

from bone.domain.universe import Universe


def pack_live(universe: Universe, point_stride: int | None = None) -> dict:
    stride = max(1, int(point_stride or universe.config.io.point_stride))
    p = np.ascontiguousarray(universe.positions[::stride], dtype=np.float32)
    v = np.ascontiguousarray(universe.velocities[::stride], dtype=np.float32)
    speed = np.linalg.norm(v, axis=1)
    cmin = float(speed.min()) if speed.size else 0.0
    cmax = float(speed.max()) if speed.size else 1.0
    cn = ((speed - cmin) / (cmax - cmin + 1e-12)).astype(np.float32)
    half = float(np.max(np.abs(p)) * 1.05 + 1.0) if p.size else 12.0
    return {
        "live": True,
        "t": float(universe.t),
        "step": int(universe.step),
        "half": half,
        "n": int(p.shape[0]),
        "x": p[:, 0].tolist(),
        "y": p[:, 1].tolist(),
        "z": p[:, 2].tolist(),
        "c": cn.tolist(),
    }
