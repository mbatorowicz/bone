"""Checkpoint — config + r,v,m."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bone.config.schema import AppConfig
from bone.domain.universe import Universe


def save_checkpoint(universe: Universe, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "checkpoint.npz"
    np.savez_compressed(
        path,
        positions=universe.positions.astype(np.float32),
        velocities=universe.velocities.astype(np.float32),
        mass=universe.mass.astype(np.float32),
        alive=universe.alive.astype(np.uint8),
        t=np.array([universe.t], dtype=np.float64),
        step=np.array([universe.step], dtype=np.int64),
    )
    (out / "config.json").write_text(
        json.dumps(universe.config.to_flat(), indent=2), encoding="utf-8"
    )
    return path


def load_checkpoint(out_dir: str | Path, cfg: AppConfig | None = None) -> Universe:
    out = Path(out_dir)
    data = np.load(out / "checkpoint.npz", allow_pickle=False)
    cfg_path = out / "config.json"
    if cfg is None and cfg_path.exists():
        cfg = AppConfig.from_flat(json.loads(cfg_path.read_text(encoding="utf-8")))
    cfg = cfg or AppConfig()
    return Universe(
        positions=np.asarray(data["positions"], dtype=np.float64),
        velocities=np.asarray(data["velocities"], dtype=np.float64),
        mass=np.asarray(data["mass"], dtype=np.float64),
        alive=np.asarray(data["alive"], dtype=bool),
        config=cfg,
        t=float(data["t"][0]),
        step=int(data["step"][0]),
    )
