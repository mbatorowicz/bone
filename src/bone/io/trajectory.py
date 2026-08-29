"""Chunkowana trajektoria."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bone.domain.universe import Universe


class TrajectoryWriter:
    def __init__(self, out_dir: str | Path, chunk_size: int = 50, stride: int = 4):
        self.out = Path(out_dir)
        self.frames_dir = self.out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.stride = max(1, stride)
        self._buf_pos: list[np.ndarray] = []
        self._buf_c: list[np.ndarray] = []
        self._times: list[float] = []
        self._chunk_i = 0
        self.n_frames = 0

    def add(self, universe: Universe) -> None:
        p = universe.positions[:: self.stride].astype(np.float32)
        speed = np.linalg.norm(universe.velocities[:: self.stride], axis=1)
        cmin, cmax = float(speed.min()) if speed.size else 0.0, float(speed.max()) if speed.size else 1.0
        c = ((speed - cmin) / (cmax - cmin + 1e-12)).astype(np.float32)
        self._buf_pos.append(p)
        self._buf_c.append(c)
        self._times.append(float(universe.t))
        self.n_frames += 1
        if len(self._buf_pos) >= self.chunk_size:
            self.flush_chunk()

    def flush_chunk(self) -> None:
        if not self._buf_pos:
            return
        path = self.frames_dir / f"chunk_{self._chunk_i:04d}.npz"
        np.savez_compressed(
            path,
            positions=np.stack(self._buf_pos),
            colors=np.stack(self._buf_c),
            times=np.asarray(self._times, dtype=np.float64),
        )
        self._chunk_i += 1
        self._buf_pos.clear()
        self._buf_c.clear()
        self._times.clear()
        self._write_meta()

    def close(self) -> None:
        self.flush_chunk()
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "n_frames": self.n_frames,
            "n_chunks": self._chunk_i,
            "stride": self.stride,
        }
        (self.out / "trajectory_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )


def load_frame(out_dir: str | Path, index: int, stride: int = 4) -> dict | None:
    out = Path(out_dir)
    meta_path = out / "trajectory_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if index < 0 or index >= meta["n_frames"]:
        return None
    # znajdź chunk
    chunk_size = 50
    for p in sorted((out / "frames").glob("chunk_*.npz")):
        data = np.load(p)
        times = data["times"]
        n = int(times.shape[0])
        if index < n:
            pos = data["positions"][index]
            col = data["colors"][index]
            half = float(np.max(np.abs(pos)) * 1.05 + 1.0)
            return {
                "i": index,
                "t": float(times[index]),
                "half": half,
                "x": pos[:, 0].tolist(),
                "y": pos[:, 1].tolist(),
                "z": pos[:, 2].tolist(),
                "c": col.tolist(),
            }
        index -= n
    return None
