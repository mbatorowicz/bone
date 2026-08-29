"""Zapis trajektorii w chunkach (append) + scalamie do trajectory.npz."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHUNK_SIZE = 50


class TrajectoryWriter:
    """RAM buffer → chunk_XXX.npz; na końcu merge → trajectory.npz."""

    def __init__(self, out_dir: Path, *, chunk_size: int = CHUNK_SIZE) -> None:
        self.out_dir = Path(out_dir)
        self.frames_dir = self.out_dir / "frames"
        self.chunk_size = max(1, int(chunk_size))
        self.pending_pos: list[np.ndarray] = []
        self.pending_times: list[float] = []
        self.pending_colors: list[np.ndarray] = []
        self.chunk_id = 0
        self.total_frames = 0
        self.half: float | None = None
        self.point_stride = 1
        self._chunk_files: list[str] = []

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        # kontynuacja: policz istniejące chunki
        existing = sorted(self.frames_dir.glob("chunk_*.npz"))
        if existing:
            self.chunk_id = len(existing)
            for p in existing:
                self._chunk_files.append(p.name)
                with np.load(p) as data:
                    self.total_frames += int(data["times"].shape[0])

    def add(
        self,
        positions: np.ndarray,
        t: float,
        colors: np.ndarray,
        *,
        half: float | None = None,
    ) -> None:
        self.pending_pos.append(np.asarray(positions, dtype=np.float32))
        self.pending_times.append(float(t))
        self.pending_colors.append(np.asarray(colors, dtype=np.float32))
        if half is not None:
            self.half = float(half)
        if len(self.pending_pos) >= self.chunk_size:
            self.flush_chunk(force=True)

    def flush_chunk(self, *, force: bool = False) -> None:
        """Zapisz pending do chunku gdy pełny (lub force=True na Stop)."""
        if not self.pending_pos:
            return
        if not force and len(self.pending_pos) < self.chunk_size:
            self._write_meta()
            return
        name = f"chunk_{self.chunk_id:04d}.npz"
        path = self.frames_dir / name
        np.savez(
            path,
            positions=np.stack(self.pending_pos, axis=0),
            times=np.asarray(self.pending_times, dtype=np.float64),
            colors=np.stack(self.pending_colors, axis=0),
        )
        n = len(self.pending_pos)
        self.total_frames += n
        self._chunk_files.append(name)
        self.chunk_id += 1
        self.pending_pos.clear()
        self.pending_times.clear()
        self.pending_colors.clear()
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "n_frames": self.total_frames + len(self.pending_pos),
            "chunks": list(self._chunk_files),
            "pending": len(self.pending_pos),
            "half": self.half,
            "point_stride": self.point_stride,
            "chunk_size": self.chunk_size,
        }
        (self.out_dir / "trajectory_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

    def finalize(self) -> Path | None:
        """Dopisz ostatni chunk i scal do trajectory.npz."""
        self.flush_chunk(force=True)
        # pending już wchunkowany; total_frames obejmuje zapisane
        n_pending_meta = self.total_frames + len(self.pending_pos)
        if n_pending_meta == 0 and not self._chunk_files:
            return None
        all_pos: list[np.ndarray] = []
        all_t: list[np.ndarray] = []
        all_c: list[np.ndarray] = []
        for name in self._chunk_files:
            with np.load(self.frames_dir / name) as data:
                all_pos.append(data["positions"])
                all_t.append(data["times"])
                all_c.append(data["colors"])
        if not all_pos:
            return None
        path = self.out_dir / "trajectory.npz"
        np.savez_compressed(
            path,
            positions=np.concatenate(all_pos, axis=0),
            times=np.concatenate(all_t, axis=0),
            colors=np.concatenate(all_c, axis=0),
        )
        self.total_frames = int(sum(a.shape[0] for a in all_pos))
        self._write_meta()
        return path


def load_trajectory_meta(out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    meta_path = out_dir / "trajectory_meta.json"
    traj_path = out_dir / "trajectory.npz"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}
    if traj_path.exists():
        with np.load(traj_path) as data:
            times = np.asarray(data["times"])
            pos = data["positions"]
            meta.setdefault("n_frames", int(times.shape[0]))
            meta.setdefault("n_points", int(pos.shape[1]))
            meta["times"] = times.tolist()
            if "half" not in meta or meta["half"] is None:
                meta["half"] = float(np.max(np.abs(pos[-1])) * 1.05 + 1.0)
    elif meta.get("chunks"):
        # zbierz times z chunków
        times_list: list[float] = []
        n_points = 0
        for name in meta["chunks"]:
            p = out_dir / "frames" / name
            if not p.exists():
                continue
            with np.load(p) as data:
                times_list.extend(data["times"].tolist())
                n_points = int(data["positions"].shape[1])
        meta["n_frames"] = len(times_list)
        meta["n_points"] = n_points
        meta["times"] = times_list
    else:
        meta.setdefault("n_frames", 0)
        meta.setdefault("times", [])
    return meta


def load_trajectory_frame(
    out_dir: Path,
    index: int,
    *,
    point_stride: int = 2,
) -> dict | None:
    """Jedna klatka do replay (z trajectory.npz lub chunków)."""
    out_dir = Path(out_dir)
    traj_path = out_dir / "trajectory.npz"
    if traj_path.exists():
        with np.load(traj_path) as data:
            f = int(data["times"].shape[0])
            if index < 0 or index >= f:
                return None
            pos = data["positions"][index]
            col = data["colors"][index]
            t = float(data["times"][index])
            return _pack_frame(pos, col, t, index, point_stride=point_stride)

    meta_path = out_dir / "trajectory_meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    chunks = meta.get("chunks") or []
    cursor = 0
    for name in chunks:
        p = out_dir / "frames" / name
        if not p.exists():
            continue
        with np.load(p) as data:
            n = int(data["times"].shape[0])
            if index < cursor + n:
                local = index - cursor
                return _pack_frame(
                    data["positions"][local],
                    data["colors"][local],
                    float(data["times"][local]),
                    index,
                    point_stride=point_stride,
                )
            cursor += n
    return None


def _pack_frame(
    pos: np.ndarray,
    col: np.ndarray,
    t: float,
    index: int,
    *,
    point_stride: int,
) -> dict:
    p = np.asarray(pos)[::point_stride]
    c = np.asarray(col)[::point_stride]
    cmin, cmax = float(c.min()), float(c.max())
    cn = (c - cmin) / (cmax - cmin + 1e-12)
    half = float(np.max(np.abs(p)) * 1.05 + 1.0) if p.size else 12.0
    return {
        "live": False,
        "i": int(index),
        "t": float(t),
        "step": int(index),
        "half": half,
        "n": int(p.shape[0]),
        "x": p[:, 0].astype(np.float32).tolist(),
        "y": p[:, 1].astype(np.float32).tolist(),
        "z": p[:, 2].astype(np.float32).tolist(),
        "c": cn.astype(np.float32).tolist(),
    }
