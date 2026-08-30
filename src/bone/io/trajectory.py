"""Zapis trajektorii w kawałkach, z indeksem.

Indeks trzyma numer kawałka i pozycję w kawałku dla każdej klatki, więc odczyt
klatki to jedno otwarcie pliku. Poprzednia wersja przeglądała wszystkie kawałki
po kolei, a odtwarzacz pyta o klatkę kilka razy na sekundę — koszt rósł liniowo
z długością nagrania.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bone.config import Config
from bone.state import State

META = "trajectory.json"


class TrajectoryWriter:
    def __init__(self, out_dir: str | Path, chunk_size: int = 64, stride: int = 1) -> None:
        self.out = Path(out_dir)
        self.frames_dir = self.out / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = max(1, chunk_size)
        self.stride = max(1, stride)
        self._positions: list[np.ndarray] = []
        self._shades: list[np.ndarray] = []
        self._times: list[float] = []
        self._chunk = 0
        self.index: list[list[int]] = []  # [chunk, offset] dla każdej klatki

    @property
    def n_frames(self) -> int:
        return len(self.index)

    def add(self, state: State, cfg: Config) -> None:
        positions = np.ascontiguousarray(state.positions[:: self.stride], dtype=np.float32)
        shade = np.ascontiguousarray(
            state.speed_over_c(cfg.physics.c)[:: self.stride], dtype=np.float32
        )
        self.index.append([self._chunk, len(self._positions)])
        self._positions.append(positions)
        self._shades.append(shade)
        self._times.append(float(state.time))
        if len(self._positions) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._positions:
            return
        np.savez_compressed(
            self.frames_dir / f"chunk_{self._chunk:05d}.npz",
            positions=np.stack(self._positions),
            shades=np.stack(self._shades),
            times=np.asarray(self._times, dtype=np.float64),
        )
        self._chunk += 1
        self._positions.clear()
        self._shades.clear()
        self._times.clear()
        self._write_meta()

    def close(self) -> None:
        self.flush()
        self._write_meta()

    def _write_meta(self) -> None:
        (self.out / META).write_text(
            json.dumps(
                {"n_frames": self.n_frames, "stride": self.stride, "index": self.index}
            ),
            encoding="utf-8",
        )


def read_meta(out_dir: str | Path) -> dict:
    path = Path(out_dir) / META
    if not path.exists():
        return {"n_frames": 0, "stride": 1, "index": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_frame(out_dir: str | Path, index: int) -> tuple[np.ndarray, np.ndarray, float] | None:
    meta = read_meta(out_dir)
    entries = meta.get("index", [])
    if not 0 <= index < len(entries):
        return None
    chunk, offset = entries[index]
    path = Path(out_dir) / "frames" / f"chunk_{chunk:05d}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return (
            np.asarray(data["positions"][offset]),
            np.asarray(data["shades"][offset]),
            float(data["times"][offset]),
        )
