"""Zapis i odczyt pełnego stanu.

Zapisujemy PĘD, nie prędkość. Prędkość zależy od ``c``, więc checkpoint zapisany
z prędkościami zmieniałby fizykę po wczytaniu z innym ``c`` — pęd jest niezależną
od tego zmienną stanu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bone.config import Config
from bone.state import State

CHECKPOINT = "checkpoint.npz"
CONFIG = "config.json"


def save(state: State, cfg: Config, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / CHECKPOINT
    np.savez_compressed(
        path,
        positions=state.positions,
        momenta=state.momenta,
        masses=state.masses,
        time=np.float64(state.time),
        step=np.int64(state.step),
    )
    (out / CONFIG).write_text(
        json.dumps(cfg.to_flat(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path


def load(out_dir: str | Path) -> tuple[State, Config]:
    out = Path(out_dir)
    data = np.load(out / CHECKPOINT, allow_pickle=False)
    cfg_path = out / CONFIG
    cfg = (
        Config.from_flat(json.loads(cfg_path.read_text(encoding="utf-8")))
        if cfg_path.exists()
        else Config()
    )
    state = State(
        positions=data["positions"],
        momenta=data["momenta"],
        masses=data["masses"],
        time=float(data["time"]),
        step=int(data["step"]),
    )
    return state, cfg


def exists(out_dir: str | Path) -> bool:
    return (Path(out_dir) / CHECKPOINT).exists()
