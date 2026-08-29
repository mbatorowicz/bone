"""Zapis / wczytanie pełnego stanu układu (kontynuacja ewolucji)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bone.constants import SimConfig
from bone.traits import TRAIT_DTYPE, Universe


def save_checkpoint(
    universe: Universe,
    path: Path,
    *,
    compressed: bool = True,
) -> Path:
    """Zapisz pełny stan (pozycje, prędkości, wszystkie cechy, t, step)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "positions": universe.positions,
        "velocities": universe.velocities,
        "t": np.array([universe.t], dtype=np.float64),
        "step": np.array([universe.step], dtype=np.int64),
        "grid_n": np.array([universe.config.grid_n], dtype=np.int64),
        "dt": np.array([universe.config.dt], dtype=np.float64),
        "G": np.array([universe.config.G], dtype=np.float64),
    }
    for name in TRAIT_DTYPE.names:
        payload[f"trait_{name}"] = universe.traits[name]
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def load_checkpoint(path: Path, cfg: SimConfig | None = None) -> Universe:
    """Wczytaj stan i podepnij konfigurację (domyślnie z checkpointu + cfg)."""
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    n = data["positions"].shape[0]
    traits = np.zeros(n, dtype=TRAIT_DTYPE)
    for name in TRAIT_DTYPE.names:
        key = f"trait_{name}"
        if key in data:
            traits[name] = data[key]
        elif name in data:  # stary final_state.npz (częściowy)
            traits[name] = data[name]

    # uzupełnij brakujące pola sensownymi wartościami
    if "trait_predisposition" not in data and "predisposition" not in data:
        pred = np.random.default_rng(0).normal(size=(n, 3))
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        traits["predisposition"] = pred
    for name in TRAIT_DTYPE.names:
        if name == "alive":
            if not np.any(traits["alive"]) and name not in data and f"trait_{name}" not in data:
                traits["alive"] = True
            continue
        if name == "predisposition":
            continue
        # jeśli same zera i nie było w pliku — ustaw średnie noworodkowe
        key = f"trait_{name}"
        if key not in data and name not in data:
            if name == "wisdom":
                traits[name] = 0.0
            elif name == "wealth":
                traits[name] = 1.0
            else:
                traits[name] = 0.35

    # cfg z requestu ma pierwszeństwo (suwaki Studio / CLI) — nie nadpisuj grid_n z pliku
    base = cfg or SimConfig()

    return Universe(
        positions=np.array(data["positions"], dtype=np.float64, copy=True),
        velocities=np.array(data["velocities"], dtype=np.float64, copy=True),
        traits=traits,
        config=base,
        t=float(np.asarray(data["t"]).reshape(-1)[0]) if "t" in data else 0.0,
        step=int(np.asarray(data["step"]).reshape(-1)[0]) if "step" in data else 0,
    )


def load_trajectory_lists(path: Path) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:
    path = Path(path)
    if not path.exists():
        return [], [], []
    data = np.load(path)
    frames = [data["positions"][i].copy() for i in range(data["positions"].shape[0])]
    times = [float(data["times"][i]) for i in range(data["times"].shape[0])]
    colors = [data["colors"][i].copy() for i in range(data["colors"].shape[0])]
    return frames, times, colors
