"""Widok sześcianu 3D: animacja ruchu punktów jak materia we wszechświecie."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def save_trajectory(
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    out_dir: Path,
) -> Path:
    """Zapisz trajektorię do NPZ (pozycje + kolor na klatkę)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "trajectory.npz"
    np.savez_compressed(
        path,
        positions=np.stack(frames, axis=0),  # (F, N, 3)
        times=np.asarray(times, dtype=np.float64),
        colors=np.stack(colors, axis=0),  # (F, N) — np. love-hatred lub gęstość
    )
    return path


def _cube_edges(half: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Krawędzie sześcianu do narysowania."""
    corners = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * half
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return [(corners[a], corners[b]) for a, b in edges]


def save_cube_frames_png(
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    out_dir: Path,
    half: float,
    which: tuple[int, ...] = (0, -1),
) -> list[Path]:
    """Statyczne klatki 3D (start / koniec)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in which:
        i = idx if idx >= 0 else len(frames) + idx
        if i < 0 or i >= len(frames):
            continue
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        pos = frames[i]
        c = colors[i]
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=c, cmap="coolwarm", s=2, alpha=0.7, linewidths=0)
        for a, b in _cube_edges(half):
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#888", lw=0.6, alpha=0.5)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_zlim(-half, half)
        ax.set_title(f"Szescian punktow — t={times[i]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        tag = "start" if i == 0 else ("final" if i == len(frames) - 1 else f"f{i}")
        path = out_dir / f"cube_3d_{tag}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def save_cube_gif(
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    out_dir: Path,
    half: float,
    fps: int = 12,
    stride: int = 1,
    point_stride: int = 2,
) -> Path:
    """Animowany GIF sześcianu — widać ruch i zbieranie w grupy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = list(range(0, len(frames), max(1, stride)))
    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    pos0 = frames[idx[0]][::point_stride]
    c0 = colors[idx[0]][::point_stride]
    sc = ax.scatter(pos0[:, 0], pos0[:, 1], pos0[:, 2], c=c0, cmap="coolwarm", s=3, alpha=0.75, linewidths=0)
    for a, b in _cube_edges(half):
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#666", lw=0.7, alpha=0.45)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(-half, half)
    title = ax.set_title(f"t={times[idx[0]]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def update(frame_i: int):
        fi = idx[frame_i]
        p = frames[fi][::point_stride]
        sc._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
        sc.set_array(colors[fi][::point_stride])
        title.set_text(f"Szescian — ruch pod wplywem grawitacji  t={times[fi]:.2f}")
        return sc, title

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 // fps, blit=False)
    path = out_dir / "cube_animation.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def save_cube_html(
    frames: list[np.ndarray],
    times: list[float],
    colors: list[np.ndarray],
    out_dir: Path,
    half: float,
    point_stride: int = 2,
    max_frames: int = 400,
) -> Path:
    """Interaktywny HTML ze wszystkimi suwakami parametrów + widok 3D."""
    from bone.studio_page import build_studio_html
    from bone.ui_schema import default_param_values

    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = len(frames)
    step = max(1, n_frames // max_frames)
    sel = list(range(0, n_frames, step))
    if sel[-1] != n_frames - 1:
        sel.append(n_frames - 1)

    packed = []
    for i in sel:
        p = frames[i][::point_stride]
        c = colors[i][::point_stride]
        cmin, cmax = float(c.min()), float(c.max())
        cn = (c - cmin) / (cmax - cmin + 1e-12)
        packed.append(
            {
                "t": float(times[i]),
                "x": p[:, 0].round(3).tolist(),
                "y": p[:, 1].round(3).tolist(),
                "z": p[:, 2].round(3).tolist(),
                "c": cn.round(3).tolist(),
            }
        )

    html = build_studio_html(
        trajectory={"half": half, "frames": packed},
        params=default_param_values(),
        studio_mode=False,
    )
    path = out_dir / "cube_view.html"
    path.write_text(html, encoding="utf-8")
    return path


def local_density_colors(
    positions: np.ndarray,
    radius: float = 1.6,
    *,
    degree: np.ndarray | None = None,
) -> np.ndarray:
    """Kolor ≈ lokalna gęstość — preferuj degree z NeighborSet (bez osobnego KDTree)."""
    if degree is not None:
        return np.asarray(degree, dtype=np.float64)
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    counts = tree.query_ball_point(positions, r=radius, return_length=True)
    return np.asarray(counts, dtype=np.float64)


def pack_live_frame(
    positions: np.ndarray,
    velocities: np.ndarray,
    t: float,
    step: int,
    *,
    point_stride: int = 4,
    colors: np.ndarray | None = None,
) -> dict:
    """Jedna klatka „teraz” do podglądu na żywo (nie trajektoria)."""
    stride = max(1, int(point_stride))
    p = np.ascontiguousarray(positions[::stride], dtype=np.float32)
    if colors is not None:
        c_src = np.asarray(colors[::stride], dtype=np.float32)
        cmin, cmax = float(c_src.min()) if c_src.size else 0.0, float(c_src.max()) if c_src.size else 1.0
        cn = ((c_src - cmin) / (cmax - cmin + 1e-12)).astype(np.float32)
    else:
        v = np.ascontiguousarray(velocities[::stride], dtype=np.float32)
        speed = np.linalg.norm(v, axis=1)
        cmin, cmax = float(speed.min()) if speed.size else 0.0, float(speed.max()) if speed.size else 1.0
        cn = ((speed - cmin) / (cmax - cmin + 1e-12)).astype(np.float32)
    half = float(np.max(np.abs(p)) * 1.05 + 1.0) if p.size else 12.0
    return {
        "live": True,
        "t": float(t),
        "step": int(step),
        "half": half,
        "n": int(p.shape[0]),
        "x": p[:, 0].tolist(),
        "y": p[:, 1].tolist(),
        "z": p[:, 2].tolist(),
        "c": cn.tolist(),
    }


def save_live_frame(
    positions: np.ndarray,
    velocities: np.ndarray,
    t: float,
    step: int,
    out_dir: Path,
    *,
    point_stride: int = 1,
) -> Path:
    """Zapis bieżącego stanu do out/live.npz (podgląd zewnętrzny)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "live.npz"
    np.savez(
        path,
        positions=positions.astype(np.float32),
        velocities=velocities.astype(np.float32),
        t=np.array([t], dtype=np.float64),
        step=np.array([step], dtype=np.int64),
        point_stride=np.array([point_stride], dtype=np.int64),
    )
    return path


def load_live_frame(out_dir: Path, point_stride: int = 2) -> dict | None:
    path = Path(out_dir) / "live.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return pack_live_frame(
        data["positions"],
        data["velocities"],
        float(data["t"][0]),
        int(data["step"][0]),
        point_stride=point_stride,
    )
