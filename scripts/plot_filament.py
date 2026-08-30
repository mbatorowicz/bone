#!/usr/bin/env python
"""Rzut układu i jego profil gęstości podłużnej w kilku chwilach.

Rzut pokazuje, czy zgęstki są; profil pokazuje ILE ich jest i czy są rozłożone
regularnie. Regularność jest tu istotna: niestabilność Jeansa ma wyróżnioną
długość fali, więc równe odstępy między zgęstkami są dowodem, że to wzrost modu,
a nie przypadkowe skupiska szumu.

Współrzędna podłużna zależy od kształtu: dla włókna to x, dla pierścienia kąt
azymutalny. Profil po niewłaściwej współrzędnej rozmywa zgęstki i pokazuje
gładką krzywą tam, gdzie struktura jest — dlatego to jawny przełącznik.

    python scripts/plot_filament.py runs/frag_ring --along angle
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bone.io import trajectory


def _pick_frames(total: int, count: int) -> list[int]:
    if total <= count:
        return list(range(total))
    return list(np.linspace(0, total - 1, count).astype(int))


def main() -> int:
    ap = argparse.ArgumentParser(description="Rzut i profil gęstości podłużnej")
    ap.add_argument("run")
    ap.add_argument("--out", default="docs/filament.png")
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument(
        "--along", choices=("x", "angle"), default="x",
        help="współrzędna podłużna: x dla włókna, angle dla pierścienia",
    )
    args = ap.parse_args()

    meta = trajectory.read_meta(args.run)
    total = int(meta.get("n_frames", 0))
    if total == 0:
        print(f"brak klatek w {args.run}")
        return 1

    chosen = _pick_frames(total, args.panels)
    frames = [trajectory.load_frame(args.run, i) for i in chosen]
    frames = [f for f in frames if f is not None]

    # wspólne granice osi dla wszystkich paneli — inaczej rosnące zagęszczenie
    # wyglądałoby jak zmiana skali, a nie zmiana układu
    all_x = np.concatenate([f[0][:, 0] for f in frames])
    all_y = np.concatenate([f[0][:, 1] for f in frames])
    xlim = float(np.percentile(np.abs(all_x), 99.5))
    ylim = float(np.percentile(np.abs(all_y), 99.5))

    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 5.6), squeeze=False)
    fig.patch.set_facecolor("#0b0d12")

    if args.along == "angle":
        edges = np.linspace(-np.pi, np.pi, args.bins + 1)
        along_label = "kąt azymutalny θ  [rad]"
    else:
        edges = np.linspace(-xlim, xlim, args.bins + 1)
        along_label = "x"
    centers = 0.5 * (edges[:-1] + edges[1:])

    def longitudinal(positions: np.ndarray) -> np.ndarray:
        if args.along == "angle":
            return np.arctan2(positions[:, 1], positions[:, 0])
        return positions[:, 0]

    for column, (positions, _shade, when) in enumerate(frames):
        ax = axes[0][column]
        ax.set_facecolor("#0b0d12")
        ax.scatter(
            positions[:, 0], positions[:, 1],
            s=0.12, c="#8fd3ff", alpha=0.35, linewidths=0, rasterized=True,
        )
        ax.set_xlim(-xlim, xlim)
        ax.set_ylim(-ylim, ylim)
        ax.set_title(f"t = {when:.2f}", color="#e6edf6", fontsize=11)
        ax.tick_params(colors="#7c8798", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2a3140")
        if column == 0:
            ax.set_ylabel("y", color="#e6edf6")

        profile, _ = np.histogram(longitudinal(positions), bins=edges)
        bx = axes[1][column]
        bx.set_facecolor("#0b0d12")
        bx.fill_between(centers, profile, color="#8fd3ff", alpha=0.85, linewidth=0)
        bx.set_xlim(edges[0], edges[-1])
        bx.set_xlabel(along_label, color="#e6edf6")
        bx.tick_params(colors="#7c8798", labelsize=8)
        for spine in bx.spines.values():
            spine.set_color("#2a3140")
        if column == 0:
            bx.set_ylabel("cząstek na przedział", color="#e6edf6", fontsize=9)

    # wspólna skala pionowa profili, żeby dało się porównać kontrast
    top = max(bx.get_ylim()[1] for bx in axes[1])
    for bx in axes[1]:
        bx.set_ylim(0, top)

    what = "pierścień" if args.along == "angle" else "włókno"
    fig.suptitle(
        f"Fragmentacja: {what} — rzut na płaszczyznę xy (góra) i gęstość podłużna (dół)",
        color="#e6edf6", fontsize=13,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"zapisano {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
