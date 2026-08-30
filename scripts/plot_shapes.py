"""Katalog kształtów startowych — jeden rzut oka zamiast dziesięciu biegów.

    python scripts/plot_shapes.py                      # wszystkie kształty
    python scripts/plot_shapes.py --flatten 0.2        # te same, spłaszczone
    python scripts/plot_shapes.py --axis xy            # rzut z góry

Każdy kształt jest pokazany w tej samej skali, bo cały sens katalogu polega na
porównywaniu proporcji. Pod nazwą wypisany jest zmierzony stosunek wirialny przy
zadanej dyspersji — to ta liczba pokazuje, dlaczego temperatura nie jest
porównywalna między kształtami i dlaczego istnieje parametr `virial`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bone.config import GEOMETRIES, GEOMETRY_LABELS, Config  # noqa: E402
from bone.engine import Engine  # noqa: E402
from bone.spawn import sample_positions  # noqa: E402

_PLANES = {"xz": (0, 2, "x", "z"), "xy": (0, 1, "x", "y"), "yz": (1, 2, "y", "z")}


def _virial(geometry: str, thickness: float, flatten: float) -> float:
    """Wirial 2K/|U| przy dyspersji wyliczonej dla KULI o tej masie i promieniu.

    Celowo ta sama dyspersja dla wszystkich kształtów: rozrzut tych liczb jest
    treścią wykresu, a nie usterką.
    """
    cfg = Config().replace_flat({
        "geometry": geometry, "n_particles": 2000, "radius": 8.0, "total_mass": 4000.0,
        "thickness": thickness, "flatten": flatten,
        "rotation": 0.0, "temperature": 0.155, "G": 0.072, "c": 30.0,
        "softening": 0.15, "backend": "exact", "device": "cpu",
    })
    engine = Engine(cfg)
    try:
        return float(engine.collect_diagnostics()["virial"])
    finally:
        engine.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000, help="cząstek na panel")
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--thickness", type=float, default=0.15)
    ap.add_argument("--flatten", type=float, default=1.0)
    ap.add_argument("--axis", choices=sorted(_PLANES), default="xz", help="płaszczyzna rzutu")
    ap.add_argument("--no-virial", action="store_true", help="pomiń pomiar wiriału (szybciej)")
    ap.add_argument("--out", default="docs/shapes.png")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizontal, vertical, hlabel, vlabel = _PLANES[args.axis]
    columns = 5
    rows = -(-len(GEOMETRIES) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.15 * rows))
    fig.patch.set_facecolor("#0b0f16")

    # wspólny zakres osi: bez tego porównanie proporcji jest bezwartościowe
    limit = args.radius * 1.85 * max(1.0, args.flatten)

    for ax, geometry in zip(axes.ravel(), GEOMETRIES, strict=False):
        rng = np.random.default_rng(11)
        points = sample_positions(
            geometry, rng, args.n, args.radius, args.thickness, args.flatten
        )
        ax.scatter(
            points[:, horizontal], points[:, vertical],
            s=0.7, c="#7fd4ff", alpha=0.5, linewidths=0,
        )
        title = GEOMETRY_LABELS.get(geometry, geometry).split(" — ")[0]
        if not args.no_virial:
            ratio = _virial(geometry, args.thickness, args.flatten)
            title += f"\n2K/|U| = {ratio:.2f}"
        ax.set_title(title, color="#e8eef7", fontsize=9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_facecolor("#0b0f16")
        ax.tick_params(colors="#40506a", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#22304a")

    for ax in axes.ravel()[len(GEOMETRIES):]:
        ax.axis("off")

    axes.ravel()[0].set_xlabel(hlabel, color="#40506a")
    axes.ravel()[0].set_ylabel(vlabel, color="#40506a")
    fig.suptitle(
        f"Kształty startowe · rzut {args.axis} · grubość {args.thickness:g} · "
        f"spłaszczenie {args.flatten:g}",
        color="#e8eef7", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"zapisano {out}")


if __name__ == "__main__":
    main()
