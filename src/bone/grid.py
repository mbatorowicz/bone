"""Pudło siatki i wagi cloud-in-cell.

Wydzielone z solvera siatkowego, bo nie są jego własnością: dokładnie tego
samego dopasowania pudła i tych samych wag potrzebuje chłodzenie, które musi
znać lokalny przepływ masowy. Duplikat CIC w dwóch miejscach byłby gwarancją,
że któryś z nich kiedyś przestanie być symetryczny — a symetria deposit ↔ gather
jest tym, co utrzymuje zachowanie pędu.

Margines ``edge`` jest parametrem, nie stałą modułu, bo zależy od tego, co się
z siatką robi. Solver liczy gradient szablonem czwartego rzędu sięgającym dwóch
komórek, więc potrzebuje trzech pustych komórek przy ścianie; chłodzenie tylko
uśrednia w komórce i nie potrzebuje żadnego.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    origin: np.ndarray  # (3,) lewy dolny róg siatki
    h: float  # rozmiar oczka (izotropowy)
    ng: int  # bok siatki
    edge: int = 0  # ile komórek przy ścianie musi zostać puste

    def contains(self, positions: np.ndarray) -> bool:
        local = (positions - self.origin) / self.h
        return bool(
            local.min() >= self.edge - 0.5
            and local.max() <= self.ng - self.edge - 0.5
        )


def fit_box(positions: np.ndarray, ng: int, margin: float, edge: int = 0) -> Box:
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float((hi - lo).max())
    if not np.isfinite(span) or span <= 0.0:
        span = 1.0
    usable = ng - 2 * edge
    h = span * (1.0 + max(margin, 0.0)) / max(usable, 1)
    origin = center - 0.5 * ng * h
    return Box(origin=np.asarray(origin, dtype=np.float64), h=float(h), ng=int(ng), edge=int(edge))


def cic_weights(positions: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
    """Wagi cloud-in-cell: dla każdej cząstki 8 węzłów i 8 wag sumujących się do 1.

    Te same wagi służą do rozłożenia wielkości na siatkę i do odczytu z niej.
    """
    ng = box.ng
    local = (positions - box.origin) / box.h - 0.5
    base = np.floor(local)
    frac = local - base
    base = base.astype(np.int64)
    np.clip(base, 0, ng - 2, out=base)

    n = positions.shape[0]
    idx = np.empty((n, 8), dtype=np.int64)
    wgt = np.empty((n, 8), dtype=np.float64)
    corner = 0
    for dx in (0, 1):
        wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
        ix = base[:, 0] + dx
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            iy = base[:, 1] + dy
            for dz in (0, 1):
                wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                iz = base[:, 2] + dz
                idx[:, corner] = (ix * ng + iy) * ng + iz
                wgt[:, corner] = wx * wy * wz
                corner += 1
    return idx, wgt
