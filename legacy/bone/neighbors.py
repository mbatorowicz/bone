"""Wspólne sąsiedztwo — kNN z limitem (bez eksplozji P≈N² w klastrach)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from bone.traits import Universe


@dataclass
class NeighborSet:
    """Pary lokalne względem alive_idx, zbudowane dla r_build (ze skin)."""

    alive_idx: np.ndarray  # (Na,) indeksy globalne
    pairs: np.ndarray  # (P, 2) indeksy lokalne 0..Na-1
    r_max: float  # promień interakcji fizycznej (r_cut bez skin)
    r_build: float  # promień przy budowie (r_max + skin)
    generation: int = 0  # id rebuildu — do reuse GPU
    built_at_step: int = -1
    _pos_at_build: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_alive(self) -> int:
        return int(self.alive_idx.size)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0]) if self.pairs.size else 0

    def global_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        if self.pairs.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty
        i = self.alive_idx[self.pairs[:, 0]]
        j = self.alive_idx[self.pairs[:, 1]]
        return i, j

    def pairs_within(
        self, positions: np.ndarray, radius: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Globalne (i, j) z odległością ≤ radius."""
        i, j = self.global_pairs()
        if i.size == 0:
            return i, j
        if radius >= self.r_build - 1e-15:
            return i, j
        rij = positions[i] - positions[j]
        r2 = np.sum(rij * rij, axis=1)
        mask = r2 <= radius * radius
        return i[mask], j[mask]

    def pair_degree(self, n: int, radius: float | None, positions: np.ndarray) -> np.ndarray:
        """Stopień wierzchołka — do kolorów gęstości."""
        deg = np.zeros(n, dtype=np.float64)
        if radius is None or radius >= self.r_build - 1e-15:
            i, j = self.global_pairs()
        else:
            i, j = self.pairs_within(positions, radius)
        if i.size:
            np.add.at(deg, i, 1.0)
            np.add.at(deg, j, 1.0)
        return deg


_GEN = 0


def _knn_undirected_pairs(
    tree: cKDTree,
    n: int,
    r: float,
    k: int,
    max_pairs: int,
) -> np.ndarray:
    """
    Max K najbliższych w promieniu r na punkt; pary undirected (i<j).
    Unika pełnego query_pairs w gęstym klastrze.
    """
    kk = int(min(max(1, k), max(1, n - 1)))
    dist, idx = tree.query(tree.data, k=kk + 1, workers=-1)
    if kk == 1:
        dist = np.asarray(dist, dtype=np.float64)[:, None]
        idx = np.asarray(idx, dtype=np.int64)[:, None]
    else:
        dist = np.asarray(dist, dtype=np.float64)
        idx = np.asarray(idx, dtype=np.int64)

    i_all = np.repeat(np.arange(n, dtype=np.int64), dist.shape[1])
    j_all = idx.reshape(-1)
    d_all = dist.reshape(-1)
    mask = (
        (j_all > i_all)
        & (j_all >= 0)
        & (j_all < n)
        & np.isfinite(d_all)
        & (d_all <= r)
        & (j_all != i_all)
    )
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.column_stack([i_all[mask], j_all[mask]])
    # unikalne (query jest symetryczny tylko częściowo przy limicie K)
    pairs = np.unique(pairs, axis=0)
    if pairs.shape[0] > max_pairs:
        rng = np.random.default_rng(0)
        pairs = pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]
    return pairs.astype(np.int64, copy=False)


def build_neighbors(
    universe: Universe,
    r_max: float | None = None,
    *,
    skin: float = 0.0,
) -> NeighborSet:
    """
    KDTree + kNN cap (max_neighbors), opcjonalny skin do listy Verlet.
    """
    global _GEN
    cfg = universe.config
    r_phys = float(r_max if r_max is not None else cfg.r_cut)
    skin = float(skin)
    r_build = r_phys + skin
    k = int(max(4, min(256, getattr(cfg, "max_neighbors", 48))))
    max_pairs = int(max(10_000, getattr(cfg, "max_pairs", 1_500_000)))

    alive_idx = np.flatnonzero(universe.alive_mask)
    if alive_idx.size < 2:
        _GEN += 1
        return NeighborSet(
            alive_idx=alive_idx,
            pairs=np.zeros((0, 2), dtype=np.int64),
            r_max=r_phys,
            r_build=r_build,
            generation=_GEN,
            built_at_step=int(universe.step),
        )

    pos = universe.positions[alive_idx]
    # KDTree wymaga finite — napraw eksplozję numeryczną przed budową
    finite = np.isfinite(pos).all(axis=1)
    if not np.all(finite):
        bad_g = alive_idx[~finite]
        universe.positions[bad_g] = 0.0
        universe.velocities[bad_g] = 0.0
        pos = universe.positions[alive_idx]
        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
            universe.positions[alive_idx] = pos
    tree = cKDTree(pos)
    n = int(alive_idx.size)

    # małe N: pełne pary OK; duże / gęste: kNN
    if n * (n - 1) // 2 <= min(max_pairs, n * k):
        pairs = tree.query_pairs(r=r_build, output_type="ndarray")
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        else:
            pairs = np.asarray(pairs, dtype=np.int64)
            if pairs.shape[0] > max_pairs:
                rng = np.random.default_rng(0)
                pairs = pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]
    else:
        pairs = _knn_undirected_pairs(tree, n, r_build, k, max_pairs)

    _GEN += 1
    return NeighborSet(
        alive_idx=alive_idx,
        pairs=pairs,
        r_max=r_phys,
        r_build=r_build,
        generation=_GEN,
        built_at_step=int(universe.step),
        _pos_at_build=pos.copy(),
    )


def max_interaction_radius(universe: Universe) -> float:
    """Promień obejmujący grawitację + lokalne interakcje."""
    cfg = universe.config
    return float(
        max(
            cfg.r_cut,
            cfg.spacing * 2.2,
            cfg.spacing * 1.75,
            cfg.spacing * 1.6,
        )
    )


def neighbor_skin(universe: Universe) -> float:
    cfg = universe.config
    frac = float(getattr(cfg, "neighbor_skin", 0.35))
    return max(0.0, frac * cfg.spacing)


def should_rebuild_neighbors(universe: Universe, cached: NeighborSet | None) -> bool:
    """Czy budować KDTree od nowa (lista Verlet wygasła / brak cache / drift)."""
    if cached is None or cached.pairs is None:
        return True
    cfg = universe.config
    every = int(max(1, getattr(cfg, "neighbor_rebuild_every", 8)))
    if int(universe.step) - int(cached.built_at_step) >= every:
        return True
    alive = np.flatnonzero(universe.alive_mask)
    if alive.size != cached.alive_idx.size:
        return True
    if alive.size and not np.array_equal(alive, cached.alive_idx):
        return True
    # drift: max|Δx| > skin/2 względem pozycji przy budowie
    pos0 = cached._pos_at_build
    if pos0 is not None and alive.size:
        skin = neighbor_skin(universe)
        if skin > 0.0:
            cur = universe.positions[cached.alive_idx]
            if cur.shape == pos0.shape:
                if not np.isfinite(cur).all():
                    return True
                d2 = np.sum((cur - pos0) ** 2, axis=1)
                if d2.size:
                    dmax = float(np.sqrt(np.nanmax(d2)))
                    if np.isfinite(dmax) and dmax > 0.5 * skin:
                        return True
    return False


def get_or_build_neighbors(universe: Universe) -> NeighborSet:
    """Reuse listy Verlet albo przebuduj ze skin."""
    cached = getattr(universe, "neighbors", None)
    r_max = max_interaction_radius(universe)
    skin = neighbor_skin(universe)
    if should_rebuild_neighbors(universe, cached):
        neigh = build_neighbors(universe, r_max=r_max, skin=skin)
        universe.neighbors = neigh  # type: ignore[attr-defined]
        return neigh
    return cached  # type: ignore[return-value]
