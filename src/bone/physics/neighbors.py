"""kNN + lista Verlet — bez pełnego query_pairs przy dużym N."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from bone.domain.universe import Universe


@dataclass
class NeighborSet:
    alive_idx: np.ndarray
    pairs: np.ndarray
    r_max: float
    r_build: float
    generation: int = 0
    built_at_step: int = -1
    _pos_at_build: np.ndarray | None = field(default=None, repr=False)

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0]) if self.pairs.size else 0

    def global_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        if self.pairs.size == 0:
            z = np.zeros(0, dtype=np.int64)
            return z, z
        return self.alive_idx[self.pairs[:, 0]], self.alive_idx[self.pairs[:, 1]]

    def pairs_within(self, positions: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        i, j = self.global_pairs()
        if i.size == 0 or radius >= self.r_build - 1e-15:
            return i, j
        rij = positions[i] - positions[j]
        mask = np.sum(rij * rij, axis=1) <= radius * radius
        return i[mask], j[mask]


_GEN = 0


def _knn_pairs(tree: cKDTree, n: int, r: float, k: int, max_pairs: int) -> np.ndarray:
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
    mask = (j_all > i_all) & (j_all >= 0) & (j_all < n) & np.isfinite(d_all) & (d_all <= r)
    if not np.any(mask):
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.unique(np.column_stack([i_all[mask], j_all[mask]]), axis=0)
    if pairs.shape[0] > max_pairs:
        rng = np.random.default_rng(0)
        pairs = pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]
    return pairs.astype(np.int64, copy=False)


def max_interaction_radius(universe: Universe) -> float:
    p = universe.config.physics
    return float(max(p.r_cut, universe.config.spawn.spacing * 2.0))


def neighbor_skin(universe: Universe) -> float:
    p = universe.config.physics
    return max(0.0, p.neighbor_skin * universe.config.spawn.spacing)


def build_neighbors(universe: Universe, r_max: float | None = None, *, skin: float = 0.0) -> NeighborSet:
    global _GEN
    phys = universe.config.physics
    r_phys = float(r_max if r_max is not None else phys.r_cut)
    r_build = r_phys + float(skin)
    k = int(max(4, min(128, phys.max_neighbors)))
    max_pairs = int(max(10_000, phys.max_pairs))
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
    if not np.isfinite(pos).all():
        bad = alive_idx[~np.isfinite(pos).all(axis=1)]
        universe.positions[bad] = 0.0
        universe.velocities[bad] = 0.0
        pos = np.nan_to_num(universe.positions[alive_idx], nan=0.0)
        universe.positions[alive_idx] = pos
    tree = cKDTree(pos)
    n = int(alive_idx.size)
    if n <= 2000 and n * (n - 1) // 2 <= min(max_pairs, n * k):
        pairs = tree.query_pairs(r=r_build, output_type="ndarray")
        pairs = np.asarray(pairs, dtype=np.int64) if pairs.size else np.zeros((0, 2), dtype=np.int64)
        if pairs.shape[0] > max_pairs:
            rng = np.random.default_rng(0)
            pairs = pairs[rng.choice(pairs.shape[0], size=max_pairs, replace=False)]
    else:
        pairs = _knn_pairs(tree, n, r_build, k, max_pairs)
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


def should_rebuild(universe: Universe, cached: NeighborSet | None) -> bool:
    if cached is None:
        return True
    every = int(max(1, universe.config.physics.neighbor_rebuild_every))
    if int(universe.step) - int(cached.built_at_step) >= every:
        return True
    alive = np.flatnonzero(universe.alive_mask)
    if alive.size != cached.alive_idx.size or (
        alive.size and not np.array_equal(alive, cached.alive_idx)
    ):
        return True
    pos0 = cached._pos_at_build
    if pos0 is not None and alive.size:
        skin = neighbor_skin(universe)
        cur = universe.positions[cached.alive_idx]
        if cur.shape == pos0.shape:
            if not np.isfinite(cur).all():
                return True
            dmax = float(np.sqrt(np.nanmax(np.sum((cur - pos0) ** 2, axis=1))))
            if np.isfinite(dmax) and skin > 0 and dmax > 0.5 * skin:
                return True
    return False


def get_or_build_neighbors(universe: Universe) -> NeighborSet:
    cached = universe.neighbors if isinstance(universe.neighbors, NeighborSet) else None
    if should_rebuild(universe, cached):
        neigh = build_neighbors(
            universe, r_max=max_interaction_radius(universe), skin=neighbor_skin(universe)
        )
        universe.neighbors = neigh
        return neigh
    return cached  # type: ignore[return-value]
