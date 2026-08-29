"""Metryki społeczne i first-hitting-times."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from bone.angular import angular_momentum, flattening_ratio
from bone.constants import SimConfig
from bone.economy import FlowStats, gini_coefficient
from bone.mapping import pairwise_affinity, summarize_physics
from bone.neighbors import _knn_undirected_pairs
from bone.singularity import (
    concentration_reached,
    half_mass_radius,
    inequality_progress,
    top_mass_fraction,
    top_wealth_fraction,
)
from bone.traits import Universe


def _cluster_edge_pairs(universe: Universe, radius: float) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Lokalne pary (indeksy w alive) do FoF / hate — bez pełnego query_pairs przy dużym N.
    Zwraca (pairs_local (P,2), alive_idx, n).
    """
    alive_idx = np.flatnonzero(universe.alive_mask)
    n = int(alive_idx.size)
    if n < 2:
        return np.zeros((0, 2), dtype=np.int64), alive_idx, n

    # reuse NeighborSet jeśli dostępny
    neigh = getattr(universe, "neighbors", None)
    if (
        neigh is not None
        and getattr(neigh, "alive_idx", None) is not None
        and neigh.alive_idx.size == n
        and np.array_equal(neigh.alive_idx, alive_idx)
        and neigh.pairs.size
    ):
        i_g, j_g = neigh.pairs_within(universe.positions, radius)
        if i_g.size:
            # global → lokalny 0..n-1
            inv = np.full(universe.n, -1, dtype=np.int64)
            inv[alive_idx] = np.arange(n, dtype=np.int64)
            pairs = np.column_stack([inv[i_g], inv[j_g]])
            return pairs, alive_idx, n

    pos = universe.positions[alive_idx]
    tree = cKDTree(pos)
    k = int(min(32, max(4, getattr(universe.config, "max_neighbors", 24))))
    if n <= 2000 and n * (n - 1) // 2 <= 500_000:
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        else:
            pairs = np.asarray(pairs, dtype=np.int64)
    else:
        pairs = _knn_undirected_pairs(tree, n, radius, k, max_pairs=n * k)
    return pairs, alive_idx, n


@dataclass
class HitEvents:
    """Czasy pierwszego przekroczenia progów (w jednostkach czasu lab.)."""

    knowledge: float | None = None
    cluster: float | None = None
    conflict: float | None = None
    death: float | None = None
    inequality: float | None = None
    singularity: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "T_knowledge": self.knowledge,
            "T_cluster": self.cluster,
            "T_conflict": self.conflict,
            "T_death": self.death,
            "T_inequality": self.inequality,
            "T_singularity": self.singularity,
        }


@dataclass
class MetricsTracker:
    cfg: SimConfig
    events: HitEvents = field(default_factory=HitEvents)
    history: list[dict[str, float]] = field(default_factory=list)
    _r_half0: float | None = None

    def friends_of_friends(self, universe: Universe) -> tuple[int, int]:
        """Zwraca (liczba_klastrów≥min, rozmiar_największego)."""
        pairs, alive_idx, n = _cluster_edge_pairs(
            universe, float(self.cfg.cluster_link_radius)
        )
        if n == 0:
            return 0, 0
        if pairs.size == 0:
            return 0, 1 if n else 0

        gi = alive_idx[pairs[:, 0]]
        gj = alive_idx[pairs[:, 1]]
        s = pairwise_affinity(universe.traits, gi, gj)
        keep = s >= self.cfg.cluster_affinity_min
        edges = pairs[keep]
        if edges.size == 0:
            return 0, 1

        row = np.concatenate([edges[:, 0], edges[:, 1]])
        col = np.concatenate([edges[:, 1], edges[:, 0]])
        data = np.ones(row.shape[0], dtype=np.uint8)
        graph = coo_matrix((data, (row, col)), shape=(n, n))
        _n_comp, labels = connected_components(graph, directed=False)
        _, counts = np.unique(labels, return_counts=True)
        max_allowed = max(
            self.cfg.cluster_min_size,
            int(self.cfg.cluster_max_frac * n),
        )
        community = counts[
            (counts >= self.cfg.cluster_min_size) & (counts <= max_allowed)
        ]
        largest_community = int(community.max()) if community.size else 0
        return int(community.size), largest_community

    def polarization(self, universe: Universe) -> float:
        """
        Korelacja: czy większa nienawiść idzie z większą odległością
        od lokalnej średniej „miłości”. Dodatnia = polaryzacja.
        """
        alive = universe.alive_mask
        if alive.sum() < 10:
            return 0.0
        hate = universe.traits["hatred"][alive]
        love = universe.traits["love"][alive]
        # prosty wskaźnik: std(hate) * (1 - korelacja love-hate lokalna)
        if hate.std() < 1e-9 or love.std() < 1e-9:
            return float(hate.std())
        corr = np.corrcoef(love, hate)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        return float(hate.std() * (1.0 - corr) / 2.0)

    def local_hate_density(self, universe: Universe) -> float:
        """Max średniej nienawiści po sąsiadach z listy par (bez query_ball_point)."""
        r = float(self.cfg.cluster_link_radius) * 1.2
        pairs, alive_idx, n = _cluster_edge_pairs(universe, r)
        if n == 0 or pairs.size == 0:
            return 0.0
        hate = universe.traits["hatred"][alive_idx]
        sum_h = np.zeros(n, dtype=np.float64)
        deg = np.zeros(n, dtype=np.float64)
        i = pairs[:, 0]
        j = pairs[:, 1]
        np.add.at(sum_h, i, hate[j])
        np.add.at(sum_h, j, hate[i])
        np.add.at(deg, i, 1.0)
        np.add.at(deg, j, 1.0)
        ok = deg >= 3.0
        if not np.any(ok):
            return 0.0
        return float((sum_h[ok] / deg[ok]).max())

    def observe(self, universe: Universe) -> dict[str, float]:
        alive = universe.alive_mask
        n_alive = int(alive.sum())
        n_clusters, largest = self.friends_of_friends(universe)
        pol = self.polarization(universe)
        hate_d = self.local_hate_density(universe)
        phys = summarize_physics(universe)

        if n_alive:
            wealth = universe.traits["wealth"][alive]
            mean_w = float(wealth.mean())
            max_w = float(wealth.max())
            min_w = float(wealth.min())
            total_w = float(wealth.sum())
            gini = gini_coefficient(wealth)
        else:
            mean_w = max_w = min_w = total_w = gini = 0.0

        flow = universe.last_flow if isinstance(universe.last_flow, FlowStats) else FlowStats()
        velocity = float(flow.total_abs_flow / (total_w + 1e-9)) if total_w > 0 else 0.0

        r_half = half_mass_radius(universe.positions[alive]) if n_alive else 0.0
        if self._r_half0 is None and r_half > 0:
            self._r_half0 = r_half
        ineq_p = inequality_progress(universe)
        collapse_ratio = float(r_half / (self._r_half0 + 1e-9)) if self._r_half0 else 1.0
        top_w = top_wealth_fraction(universe, 0.05) if n_alive else 0.0
        top_m = top_mass_fraction(universe, 0.05) if n_alive else 0.0
        _L, _Lh, Lmag = angular_momentum(universe) if n_alive else (None, None, 0.0)
        flat = flattening_ratio(universe) if n_alive else 1.0

        row = {
            "step": float(universe.step),
            "t": universe.t,
            "n_alive": float(n_alive),
            "mean_knowledge": float(universe.traits["knowledge"][alive].mean()) if n_alive else 0.0,
            "max_knowledge": float(universe.traits["knowledge"][alive].max()) if n_alive else 0.0,
            "mean_wisdom": float(universe.traits["wisdom"][alive].mean()) if n_alive else 0.0,
            "mean_love": float(universe.traits["love"][alive].mean()) if n_alive else 0.0,
            "mean_hatred": float(universe.traits["hatred"][alive].mean()) if n_alive else 0.0,
            "mean_loyalty": float(universe.traits["loyalty"][alive].mean()) if n_alive else 0.0,
            "mean_health": float(universe.traits["health"][alive].mean()) if n_alive else 0.0,
            "mean_anger": float(universe.traits["anger"][alive].mean()) if n_alive else 0.0,
            "mean_wealth": mean_w,
            "max_wealth": max_w,
            "min_wealth": min_w,
            "total_wealth": total_w,
            "gini": gini,
            "flow_traded": flow.traded,
            "flow_exploited": flow.exploited,
            "flow_gifted": flow.gifted,
            "money_velocity": velocity,
            "n_clusters": float(n_clusters),
            "largest_cluster": float(largest),
            "polarization": pol,
            "hate_density": hate_d,
            "r_half": r_half,
            "collapse_ratio": collapse_ratio,
            "inequality_progress": ineq_p,
            "singularity_progress": ineq_p,  # alias logów
            "top_wealth_frac": top_w,
            "top_mass_frac": top_m,
            "L_mag": float(Lmag),
            "flattening": float(flat),
            **phys,
        }
        self.history.append(row)
        self._update_events(universe, row)
        return row

    def _update_events(self, universe: Universe, row: dict[str, float]) -> None:
        t = universe.t
        if self.events.knowledge is None and row["max_knowledge"] >= self.cfg.knowledge_threshold:
            self.events.knowledge = t
        if (
            self.events.cluster is None
            and row["n_clusters"] >= 1
            and row["largest_cluster"] >= self.cfg.cluster_min_size
        ):
            self.events.cluster = t
        if self.events.conflict is None and row["hate_density"] >= self.cfg.hate_conflict_threshold:
            self.events.conflict = t
        if self.events.death is None and row["n_alive"] < universe.n:
            self.events.death = t
        if self.events.inequality is None and row["gini"] >= self.cfg.gini_threshold:
            self.events.inequality = t
        if self.events.singularity is None and concentration_reached(
            universe, row.get("collapse_ratio", 1.0)
        ):
            self.events.singularity = t
