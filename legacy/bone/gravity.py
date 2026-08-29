"""Siły grawitacyjne N-body — masa z cech/bogactwa; CUDA gdy dostępna."""

from __future__ import annotations

import numpy as np

from bone.gpu import as_tensor, gpu_enabled, to_numpy
from bone.mapping import gravitational_mass, pairwise_affinity
from bone.neighbors import NeighborSet, build_neighbors
from bone.singularity import (
    effective_core_repulsion,
    effective_G,
    effective_soft_eps,
    effective_wall_stiffness,
)
from bone.traits import Universe

# rezydentne bufory CUDA (reuse między wywołaniami)
_GPU_CACHE: dict | None = None


def compute_forces(
    universe: Universe,
    neighbors: NeighborSet | None = None,
) -> np.ndarray:
    """
    F_i = Σ_j -G m_i m_j s_ij r_ij / (r² + ε)^{3/2}
    + jądro + ściany.
    m = wytrwałość (m₀) + energia/zdrowie + bogactwo — geometria emergentna.
    """
    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=universe.config.r_cut)
    if gpu_enabled():
        return _compute_forces_cuda(universe, neighbors)
    return _compute_forces_cpu(universe, neighbors)


def _pair_ij(
    neighbors: NeighborSet, positions: np.ndarray, r_cut: float
) -> tuple[np.ndarray, np.ndarray]:
    return neighbors.pairs_within(positions, r_cut)


def _compute_forces_cpu(universe: Universe, neighbors: NeighborSet) -> np.ndarray:
    cfg = universe.config
    pos = universe.positions
    vel = universe.velocities
    traits = universe.traits
    alive = universe.alive_mask
    n = universe.n
    forces = np.zeros((n, 3), dtype=np.float64)

    alive_idx = neighbors.alive_idx
    if alive_idx.size < 2:
        return forces

    g_eff = effective_G(universe)
    eps = effective_soft_eps(universe)
    core_rep = effective_core_repulsion(universe)
    wall_k = effective_wall_stiffness(universe)

    m = gravitational_mass(traits, cfg)
    i, j = _pair_ij(neighbors, pos, cfg.r_cut)

    if i.size:
        rij = pos[i] - pos[j]
        r2 = np.sum(rij * rij, axis=1)
        r = np.sqrt(r2 + 1e-18)
        s = pairwise_affinity(traits, i, j)

        denom = (r2 + eps) ** 1.5
        strength = g_eff * m[i] * m[j] * s / denom
        strength = np.clip(strength, -8.0, 8.0)

        fij = (-strength)[:, None] * rij

        core = cfg.spacing * 0.65
        near = r < core
        if np.any(near) and core_rep > 1e-9:
            push = core_rep * ((core - r[near]) / core) ** 2
            fij[near] += push[:, None] * (rij[near] / (r[near, None] + 1e-12))

        np.add.at(forces, i, fij)
        np.add.at(forces, j, -fij)

    speeds = np.linalg.norm(vel[alive], axis=1, keepdims=True)
    forces[alive] -= cfg.damping * vel[alive]
    forces[alive] -= cfg.drag_quad * speeds * vel[alive]

    pred = traits["predisposition"][alive]
    endurance = traits["endurance"][alive]
    forces[alive] += cfg.predisposition_force * (pred.T * (0.2 + endurance)).T

    if wall_k > 1e-9:
        half = 0.5 * (cfg.grid_n - 1) * cfg.spacing + cfg.wall_margin
        p = pos[alive]
        for axis in range(3):
            over = p[:, axis] - half
            hi = over > 0
            if np.any(hi):
                forces[alive_idx[hi], axis] -= wall_k * over[hi]
            under = -half - p[:, axis]
            lo = under > 0
            if np.any(lo):
                forces[alive_idx[lo], axis] += wall_k * under[lo]

    return forces


def _ensure_gpu_cache(n: int, torch):
    global _GPU_CACHE
    if _GPU_CACHE is not None and _GPU_CACHE.get("n") == n:
        return _GPU_CACHE
    _GPU_CACHE = {
        "n": n,
        "pos": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
        "vel": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
        "m": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "love": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "loyalty": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "hatred": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "honesty": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "endurance": torch.empty((n,), dtype=torch.float32, device="cuda"),
        "pred": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
        "forces": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
        "pair_gen": -1,
        "pair_i": None,
        "pair_j": None,
        "alive_idx": None,
        "alive_gen": -1,
    }
    return _GPU_CACHE


def _upload(t, arr: np.ndarray) -> None:
    t.copy_(as_tensor(arr, dtype=t.dtype))


def _gpu_pair_indices(cache: dict, neighbors: NeighborSet, torch):
    """Reuse globalnych indeksów par na GPU między rebuildami Verlet."""
    gen = int(neighbors.generation)
    if (
        cache.get("pair_gen") == gen
        and cache.get("pair_i") is not None
        and cache.get("alive_gen") == gen
    ):
        return cache["pair_i"], cache["pair_j"], cache["alive_idx"]

    alive_idx_np = neighbors.alive_idx
    alive_idx = torch.as_tensor(alive_idx_np, dtype=torch.long, device="cuda")
    cache["alive_idx"] = alive_idx
    cache["alive_gen"] = gen

    if neighbors.pairs.size == 0:
        empty = torch.empty(0, dtype=torch.long, device="cuda")
        cache["pair_i"] = empty
        cache["pair_j"] = empty
        cache["pair_gen"] = gen
        return empty, empty, alive_idx

    # lokalne → globalne raz na rebuild (nie przy każdym filtrze r_cut)
    li = neighbors.pairs[:, 0]
    lj = neighbors.pairs[:, 1]
    i = torch.as_tensor(alive_idx_np[li], dtype=torch.long, device="cuda")
    j = torch.as_tensor(alive_idx_np[lj], dtype=torch.long, device="cuda")
    cache["pair_i"] = i
    cache["pair_j"] = j
    cache["pair_gen"] = gen
    return i, j, alive_idx


def _compute_forces_cuda(universe: Universe, neighbors: NeighborSet) -> np.ndarray:
    """Sąsiedzi z NeighborSet; indeksy par rezydentne między rebuildami."""
    import torch

    cfg = universe.config
    traits = universe.traits
    n = universe.n
    cache = _ensure_gpu_cache(n, torch)
    forces = cache["forces"]
    forces.zero_()

    if neighbors.alive_idx.size < 2:
        return to_numpy(forces)

    g_eff = float(effective_G(universe))
    eps = float(effective_soft_eps(universe))
    core_rep = float(effective_core_repulsion(universe))
    wall_k = float(effective_wall_stiffness(universe))

    m_np = gravitational_mass(traits, cfg).astype(np.float32)
    _upload(cache["pos"], universe.positions)
    _upload(cache["vel"], universe.velocities)
    _upload(cache["m"], m_np)
    _upload(cache["love"], traits["love"])
    _upload(cache["loyalty"], traits["loyalty"])
    _upload(cache["hatred"], traits["hatred"])
    _upload(cache["honesty"], traits["honesty"])
    _upload(cache["endurance"], traits["endurance"])
    _upload(cache["pred"], traits["predisposition"])

    pos = cache["pos"]
    vel = cache["vel"]
    m = cache["m"]
    pred = cache["pred"]
    i_all, j_all, alive_idx = _gpu_pair_indices(cache, neighbors, torch)

    if i_all.numel():
        r_cut2 = float(cfg.r_cut) ** 2
        rij = pos[i_all] - pos[j_all]
        r2 = (rij * rij).sum(dim=1)
        active = r2 <= r_cut2
        i = i_all[active]
        j = j_all[active]
        rij = rij[active]
        r2 = r2[active]

        love = 0.5 * (cache["love"][i] + cache["love"][j])
        loyalty = 0.5 * (cache["loyalty"][i] + cache["loyalty"][j])
        hatred = 0.5 * (cache["hatred"][i] + cache["hatred"][j])
        honesty = 0.5 * (cache["honesty"][i] + cache["honesty"][j])
        align = (pred[i] * pred[j]).sum(dim=1)
        s = (love + loyalty) * (0.5 + 0.5 * honesty) * (0.55 + 0.45 * align)
        s = s - hatred * (1.1 + 0.9 * torch.clamp(-align, min=0.0))
        s = torch.clamp(s, -2.0, 2.0)

        r = torch.sqrt(r2 + 1e-18)
        denom = (r2 + eps) ** 1.5
        strength = torch.clamp(g_eff * m[i] * m[j] * s / denom, -8.0, 8.0)
        fij = (-strength).unsqueeze(1) * rij

        core = cfg.spacing * 0.65
        if core_rep > 1e-9:
            near = r < core
            rn = torch.where(near, r, torch.ones_like(r))
            push = core_rep * ((core - rn) / core) ** 2
            push = torch.where(near, push, torch.zeros_like(push))
            fij = fij + push.unsqueeze(1) * (rij / (rn.unsqueeze(1) + 1e-12))

        forces.index_add_(0, i, fij)
        forces.index_add_(0, j, -fij)

    v_a = vel[alive_idx]
    speeds = torch.linalg.norm(v_a, dim=1, keepdim=True)
    f_a = forces[alive_idx]
    f_a = f_a - cfg.damping * v_a
    f_a = f_a - cfg.drag_quad * speeds * v_a
    f_a = f_a + cfg.predisposition_force * pred[alive_idx] * (
        0.2 + cache["endurance"][alive_idx]
    ).unsqueeze(1)

    if wall_k > 1e-9:
        half = 0.5 * (cfg.grid_n - 1) * cfg.spacing + cfg.wall_margin
        p = pos[alive_idx]
        for axis in range(3):
            over = p[:, axis] - half
            f_a[:, axis] = torch.where(over > 0, f_a[:, axis] - wall_k * over, f_a[:, axis])
            under = -half - p[:, axis]
            f_a[:, axis] = torch.where(under > 0, f_a[:, axis] + wall_k * under, f_a[:, axis])

    forces[alive_idx] = f_a
    # float32 → host; wytnij Inf/NaN z akumulacji index_add
    out = to_numpy(forces).astype(np.float64, copy=False)
    return np.nan_to_num(out, nan=0.0, posinf=80.0, neginf=-80.0)


def potential_energy(universe: Universe, neighbors: NeighborSet | None = None) -> float:
    cfg = universe.config
    pos = universe.positions
    traits = universe.traits
    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=cfg.r_cut)
    if neighbors.alive_idx.size < 2:
        return 0.0

    g_eff = effective_G(universe)
    eps = effective_soft_eps(universe)
    m = gravitational_mass(traits, cfg)
    i, j = neighbors.pairs_within(pos, cfg.r_cut)
    if i.size == 0:
        return 0.0
    rij = pos[i] - pos[j]
    r = np.sqrt(np.sum(rij * rij, axis=1) + eps)
    s = pairwise_affinity(traits, i, j)
    u = -g_eff * m[i] * m[j] * s / r
    return float(u.sum())
