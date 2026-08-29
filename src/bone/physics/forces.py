"""Grawitacja Newtonowska na masach spoczynkowych — F do dp/dt (SR). Bez ścian."""

from __future__ import annotations

import numpy as np

from bone.domain.mapping import rest_mass
from bone.domain.universe import Universe
from bone.physics.gpu import gpu_enabled
from bone.physics.neighbors import NeighborSet, build_neighbors

_GPU_CACHE: dict | None = None


def compute_forces(universe: Universe, neighbors: NeighborSet | None = None) -> np.ndarray:
    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=universe.config.physics.r_cut)
    if gpu_enabled():
        return _forces_cuda(universe, neighbors)
    return _forces_cpu(universe, neighbors)


def _forces_cpu(universe: Universe, neighbors: NeighborSet) -> np.ndarray:
    phys = universe.config.physics
    pos = universe.positions
    vel = universe.velocities
    n = universe.n
    forces = np.zeros((n, 3), dtype=np.float64)
    alive_idx = neighbors.alive_idx
    if alive_idx.size < 2:
        return forces
    m = rest_mass(universe.mass)
    i, j = neighbors.pairs_within(pos, phys.r_cut)
    if i.size:
        rij = pos[i] - pos[j]
        r2 = np.sum(rij * rij, axis=1)
        r = np.sqrt(r2 + 1e-18)
        denom = (r2 + phys.soft_eps) ** 1.5
        # zawsze przyciąganie (s=1)
        strength = np.clip(phys.G * m[i] * m[j] / denom, 0.0, phys.force_cap)
        fij = (-strength)[:, None] * rij
        core = universe.config.spawn.spacing * 0.55
        near = r < core
        if np.any(near) and phys.core_repulsion > 1e-9:
            push = phys.core_repulsion * ((core - r[near]) / core) ** 2
            fij[near] += push[:, None] * (rij[near] / (r[near, None] + 1e-12))
        np.add.at(forces, i, fij)
        np.add.at(forces, j, -fij)
    alive = universe.alive_mask
    if phys.damping > 0:
        forces[alive] -= phys.damping * vel[alive]
    return np.nan_to_num(forces, nan=0.0, posinf=phys.force_cap, neginf=-phys.force_cap)


def _forces_cuda(universe: Universe, neighbors: NeighborSet) -> np.ndarray:
    import torch

    global _GPU_CACHE
    phys = universe.config.physics
    n = universe.n
    if _GPU_CACHE is None or _GPU_CACHE.get("n") != n:
        _GPU_CACHE = {
            "n": n,
            "pos": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
            "vel": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
            "m": torch.empty((n,), dtype=torch.float32, device="cuda"),
            "forces": torch.empty((n, 3), dtype=torch.float32, device="cuda"),
            "pair_gen": -1,
            "pair_i": None,
            "pair_j": None,
            "alive": None,
        }
    cache = _GPU_CACHE
    forces = cache["forces"]
    forces.zero_()
    if neighbors.alive_idx.size < 2:
        return forces.detach().cpu().numpy().astype(np.float64)

    def up(t, arr):
        t.copy_(torch.as_tensor(np.asarray(arr, dtype=np.float32), device="cuda"))

    up(cache["pos"], universe.positions)
    up(cache["vel"], universe.velocities)
    up(cache["m"], rest_mass(universe.mass))

    gen = int(neighbors.generation)
    if cache["pair_gen"] != gen or cache["pair_i"] is None:
        alive_np = neighbors.alive_idx
        cache["alive"] = torch.as_tensor(alive_np, dtype=torch.long, device="cuda")
        if neighbors.pairs.size == 0:
            empty = torch.empty(0, dtype=torch.long, device="cuda")
            cache["pair_i"] = cache["pair_j"] = empty
        else:
            li, lj = neighbors.pairs[:, 0], neighbors.pairs[:, 1]
            cache["pair_i"] = torch.as_tensor(alive_np[li], dtype=torch.long, device="cuda")
            cache["pair_j"] = torch.as_tensor(alive_np[lj], dtype=torch.long, device="cuda")
        cache["pair_gen"] = gen

    i_all, j_all = cache["pair_i"], cache["pair_j"]
    pos, m = cache["pos"], cache["m"]
    if i_all.numel():
        rij = pos[i_all] - pos[j_all]
        r2 = (rij * rij).sum(dim=1)
        active = r2 <= float(phys.r_cut) ** 2
        i, j = i_all[active], j_all[active]
        rij, r2 = rij[active], r2[active]
        denom = (r2 + float(phys.soft_eps)) ** 1.5
        strength = torch.clamp(
            float(phys.G) * m[i] * m[j] / denom, 0.0, float(phys.force_cap)
        )
        fij = (-strength).unsqueeze(1) * rij
        core = float(universe.config.spawn.spacing) * 0.55
        if phys.core_repulsion > 1e-9:
            r = torch.sqrt(r2 + 1e-18)
            near = r < core
            rn = torch.where(near, r, torch.ones_like(r))
            push = float(phys.core_repulsion) * ((core - rn) / core) ** 2
            push = torch.where(near, push, torch.zeros_like(push))
            fij = fij + push.unsqueeze(1) * (rij / (rn.unsqueeze(1) + 1e-12))
        forces.index_add_(0, i, fij)
        forces.index_add_(0, j, -fij)

    alive_idx = cache["alive"]
    if phys.damping > 0:
        forces[alive_idx] = forces[alive_idx] - float(phys.damping) * cache["vel"][alive_idx]
    out = forces.detach().cpu().numpy().astype(np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=phys.force_cap, neginf=-phys.force_cap)
