"""Stabilizacja orbit: cyrkularyzacja, limit v_rel, anizotropowa dyssypacja (dysk)."""

from __future__ import annotations

import numpy as np

from bone.angular import angular_momentum
from bone.mapping import gravitational_mass, pairwise_affinity
from bone.neighbors import NeighborSet, build_neighbors, max_interaction_radius
from bone.singularity import effective_G, effective_circularize
from bone.traits import Universe


def stabilize_orbits(
    universe: Universe,
    neighbors: NeighborSet | None = None,
) -> None:
    """
    Dla par przyciągających (s_ij > 0):
    1) wygaszaj składową radialną prędkości względnej (cyrkularyzacja),
    2) przytnij |v_rel| do ~v_circ,
    3) lepkość pionowa: gaś v∥L.
    """
    cfg = universe.config
    circ = effective_circularize(universe)
    flatten = float(getattr(cfg, "disk_flatten_rate", 0.0))
    if circ <= 0.0 and cfg.vrel_cap_factor <= 0.0 and flatten <= 0.0:
        return

    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=max_interaction_radius(universe))

    alive_idx = neighbors.alive_idx
    if alive_idx.size < 2:
        return

    pos = universe.positions
    vel = universe.velocities
    traits = universe.traits
    m = gravitational_mass(traits, cfg)

    radius = min(cfg.r_cut, cfg.spacing * 2.2)
    local_w = np.zeros(universe.n, dtype=np.float64)

    i_all, j_all = neighbors.pairs_within(pos, radius)
    if i_all.size and (circ > 0.0 or cfg.vrel_cap_factor > 0.0 or flatten > 0.0):
        s_all = pairwise_affinity(traits, i_all, j_all)
        attract = s_all > 0.05
        if np.any(attract):
            i = i_all[attract]
            j = j_all[attract]
            s = s_all[attract]

            rij = pos[i] - pos[j]
            r2 = np.sum(rij * rij, axis=1)
            r = np.sqrt(r2 + 1e-18)
            rhat = rij / r[:, None]

            w = np.clip(s, 0.0, 2.0) * np.clip(1.2 - r / (cfg.spacing * 2.0), 0.15, 1.0)
            np.add.at(local_w, i, w)
            np.add.at(local_w, j, w)

            if circ > 0.0:
                vij = vel[i] - vel[j]
                v_rad_scalar = np.sum(vij * rhat, axis=1)
                v_rad = v_rad_scalar[:, None] * rhat
                alpha = np.clip(circ * cfg.dt * w, 0.0, 0.45)
                mi, mj = m[i], m[j]
                inv = 1.0 / (mi + mj + 1e-12)
                vel[i] -= (alpha * mj * inv)[:, None] * v_rad
                vel[j] += (alpha * mi * inv)[:, None] * v_rad

            if cfg.vrel_cap_factor > 0.0:
                mij = m[i]
                mjj = m[j]
                vij = vel[i] - vel[j]
                vrel = np.linalg.norm(vij, axis=1)
                g_eff = effective_G(universe)
                v_circ = np.sqrt(
                    np.maximum(
                        g_eff * (mij + mjj) * np.maximum(s, 0.1) / (r + cfg.soft_eps),
                        0.0,
                    )
                )
                vmax = cfg.vrel_cap_factor * (v_circ + 0.05)
                too_fast = vrel > vmax
                if np.any(too_fast):
                    scale = vmax[too_fast] / (vrel[too_fast] + 1e-12)
                    vi = vel[i[too_fast]]
                    vj = vel[j[too_fast]]
                    vcom = 0.5 * (vi + vj)
                    vrel_vec = (vi - vj) * scale[:, None]
                    vel[i[too_fast]] = vcom + 0.5 * vrel_vec
                    vel[j[too_fast]] = vcom - 0.5 * vrel_vec

    if flatten > 0.0:
        _L, L_hat, _Lmag = angular_momentum(universe)
        dens = local_w[alive_idx]
        dens = dens / (float(dens.mean()) + 1e-9)
        dens = np.clip(dens, 0.0, 3.0)
        coup = 0.15 + 0.85 * dens
        beta = np.clip(flatten * cfg.dt * coup, 0.0, 0.55)
        v_alive = vel[alive_idx]
        v_par = np.sum(v_alive * L_hat[None, :], axis=1, keepdims=True) * L_hat[None, :]
        vel[alive_idx] = v_alive - beta[:, None] * v_par

    universe.velocities = vel
