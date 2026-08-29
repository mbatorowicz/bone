"""
Przepływ dóbr / energii — źródło emergentnej geometrii.

- wealth → masa grawitacyjna (preferential attachment)
- wyzysk / Mateusz / dary regulowane greed_bias / generosity_bias
- inequality_drive zaostrza przejmowanie w czasie
- energy_exchange: zdrowie ↔ wealth, skalowane przez γ (relatywistyka)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bone.mapping import lorentz_gamma
from bone.neighbors import NeighborSet, build_neighbors
from bone.singularity import flow_amplification, gift_suppression
from bone.traits import Universe


@dataclass
class FlowStats:
    """Agregaty przepływu w jednym kroku."""

    traded: float = 0.0
    exploited: float = 0.0
    gifted: float = 0.0
    produced: float = 0.0
    consumed: float = 0.0
    energy_moved: float = 0.0
    total_abs_flow: float = 0.0

    @property
    def money_velocity(self) -> float:
        return self.total_abs_flow


def gini_coefficient(wealth: np.ndarray) -> float:
    """Współczynnik Giniego dla nieujemnego wektora bogactwa."""
    w = np.sort(np.asarray(wealth, dtype=np.float64))
    if w.size == 0:
        return 0.0
    total = w.sum()
    if total <= 1e-18:
        return 0.0
    n = w.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * w) - (n + 1) * total) / (n * total))


def _conductivity(traits: np.ndarray, i: np.ndarray, j: np.ndarray, similarity: np.ndarray) -> np.ndarray:
    honesty = 0.5 * (traits["honesty"][i] + traits["honesty"][j])
    loyalty = 0.5 * (traits["loyalty"][i] + traits["loyalty"][j])
    love = 0.5 * (traits["love"][i] + traits["love"][j])
    hatred = 0.5 * (traits["hatred"][i] + traits["hatred"][j])
    return (
        (0.15 + honesty)
        * (0.15 + loyalty)
        * (0.2 + love)
        * (1.0 - 0.85 * hatred)
        * (0.25 + 0.75 * similarity)
    )


def exchange_wealth(
    universe: Universe,
    interaction_radius: float | None = None,
    neighbors: NeighborSet | None = None,
) -> FlowStats:
    """Jeden krok przepływu dóbr / energii między sąsiadami."""
    cfg = universe.config
    dt = cfg.dt
    traits = universe.traits
    alive = universe.alive_mask
    idx = np.flatnonzero(alive)
    stats = FlowStats()

    if idx.size == 0:
        universe.last_flow = stats
        return stats

    amp = flow_amplification(universe)
    gift_scale = gift_suppression(universe) * getattr(cfg, "generosity_bias", 1.0)
    greed = getattr(cfg, "greed_bias", 1.0)
    gamma_all = lorentz_gamma(universe.velocities, cfg.c)
    gamma_alive = gamma_all[alive]

    # produkcja — zdolność; renta kapitału wzmacniana przy inequality
    skill = 0.2 + traits["ability"][alive]
    labor = (
        cfg.labor_rate
        * dt
        * (skill**2)
        * traits["health"][alive]
        * (0.4 + traits["knowledge"][alive])
        / np.maximum(gamma_alive, 1.0)  # właściwy czas: wolniejsi produkują w dτ
    )
    w_alive = traits["wealth"][alive]
    capital = (
        cfg.capital_return
        * dt
        * amp
        * w_alive
        * (0.25 + traits["ability"][alive])
        * np.sqrt(w_alive / (cfg.wealth_mean + 1e-9))
    )
    # konsumpcja rośnie z γ (relatywistyczny koszt życia)
    consume = (
        cfg.consume_rate
        * dt
        * gamma_alive
        * (0.5 + 0.4 * w_alive / (cfg.wealth_mean + 1e-9))
    )
    traits["wealth"][alive] += labor + capital - consume
    traits["wealth"][alive] = np.maximum(traits["wealth"][alive], 0.0)
    stats.produced = float((labor + capital).sum())
    stats.consumed = float(consume.sum())

    radius = interaction_radius if interaction_radius is not None else cfg.spacing * 1.75
    pos = universe.positions
    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=radius)
    i, j = neighbors.pairs_within(pos, radius)
    if i.size == 0:
        universe.last_flow = stats
        return stats

    align = np.sum(traits["predisposition"][i] * traits["predisposition"][j], axis=1)
    similarity = 0.5 * (align + 1.0)
    kappa = np.maximum(_conductivity(traits, i, j, similarity), 0.0)

    w_i = traits["wealth"][i]
    w_j = traits["wealth"][j]
    mu_i = w_i / (0.25 + traits["health"][i])
    mu_j = w_j / (0.25 + traits["health"][j])

    trade_part = cfg.trade_rate * dt * kappa * (mu_i - mu_j)

    attract_i = traits["ability"][i] + 0.7 * traits["knowledge"][i]
    attract_j = traits["ability"][j] + 0.7 * traits["knowledge"][j]
    pool = 0.5 * (w_i + w_j)
    matthew_part = (
        cfg.matthew_rate
        * amp
        * greed
        * dt
        * kappa
        * (attract_j - attract_i)
        * pool
        / (cfg.wealth_mean + 1e-9)
    )

    # przejmowanie: greed × (złość/nienawiść/zdolność)
    greed_i = greed * (
        0.35
        + traits["anger"][i]
        + traits["hatred"][i]
        + 0.5 * traits["ability"][i]
    )
    greed_j = greed * (
        0.35
        + traits["anger"][j]
        + traits["hatred"][j]
        + 0.5 * traits["ability"][j]
    )
    power_i = w_i * (0.3 + traits["ability"][i]) * greed_i
    power_j = w_j * (0.3 + traits["ability"][j]) * greed_j
    hate = 0.5 * (traits["hatred"][i] + traits["hatred"][j])
    honesty = 0.5 * (traits["honesty"][i] + traits["honesty"][j])
    exploit_part = (
        cfg.exploit_rate
        * amp
        * dt
        * hate
        * (1.0 - similarity)
        * (1.0 - 0.65 * honesty)
        * (power_i - power_j)
    )

    # rozdawnictwo: generosity × miłość × uczciwość
    love = 0.5 * (traits["love"][i] + traits["love"][j])
    gen_i = 0.3 + traits["love"][i] + traits["honesty"][i]
    gen_j = 0.3 + traits["love"][j] + traits["honesty"][j]
    gift_part = (
        cfg.gift_rate
        * gift_scale
        * dt
        * love
        * similarity
        * 0.5
        * (gen_i + gen_j)
        * (w_i - w_j)
    )

    flow = trade_part + matthew_part + exploit_part + gift_part
    max_from_i = 0.45 * w_i
    max_from_j = 0.45 * w_j
    flow = np.clip(flow, -max_from_j, max_from_i)

    delta = np.zeros(universe.n, dtype=np.float64)
    np.add.at(delta, i, -flow)
    np.add.at(delta, j, flow)
    traits["wealth"] += delta
    traits["wealth"] = np.maximum(traits["wealth"], 0.0)

    # energia (zdrowie) ↔ wealth, zależnie od γ
    e_rate = getattr(cfg, "energy_exchange", 0.0)
    if e_rate > 1e-12:
        g_i = gamma_all[i]
        g_j = gamma_all[j]
        # szybszy / „gorętszy” traci zdrowie na rzecz bogactwa sąsiada przy wyzysku
        e_flow = (
            e_rate
            * dt
            * kappa
            * (traits["health"][i] / g_i - traits["health"][j] / g_j)
            * (0.5 + 0.5 * np.abs(exploit_part) / (cfg.wealth_mean * dt + 1e-9))
        )
        e_flow = np.clip(e_flow, -0.08, 0.08)
        traits["health"][i] = np.clip(traits["health"][i] - e_flow, 0.0, 1.0)
        traits["health"][j] = np.clip(traits["health"][j] + e_flow, 0.0, 1.0)
        # część energii zamienia się w wealth u biorcy zdrowia
        traits["wealth"][j] += 0.15 * np.maximum(e_flow, 0.0) * cfg.wealth_mean
        traits["wealth"][i] += 0.15 * np.maximum(-e_flow, 0.0) * cfg.wealth_mean
        traits["wealth"] = np.maximum(traits["wealth"], 0.0)
        stats.energy_moved = float(np.abs(e_flow).sum())

    abs_flow = np.abs(flow)
    stats.traded = float(np.abs(trade_part).sum() + np.abs(matthew_part).sum())
    stats.exploited = float(np.abs(exploit_part).sum())
    stats.gifted = float(np.abs(gift_part).sum())
    stats.total_abs_flow = float(abs_flow.sum())
    universe.last_flow = stats
    return stats


def redistribute_on_death(universe: Universe, dead_idx: np.ndarray) -> None:
    """Rozdziel majątek zmarłych między żywych sąsiadów."""
    if dead_idx.size == 0:
        return
    traits = universe.traits
    alive = universe.alive_mask
    alive_idx = np.flatnonzero(alive)
    if alive_idx.size == 0:
        traits["wealth"][dead_idx] = 0.0
        return

    pos = universe.positions
    tree = cKDTree(pos[alive_idx])
    for d in dead_idx:
        estate = float(traits["wealth"][d])
        if estate <= 0:
            continue
        neigh_local = tree.query_ball_point(pos[d], r=universe.config.spacing * 2.0)
        if not neigh_local:
            share = estate / alive_idx.size
            traits["wealth"][alive_idx] += share
        else:
            heirs = alive_idx[np.asarray(neigh_local, dtype=np.int64)]
            traits["wealth"][heirs] += estate / heirs.size
        traits["wealth"][d] = 0.0
