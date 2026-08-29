"""Ewolucja cech pod wpływem otoczenia i czasu własnego."""

from __future__ import annotations

import numpy as np

from bone.economy import exchange_wealth, redistribute_on_death
from bone.mapping import effective_learning_mass, lorentz_gamma
from bone.neighbors import NeighborSet, build_neighbors
from bone.traits import Universe


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def evolve_traits(
    universe: Universe,
    interaction_radius: float | None = None,
    neighbors: NeighborSet | None = None,
) -> None:
    """
    Aktualizacja cech po kroku ruchu.
    - mądrość rośnie z dτ = dt/γ
    - wiedza przez lokalną wymianę (zdolność × szybkość nauki)
    - miłość/nienawiść przez podobieństwo/odmienność sąsiadów
    - złość z ciśnienia kinetycznego
    - zdrowie spada przy wysokiej γ i nienawiści
    - przepływ dóbr (handel / wyzysk / dary) — patrz economy.py
    """
    cfg = universe.config
    dt = cfg.dt
    alive = universe.alive_mask
    idx = np.flatnonzero(alive)
    n_alive = idx.size
    if n_alive == 0:
        return

    radius = interaction_radius if interaction_radius is not None else cfg.spacing * 1.75
    if neighbors is None:
        neighbors = build_neighbors(universe, r_max=radius)

    # gospodarka: produkcja, konsumpcja, prądy wymiany
    exchange_wealth(universe, interaction_radius=radius, neighbors=neighbors)

    traits = universe.traits
    pos = universe.positions
    vel = universe.velocities
    gamma = lorentz_gamma(vel, cfg.c)

    # czas własny → mądrość
    traits["wisdom"][alive] += 0.15 * (dt / gamma[alive])
    traits["wisdom"] = _clip01(traits["wisdom"])

    knowledge_delta = np.zeros(universe.n, dtype=np.float64)
    love_delta = np.zeros(universe.n, dtype=np.float64)
    hate_delta = np.zeros(universe.n, dtype=np.float64)
    loyalty_delta = np.zeros(universe.n, dtype=np.float64)
    anger_delta = np.zeros(universe.n, dtype=np.float64)
    honesty_pull = np.zeros(universe.n, dtype=np.float64)
    degree = np.zeros(universe.n, dtype=np.float64)

    i, j = neighbors.pairs_within(pos, radius)
    if i.size:

        # podobieństwo cech + wyrównanie predyspozycji (plemiona)
        vi = np.column_stack(
            [
                traits["honesty"][i],
                traits["loyalty"][i],
                traits["love"][i],
                traits["hatred"][i],
            ]
        )
        vj = np.column_stack(
            [
                traits["honesty"][j],
                traits["loyalty"][j],
                traits["love"][j],
                traits["hatred"][j],
            ]
        )
        trait_sim = np.exp(-2.5 * np.linalg.norm(vi - vj, axis=1))
        align = np.sum(traits["predisposition"][i] * traits["predisposition"][j], axis=1)
        similarity = 0.55 * trait_sim + 0.45 * (0.5 * (align + 1.0))

        # wymiana wiedzy (symetryczny przepływ w stronę średniej pary)
        k_i = traits["knowledge"][i]
        k_j = traits["knowledge"][j]
        m_eff = effective_learning_mass(traits)
        flow_ij = (
            0.04
            * dt
            * traits["ability"][i]
            * traits["learning_speed"][i]
            * (k_j - k_i)
            / m_eff[i]
        )
        flow_ji = (
            0.04
            * dt
            * traits["ability"][j]
            * traits["learning_speed"][j]
            * (k_i - k_j)
            / m_eff[j]
        )
        np.add.at(knowledge_delta, i, flow_ij)
        np.add.at(knowledge_delta, j, flow_ji)

        love_gain = 0.012 * dt * (similarity - 0.45)
        hate_gain = 0.014 * dt * (0.55 - similarity)
        np.add.at(love_delta, i, love_gain)
        np.add.at(love_delta, j, love_gain)
        np.add.at(hate_delta, i, hate_gain)
        np.add.at(hate_delta, j, hate_gain)

        loy = 0.01 * dt * similarity
        np.add.at(loyalty_delta, i, loy)
        np.add.at(loyalty_delta, j, loy)

        speeds = np.linalg.norm(vel, axis=1)
        rel = np.abs(speeds[i] - speeds[j]) / (cfg.c + 1e-9)
        ang = 0.015 * dt * rel * 4.0
        np.add.at(anger_delta, i, ang)
        np.add.at(anger_delta, j, ang)

        # uczciwość → średnia pary
        np.add.at(honesty_pull, i, traits["honesty"][j])
        np.add.at(honesty_pull, j, traits["honesty"][i])
        np.add.at(degree, i, 1.0)
        np.add.at(degree, j, 1.0)

    # źródło wiedzy: doświadczenie z czasu własnego i interakcji (niezerowa suma)
    experience = (
        0.0055 * dt * traits["learning_speed"] * traits["ability"] / gamma
        + 0.002 * dt * np.minimum(degree, 12.0) * traits["learning_speed"]
    )
    traits["knowledge"] = _clip01(
        traits["knowledge"] + knowledge_delta + experience * traits["alive"]
    )
    traits["love"] = _clip01(traits["love"] + love_delta)
    traits["hatred"] = _clip01(traits["hatred"] + hate_delta)
    traits["loyalty"] = _clip01(traits["loyalty"] + loyalty_delta)
    traits["anger"] = _clip01(traits["anger"] + anger_delta - 0.01 * dt)

    has_n = degree > 0
    if np.any(has_n):
        mean_h = honesty_pull[has_n] / degree[has_n]
        traits["honesty"][has_n] += 0.01 * dt * (mean_h - traits["honesty"][has_n])
        traits["honesty"] = _clip01(traits["honesty"])

    # bieda pogarsza zdrowie; dostatek lekko chroni
    wealth_rel = traits["wealth"] / (cfg.wealth_mean + 1e-9)
    poverty = np.clip(1.0 - wealth_rel, 0.0, 1.5)
    wear = (
        0.02 * (gamma - 1.0)
        + 0.01 * traits["anger"]
        + 0.012 * traits["hatred"]
        + 0.008 * poverty
    )
    heal = (
        0.005 * traits["love"]
        + 0.003 * traits["loyalty"]
        + 0.002 * traits["honesty"]
        + 0.003 * np.clip(wealth_rel - 0.5, 0.0, 2.0)
    )
    traits["health"] = _clip01(traits["health"] + dt * (heal - wear) * traits["alive"])

    traits["endurance"] = _clip01(
        traits["endurance"] + 0.002 * dt * traits["wisdom"] * traits["alive"]
    )
    # zdolności rosną z wiedzą i dostępem do dóbr (kapitał edukacyjny)
    traits["ability"] = _clip01(
        traits["ability"]
        + 0.003 * dt * traits["knowledge"] * traits["alive"]
        + 0.001 * dt * np.clip(np.log1p(traits["wealth"]), 0.0, 2.0) * traits["alive"]
    )

    dead = traits["alive"] & (traits["health"] <= cfg.death_health)
    if np.any(dead):
        dead_idx = np.flatnonzero(dead)
        redistribute_on_death(universe, dead_idx)
        traits["alive"][dead] = False
        universe.velocities[dead] = 0.0

    pred = traits["predisposition"]
    norms = np.linalg.norm(pred, axis=1, keepdims=True)
    traits["predisposition"] = pred / np.maximum(norms, 1e-12)
