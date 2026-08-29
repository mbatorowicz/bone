"""Mapowanie cech osobowych na wielkości fizyczne."""

from __future__ import annotations

from typing import Any

import numpy as np

from bone.traits import Universe


def bare_rest_mass(traits: np.ndarray, endurance_scale: float = 1.35) -> np.ndarray:
    """Wytrwałość → masa spoczynkowa m₀ (bez bogactwa / energii)."""
    return 0.15 + endurance_scale * traits["endurance"]


def rest_mass(
    traits: np.ndarray,
    wealth_coupling: float = 0.35,
    energy_coupling: float = 0.4,
    endurance_scale: float = 1.35,
) -> np.ndarray:
    """
    Masa grawitacyjna / bezwładna — składniki:
    - m₀ z wytrwałości (endurance)
    - wkład energii: energy_coupling × health × m₀  (proxy E/c²)
    - wkład kapitału: wealth_coupling × log(1+wealth)
    """
    m0 = bare_rest_mass(traits, endurance_scale=endurance_scale)
    energy_term = energy_coupling * traits["health"] * m0
    capital = wealth_coupling * np.log1p(np.maximum(traits["wealth"], 0.0))
    return m0 + energy_term + capital


def gravitational_mass(traits: np.ndarray, cfg: Any) -> np.ndarray:
    """Masa do sił N-body z parametrów SimConfig."""
    return rest_mass(
        traits,
        wealth_coupling=getattr(cfg, "wealth_mass_coupling", 0.35),
        energy_coupling=getattr(cfg, "energy_mass_coupling", 0.4),
    )


def rest_energy(traits: np.ndarray, c: float) -> np.ndarray:
    """Zdrowie × m₀ c² — energia spoczynkowa (nie miesza bogactwa)."""
    m0 = bare_rest_mass(traits)
    return traits["health"] * m0 * (c**2)


def lorentz_gamma(velocities: np.ndarray, c: float) -> np.ndarray:
    """γ = 1 / sqrt(1 - v²/c²)."""
    v2 = np.sum(velocities**2, axis=1)
    beta2 = np.clip(v2 / (c * c), 0.0, 0.999999)
    return 1.0 / np.sqrt(1.0 - beta2)


def affinity_matrix_pair(
    traits_i: np.ndarray,
    traits_j: np.ndarray,
) -> float:
    """Skalar s_ij: miłość+lojalność+uczciwość przyciągają; nienawiść odpycha."""
    love = 0.5 * (traits_i["love"] + traits_j["love"])
    loyalty = 0.5 * (traits_i["loyalty"] + traits_j["loyalty"])
    hatred = 0.5 * (traits_i["hatred"] + traits_j["hatred"])
    honesty = 0.5 * (traits_i["honesty"] + traits_j["honesty"])
    s = (love + loyalty) * (0.5 + 0.5 * honesty) - 1.4 * hatred
    return float(np.clip(s, -2.0, 2.0))


def pairwise_affinity(traits: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """s_ij: miłość/lojalność/uczciwość + zgodność predyspozycji; nienawiść odpycha."""
    love = 0.5 * (traits["love"][i] + traits["love"][j])
    loyalty = 0.5 * (traits["loyalty"][i] + traits["loyalty"][j])
    hatred = 0.5 * (traits["hatred"][i] + traits["hatred"][j])
    honesty = 0.5 * (traits["honesty"][i] + traits["honesty"][j])
    align = np.sum(traits["predisposition"][i] * traits["predisposition"][j], axis=1)
    s = (love + loyalty) * (0.5 + 0.5 * honesty) * (0.55 + 0.45 * align)
    s -= hatred * (1.1 + 0.9 * np.maximum(0.0, -align))
    return np.clip(s, -2.0, 2.0)


def effective_learning_mass(traits: np.ndarray) -> np.ndarray:
    """Szybkość nauki → 1/m_eff (niska masa efektywna = szybsze przyswajanie)."""
    return 0.2 + 1.0 / (0.15 + traits["learning_speed"])


def knowledge_momentum(traits: np.ndarray) -> np.ndarray:
    """Wiedza jako |p| w przestrzeni cech."""
    return traits["knowledge"]


def summarize_physics(universe: Universe) -> dict[str, float]:
    """Agregaty fizyczne do logów."""
    alive = universe.alive_mask
    if not np.any(alive):
        return {
            "mean_mass": 0.0,
            "mean_m0": 0.0,
            "mean_energy": 0.0,
            "mean_gamma": 1.0,
            "mean_speed": 0.0,
            "kinetic_proxy": 0.0,
        }
    tr = universe.traits[alive]
    cfg = universe.config
    m0 = bare_rest_mass(tr)
    m = gravitational_mass(tr, cfg)
    e0 = rest_energy(tr, cfg.c)
    gamma = lorentz_gamma(universe.velocities[alive], cfg.c)
    speeds = np.linalg.norm(universe.velocities[alive], axis=1)
    return {
        "mean_mass": float(m.mean()),
        "mean_m0": float(m0.mean()),
        "mean_energy": float(e0.mean()),
        "mean_gamma": float(gamma.mean()),
        "mean_speed": float(speeds.mean()),
        "kinetic_proxy": float(np.mean(0.5 * m * speeds**2)),
    }
