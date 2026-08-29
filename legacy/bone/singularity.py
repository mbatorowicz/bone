"""
Emergencja koncentracji — bez sztucznej studni grawitacyjnej.

inequality_drive zaostrza przepływ zasobów w czasie; geometria i kolaps
wynikają z wealth→masa→N-body, nie z masy w COM.
"""

from __future__ import annotations

import numpy as np

from bone.mapping import gravitational_mass
from bone.traits import Universe


def inequality_progress(universe: Universe) -> float:
    """
    Postęp zaostrzania nierówności w [0, drive].
    0 przed inequality_delay; potem drive * (1 - exp(-(t-delay)/tau)).
    """
    cfg = universe.config
    drive = getattr(cfg, "inequality_drive", 0.0)
    if drive <= 0.0:
        return 0.0
    delay = max(0.0, getattr(cfg, "inequality_delay", 0.0))
    if universe.t < delay:
        return 0.0
    tau = max(getattr(cfg, "inequality_timescale", 40.0), 1e-6)
    return float(drive * (1.0 - np.exp(-(universe.t - delay) / tau)))


# kompatybilność nazw w metrykach / logach
singularity_progress = inequality_progress


def flow_amplification(universe: Universe) -> float:
    """Mnożnik exploit/matthew przy rosnącym inequality_progress."""
    p = inequality_progress(universe)
    return 1.0 + p


def gift_suppression(universe: Universe) -> float:
    """Przy zaostrzaniu nierówności dary słabną (0..1)."""
    p = inequality_progress(universe)
    drive = max(getattr(universe.config, "inequality_drive", 0.0), 1e-9)
    frac = min(p / drive, 1.0) if drive > 0 else 0.0
    return max(0.05, 1.0 - 0.85 * frac)


def effective_soft_eps(universe: Universe) -> float:
    return universe.config.soft_eps


def effective_G(universe: Universe) -> float:
    """Stała G z configu — bez sztucznego wzrostu od „osobliwości”."""
    return universe.config.G


def effective_circularize(universe: Universe) -> float:
    return universe.config.circularize_rate


def effective_core_repulsion(universe: Universe) -> float:
    return universe.config.core_repulsion


def effective_wall_stiffness(universe: Universe) -> float:
    return universe.config.wall_stiffness


def half_mass_radius(positions: np.ndarray) -> float:
    """Promień zawierający połowę cząstek (względem COM)."""
    if positions.shape[0] == 0:
        return 0.0
    com = positions.mean(axis=0)
    r = np.linalg.norm(positions - com, axis=1)
    return float(np.median(r))


def top_wealth_fraction(universe: Universe, top_frac: float = 0.05) -> float:
    """Ułamek bogactwa w najbogatszych top_frac jednostek."""
    alive = universe.alive_mask
    w = universe.traits["wealth"][alive]
    if w.size == 0:
        return 0.0
    total = float(w.sum())
    if total <= 1e-18:
        return 0.0
    k = max(1, int(np.ceil(top_frac * w.size)))
    top = np.partition(w, -k)[-k:]
    return float(top.sum() / total)


def top_mass_fraction(universe: Universe, top_frac: float = 0.05) -> float:
    """Ułamek masy spoczynkowej w najcięższych top_frac jednostkach."""
    alive = universe.alive_mask
    if not np.any(alive):
        return 0.0
    m = gravitational_mass(universe.traits, universe.config)[alive]
    total = float(m.sum())
    if total <= 1e-18:
        return 0.0
    k = max(1, int(np.ceil(top_frac * m.size)))
    top = np.partition(m, -k)[-k:]
    return float(top.sum() / total)


def concentration_reached(universe: Universe, collapse_ratio: float) -> bool:
    """
    Emergencja „osobliwości”: skupienie geometryczne LUB dominacja bogactwa/masy.
    """
    cfg = universe.config
    r_stop = getattr(cfg, "collapse_stop_ratio", 0.05)
    w_stop = getattr(cfg, "wealth_concentration_stop", 0.45)
    if r_stop > 0 and collapse_ratio < r_stop:
        return True
    if w_stop > 0 and top_wealth_fraction(universe, 0.05) >= w_stop:
        return True
    if w_stop > 0 and top_mass_fraction(universe, 0.05) >= w_stop:
        return True
    return False
