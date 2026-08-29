"""Stan jednostek: pozycje, prędkości i cechy osobowe."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bone.constants import SimConfig
from bone.geometry import generate_positions, resolve_geometry


TRAIT_DTYPE = np.dtype(
    [
        ("predisposition", "f8", (3,)),
        ("endurance", "f8"),
        ("health", "f8"),
        ("knowledge", "f8"),
        ("wisdom", "f8"),
        ("learning_speed", "f8"),
        ("honesty", "f8"),
        ("loyalty", "f8"),
        ("love", "f8"),
        ("anger", "f8"),
        ("hatred", "f8"),
        ("ability", "f8"),
        ("wealth", "f8"),
        ("alive", "bool"),
    ]
)


@dataclass
class Universe:
    """Stan całego układu N jednostek."""

    positions: np.ndarray  # (N, 3)
    velocities: np.ndarray  # (N, 3)
    traits: np.ndarray  # structured (N,)
    config: SimConfig
    t: float = 0.0
    step: int = 0
    last_flow: object | None = None

    @property
    def n(self) -> int:
        return self.positions.shape[0]

    @property
    def alive_mask(self) -> np.ndarray:
        return self.traits["alive"]


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def spawn_newborns(cfg: SimConfig, rng: np.random.Generator) -> Universe:
    """Pozycje wg geometrii + cechy noworodkowe. Start w bezruchu (v=0)."""
    geom = resolve_geometry(cfg.geometry)
    positions = generate_positions(
        geom,
        grid_n=cfg.grid_n,
        spacing=cfg.spacing,
        n_particles=cfg.n_particles,
        rng=rng,
    )
    n_particles = positions.shape[0]
    # zimny start — ruch wywołają wyłącznie siły z cech / grawitacji
    velocities = np.zeros_like(positions)
    traits = np.zeros(n_particles, dtype=TRAIT_DTYPE)

    mean = cfg.newborn_mean
    sigma = mean * cfg.newborn_sigma_frac

    def noise(shape=()) -> np.ndarray:
        return rng.normal(mean, sigma, size=shape)

    pred = rng.normal(0.0, 1.0, size=(n_particles, 3))
    norms = np.linalg.norm(pred, axis=1, keepdims=True)
    traits["predisposition"] = pred / np.maximum(norms, 1e-12)

    traits["endurance"] = _clip01(noise(n_particles))
    traits["health"] = _clip01(noise(n_particles) + 0.05)
    traits["knowledge"] = _clip01(noise(n_particles) * 0.4)
    traits["wisdom"] = 0.0
    traits["learning_speed"] = _clip01(noise(n_particles))
    traits["honesty"] = _clip01(noise(n_particles))
    traits["loyalty"] = _clip01(noise(n_particles))
    traits["love"] = _clip01(noise(n_particles))
    emo_sigma = sigma * 2.5
    traits["anger"] = _clip01(rng.normal(mean * 0.45, emo_sigma, size=n_particles))
    traits["hatred"] = _clip01(rng.normal(mean * 0.4, emo_sigma, size=n_particles))
    traits["ability"] = _clip01(rng.normal(mean, sigma * 2.2, size=n_particles))
    traits["wealth"] = np.maximum(
        rng.normal(
            cfg.wealth_mean,
            cfg.wealth_mean * cfg.newborn_sigma_frac * 1.8,
            size=n_particles,
        ),
        0.05 * cfg.wealth_mean,
    )
    traits["alive"] = True

    n_cat = max(1, n_particles // 70)
    cat = rng.choice(n_particles, size=n_cat, replace=False)
    traits["hatred"][cat] = _clip01(traits["hatred"][cat] + 0.25)
    traits["anger"][cat] = _clip01(traits["anger"][cat] + 0.15)
    traits["wealth"][cat] *= 1.35
    flip = rng.normal(0.0, 1.0, size=(n_cat, 3))
    flip /= np.maximum(np.linalg.norm(flip, axis=1, keepdims=True), 1e-12)
    traits["predisposition"][cat] = flip

    # opcjonalny kick: jedna globalna oś (średnia predyspozycja) → net L0, nie spin per-cząstka
    if cfg.orbital_seed_speed > 0.0:
        from bone.angular import mean_predisposition_axis

        axis = mean_predisposition_axis(traits)
        tangential = np.cross(np.broadcast_to(axis, positions.shape), positions)
        t_norm = np.linalg.norm(tangential, axis=1, keepdims=True)
        tangential /= np.maximum(t_norm, 1e-12)
        speed = cfg.orbital_seed_speed * (
            cfg.newborn_mean + cfg.newborn_sigma_frac * traits["endurance"]
        )
        velocities = tangential * speed[:, None]

    return Universe(
        positions=positions,
        velocities=velocities,
        traits=traits,
        config=cfg,
    )
