"""Wielkości zachowane i miary jakości symulacji.

To jest ta część, której poprzedniej wersji brakowało najbardziej. Bez pomiaru
energii i pędu nie wiadomo, czy symulacja liczy fizykę, czy generuje ładny szum.
Wszystkie wielkości są liczone z tego samego potencjału, który wygenerował siły,
więc dryf energii mówi o jakości CAŁKOWANIA, a nie o niespójności modelu.

Backend przybliżony dodatkowo raportuje własny błąd: dla losowej próbki cząstek
liczymy siłę dokładnie i porównujemy. Dzięki temu przybliżenie jest widoczną
liczbą, a nie cichym założeniem.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bone import relativity as sr
from bone.backends.exact import exact_forces_for
from bone.state import State


def half_mass_radius(positions: np.ndarray, masses: np.ndarray) -> float:
    if positions.shape[0] == 0:
        return 0.0
    com = np.average(positions, axis=0, weights=masses)
    r = np.linalg.norm(positions - com, axis=1)
    order = np.argsort(r)
    cumulative = np.cumsum(masses[order])
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    k = int(np.searchsorted(cumulative, 0.5 * total))
    return float(r[order[min(k, order.size - 1)]])


def angular_momentum(state: State) -> np.ndarray:
    rel = state.positions - state.center_of_mass()
    return np.cross(rel, state.momenta).sum(axis=0)


def measure_force_error(
    state: State,
    forces: np.ndarray,
    G: float,
    softening: float,
    sample: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Błąd siły backendu na losowej próbce, względem dokładnego O(N²)."""
    n = state.n
    size = int(min(max(sample, 1), n))
    rows = rng.choice(n, size=size, replace=False)
    reference = exact_forces_for(state.positions, state.masses, G, softening, rows)
    got = forces[rows]
    scale = np.linalg.norm(reference, axis=1)
    typical = float(np.sqrt(np.mean(scale**2)))
    if typical <= 0.0:
        return {"force_err_rms": 0.0, "force_err_max": 0.0}
    delta = np.linalg.norm(got - reference, axis=1)
    return {
        "force_err_rms": float(np.sqrt(np.mean(delta**2)) / typical),
        "force_err_max": float(delta.max() / typical),
    }


@dataclass
class Snapshot:
    values: dict[str, float]

    def __getitem__(self, key: str) -> float:
        return self.values[key]

    def get(self, key: str, default: float = 0.0) -> float:
        return self.values.get(key, default)


@dataclass
class Diagnostics:
    """Liczy metryki i śledzi ich dryf względem stanu początkowego."""

    history: list[dict[str, float]] = field(default_factory=list)
    reference: dict[str, float] | None = None

    def observe(
        self,
        state: State,
        potential: np.ndarray,
        c: float,
        dt: float,
        extra: dict[str, float] | None = None,
        energy_removed: float = 0.0,
    ) -> dict[str, float]:
        """``energy_removed`` to skumulowana energia odprowadzona przez dyssypację.

        Bez tego argumentu włączenie chłodzenia zamieniłoby ``E_drift`` — główny
        wskaźnik jakości całkowania w tym kodzie — w licznik tego, ile energii
        celowo wyrzuciliśmy. Wielkością zachowaną w modelu z dyssypacją jest
        E_tot + E_odprowadzona, i to jej dryf ma sens mierzyć.
        """
        m = state.masses
        kinetic = float(sr.kinetic_energy(m, state.momenta, c).sum())
        pot = 0.5 * float(np.dot(m, potential))
        total = kinetic + pot

        gamma = state.gamma(c)
        beta = state.speed_over_c(c)
        momentum_sum = state.momenta.sum(axis=0)
        momentum_scale = float(np.linalg.norm(state.momenta, axis=1).sum()) + 1e-300
        angular = angular_momentum(state)

        row = {
            "step": float(state.step),
            "t": float(state.time),
            "dt": float(dt),
            "n": float(state.n),
            "E_kin": kinetic,
            "E_pot": pot,
            "E_tot": total,
            "E_cooled": float(energy_removed),
            "virial": float(2.0 * kinetic / abs(pot)) if pot != 0.0 else 0.0,
            "P_residual": float(np.linalg.norm(momentum_sum) / momentum_scale),
            "L_mag": float(np.linalg.norm(angular)),
            "gamma_mean": float(gamma.mean()),
            "gamma_max": float(gamma.max()),
            "beta_mean": float(beta.mean()),
            "beta_max": float(beta.max()),
            "r_half": half_mass_radius(state.positions, m),
        }
        if extra:
            row.update(extra)

        if self.reference is None:
            self.reference = {
                "E_tot": total + energy_removed,
                "L_mag": row["L_mag"],
                "r_half": row["r_half"],
            }
        ref = self.reference
        row["E_drift"] = _relative(total + energy_removed, ref["E_tot"])
        row["L_drift"] = _relative(row["L_mag"], ref["L_mag"])
        row["r_half_ratio"] = row["r_half"] / (ref["r_half"] + 1e-300)

        self.history.append(row)
        return row

    def latest(self) -> dict[str, float] | None:
        return self.history[-1] if self.history else None

    def reset_reference(self) -> None:
        self.reference = None


def _relative(value: float, reference: float) -> float:
    denom = abs(reference)
    if denom < 1e-300:
        return 0.0
    return float((value - reference) / denom)
