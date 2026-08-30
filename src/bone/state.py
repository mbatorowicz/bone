"""Stan układu: położenia, PĘDY i masy spoczynkowe.

Trzymanie pędu zamiast prędkości nie jest kosmetyką. Leapfrog relatywistyczny
działa na ``dp/dt = F``, więc pęd jest naturalną zmienną stanu; wersja trzymająca
``v`` musiała przy każdym półkroku robić przejście v→p→v, co kosztuje dwa
pierwiastki i traci cyfry znaczące bez żadnego zysku.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bone import relativity as sr


@dataclass
class State:
    positions: np.ndarray  # (N,3) float64
    momenta: np.ndarray  # (N,3) float64
    masses: np.ndarray  # (N,)  float64, masa spoczynkowa
    time: float = 0.0
    step: int = 0
    #: cache siły z ostatniego wywołania — leapfrog KDK potrzebuje jednego
    #: policzenia siły na krok, o ile przenosi ją między krokami
    forces: np.ndarray | None = field(default=None, repr=False)
    potential: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float64)
        self.momenta = np.ascontiguousarray(self.momenta, dtype=np.float64)
        self.masses = np.ascontiguousarray(self.masses, dtype=np.float64)
        if self.positions.shape != self.momenta.shape:
            raise ValueError("positions i momenta muszą mieć ten sam kształt")
        if self.positions.shape[0] != self.masses.shape[0]:
            raise ValueError("liczba mas nie zgadza się z liczbą cząstek")

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    @property
    def total_mass(self) -> float:
        return float(self.masses.sum())

    def velocities(self, c: float) -> np.ndarray:
        return sr.velocity(self.masses, self.momenta, c)

    def gamma(self, c: float) -> np.ndarray:
        return sr.gamma(self.masses, self.momenta, c)

    def speed_over_c(self, c: float) -> np.ndarray:
        return sr.speed_over_c(self.masses, self.momenta, c)

    def center_of_mass(self) -> np.ndarray:
        return np.average(self.positions, axis=0, weights=self.masses)

    def extent(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def copy(self) -> State:
        return State(
            positions=self.positions.copy(),
            momenta=self.momenta.copy(),
            masses=self.masses.copy(),
            time=self.time,
            step=self.step,
        )

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.positions).all() and np.isfinite(self.momenta).all())
