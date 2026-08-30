"""Wspólny kontrakt backendów liczących grawitację.

Każdy backend zwraca siłę ORAZ potencjał w jednym przebiegu. Potencjał jest
przy okazji prawie darmowy, a bez niego nie da się policzyć energii — czyli
jedynej liczby, która mówi, czy symulacja jeszcze jest symulacją.

Konwencja (softening Plummera, ε o wymiarze długości):

    φ_i = −G Σ_{j≠i} m_j / √(r_ij² + ε²)
    F_i = −G m_i Σ_{j≠i} m_j (x_i − x_j) / (r_ij² + ε²)^{3/2}
    U   = ½ Σ_i m_i φ_i

Para (φ, F) jest spójna: F = −m∇φ dokładnie dla tego samego ε. To dlatego
dryf energii jest sensowną miarą jakości całkowania, a nie artefaktem tego,
że siła i potencjał pochodzą z dwóch różnych modeli.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Field:
    """Wynik jednego wywołania backendu."""

    force: np.ndarray  # (N,3)
    potential: np.ndarray  # (N,) potencjał właściwy φ_i

    def energy(self, masses: np.ndarray) -> float:
        """U = ½ Σ mᵢφᵢ — połowa, bo każda para liczy się dwa razy."""
        return 0.5 * float(np.dot(masses, self.potential))


class Backend(ABC):
    """Interfejs solwera grawitacji."""

    name: str = "base"

    @abstractmethod
    def compute(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        G: float,
        softening: float,
    ) -> Field:
        """Policz siły i potencjał dla podanego stanu."""

    @property
    def approximate(self) -> bool:
        """Czy wynik jest przybliżony. Jeśli tak, błąd powinien być mierzalny."""
        return False

    def effective_softening(self, requested: float) -> float:
        """Softening, którym backend NAPRAWDĘ liczy.

        Solver siatkowy nie rozdzieli skali mniejszej od oczka, więc może
        pracować z większym ε niż zamówione. Zwracanie tej liczby pozwala
        pokazać ją użytkownikowi i porównywać błąd z właściwym wzorcem, zamiast
        po cichu liczyć co innego, niż napisano w panelu.
        """
        return float(requested)

    def describe(self) -> str:
        return self.name

    def close(self) -> None:  # noqa: B027 — hak opcjonalny, backendy CPU nie mają czego zwalniać
        """Zwolnij zasoby (bufory GPU itp.)."""
