"""Układ jednostek kodu: Mpc/h, 10¹⁰ M☉/h, km/s.

To konwencja GADGET-a. Wybrana nie z sentymentu, tylko dlatego, że jest jedynym
układem, w którym wszystkie trzy liczby, na które się patrzy w kosmologii, są
rzędu jedności: rozmiary pudeł to setki, prędkości pekuliarne to setki, a masy
halo to jednostki–tysiące. Zapis w SI dawałby wykładniki 10²², 10⁴⁰ i 10⁵ w tym
samym wyrażeniu, co w float64 kosztuje cyfry znaczące na każdym mnożeniu.

Kluczowa i niebanalna własność: ``h`` znika ze wszystkich stałych. Długość niesie
``h⁻¹`` i masa niesie ``h⁻¹``, więc w ``G·M/L`` się skracają. Z tego samego powodu
``H₀ = 100 h km/s/Mpc`` wyrażone w jednostce czasu ``(Mpc/h)/(km/s)`` daje równe
``100`` — bez ``h``. Dlatego solver nigdy nie musi wiedzieć, ile wynosi ``h``;
potrzebuje tego dopiero przeliczanie na lata i megaparseki na wykresie.

Wartość ``G`` jest podana za GADGET-em, czyli policzona ze starszych stałych
``G = 6,672·10⁻⁸`` cgs i ``M☉ = 1,989·10³³`` g. Z dzisiejszym CODATA wyszłoby
``43,021`` — o 0,03% więcej. Trzymamy wersję GADGET-a, żeby wyniki dały się
porównywać z literaturą, w której praktycznie każda symulacja używa tej liczby;
0,03% jest o dwa rzędy wielkości poniżej błędu samego solvera PM.
"""

from __future__ import annotations

import math

#: kilometry w jednym megaparseku — mostek między jednostką długości a prędkości
MPC_KM = 3.085677581491367e19

#: sekundy w miliardzie lat julijskich (365,25 dnia)
GYR_S = 3.15576e16

#: stała grawitacji w jednostkach kodu, [Mpc/h · (km/s)² / (10¹⁰ M☉/h)]
G = 43.0071

#: H₀ w jednostkach kodu dla dowolnego h — patrz uwaga o skracaniu się h
H100 = 100.0

#: ile miliardów lat (razy h⁻¹) trwa jednostka czasu kodu, czyli (Mpc/h)/(km/s)
GYR_PER_CODE_TIME = MPC_KM / GYR_S


def critical_density_0(h: float = 1.0) -> float:
    """Gęstość krytyczna dziś, ``3H₀²/(8πG)``, w jednostkach kodu.

    Wynik nie zależy od ``h`` — to nie przeoczenie, tylko konsekwencja układu
    jednostek. W jednostkach fizycznych ``ρ_crit,0 = 2,7754·10¹¹ h² M☉/Mpc³``,
    a przeliczenie na ``10¹⁰ M☉/h`` na ``(Mpc/h)³`` wnosi czynnik ``h⁻²``.
    Argument zostaje w sygnaturze, bo wywołanie ma być czytelne w miejscu użycia
    i bo ``h ≤ 0`` warto odrzucić tu, a nie trzy warstwy dalej.
    """
    if not h > 0.0:
        raise ValueError(f"h musi być dodatnie, dostałem {h}")
    return 3.0 * H100**2 / (8.0 * math.pi * G)
