"""Dyssypacja: odprowadzanie energii z ruchu nieuporządkowanego.

To jest jedyny element tego modelu, który nie jest zachowawczy, i dlatego jedyny,
który pozwala układowi zrobić coś, czego sama grawitacja zrobić nie może. Bez
niego zwirializowana chmura jest zamknięta: ma rozkład prędkości, który
podtrzymuje ją przeciw dalszemu zapadaniu, a energii nie ma gdzie podziać. Żadne
zagęszczenie w niej nie powstanie, niezależnie od tego, jak długo liczyć.

CO JEST TŁUMIONE. Prędkość każdej cząstki rozkładamy na lokalny przepływ masowy
i odchyłkę od niego, a tłumimy WYŁĄCZNIE odchyłkę:

    v ← v_masowe + (v − v_masowe)·exp(−λΔt)

To nie jest szczegół implementacyjny, to cała fizyka tego modułu. Chłodzenie
radiacyjne w gazie odprowadza energię ruchu termicznego, a nie energię ruchu
całego obłoku — wirujący dysk, który się chłodzi, staje się cieńszy i zimniejszy,
ale nie przestaje wirować.

DLACZEGO NIE TARCIE. Oczywistsze „v ← v·(1−λΔt)" jest błędne i to widowiskowo.
Tłumi ono również ruch masowy, więc każda orbita traci moment pędu i cały układ
spada do środka w czasie 1/λ, niezależnie od swojej fizyki. Wygląda to jak
kolaps grawitacyjny, a jest zwykłym oporem lepkim, i nie da się tego rozpoznać
po samym obrazku — dlatego różnica jest tutaj wypisana, a nie domyślna.

SKALA WYGŁADZANIA. Lokalny przepływ liczymy przez siatkę tymi samymi wagami CIC,
którymi solver rozkłada masę, bo to jest tanie i symetryczne. Ma to konsekwencję,
o której trzeba wiedzieć: „lokalny" znaczy „w skali oczka siatki chłodzenia".
Zbyt gęsta siatka daje kilka cząstek na komórkę i dyspersję zdominowaną przez
szum próbkowania; zbyt zgrubna wlicza do ruchu masowego względny ruch osobnych
zgęstek i chłodzenie zaczyna je sztucznie scalać. Dlatego ``describe`` raportuje
liczbę cząstek na zajętą komórkę — bez tej liczby wynik nie ma jak być oceniony.

BILANS ENERGII. Moduł zwraca ilość odprowadzonej energii, a silnik ją sumuje.
Dzięki temu ``E_drift`` w diagnostyce dalej mierzy jakość CAŁKOWANIA: sprawdzaną
wielkością zachowaną jest E_tot + E_wypromieniowana, a nie samo E_tot. Bez tego
włączenie chłodzenia zamieniłoby główny wskaźnik jakości kodu w licznik czegoś
zupełnie innego.
"""

from __future__ import annotations

import warnings

import numpy as np

from bone import relativity as sr
from bone.config import PhysicsConfig
from bone.grid import Box, cic_weights, fit_box
from bone.state import State

#: Górny limit wzmocnienia tempa przez kontrast gęstości. Bez niego jedna komórka
#: o gęstości tysiąc razy większej od średniej dostawałaby λΔt rzędu setek i jej
#: dyspersja spadałaby do zera w jednym kroku — a to jest właśnie ta skala, na
#: której gęstość z siatki jest najmniej wiarygodna.
_MAX_DENSITY_BOOST = 50.0

#: Poniżej tylu cząstek na zajętą komórkę estymator dyspersji przestaje być
#: estymatorem dyspersji. Przy jednej cząstce w komórce lokalny „przepływ masowy"
#: to jej własna prędkość, odchyłka wychodzi zerowa i chłodzenie po cichu nic nie
#: robi — najgorszy możliwy tryb awarii, bo wygląda jak poprawny bieg.
_MIN_PARTICLES_PER_CELL = 4.0


def auto_grid(n_particles: int) -> int:
    """Bok siatki dający około ośmiu cząstek na zajętą komórkę.

    Ta wielkość MUSI skalować się z liczbą cząstek i dlatego nie ma sensownej
    stałej wartości domyślnej. Kula wypełnia około 0,52 objętości pudła, więc
    zajętych komórek jest ~0,52·ng³; żądanie ośmiu cząstek na komórkę daje
    ng ≈ (N/4,2)^(1/3). Wychodzi 10 dla czterech tysięcy cząstek i 31 dla stu
    dwudziestu tysięcy — czyli dokładnie ten zakres, w którym stała wartość
    myliłaby się o rząd wielkości w jedną albo drugą stronę.
    """
    return int(max(4, min(round((max(n_particles, 1) / 4.2) ** (1 / 3)), 256)))


#: Pola rozkładane na siatkę: masa, trzy składowe pędu masowego i masa·|v|².
#: Trzymane razem jako jedna tablica (N,5), bo wszystkie idą tym samym scatterem
#: i tym samym gatherem — rozdzielone na pięć przebiegów kosztowały dwa razy
#: więcej niż całe rozwiązanie grawitacji na GPU.
_N_FIELDS = 5


class Cooling:
    """Operator chłodzenia. Trzymany przez silnik, bo cache'uje siatkę i urządzenie."""

    def __init__(self, grid: int = 0, margin: float = 0.15, device: str = "cpu") -> None:
        if grid != 0 and grid < 4:
            raise ValueError("siatka chłodzenia poniżej 4 komórek nie ma sensu (0 = automat)")
        #: 0 oznacza „dobierz z liczby cząstek" — patrz `auto_grid`
        self.requested_grid = int(grid)
        self.margin = float(margin)
        self.device_name = device
        self._torch = None
        if device == "cuda":
            import torch

            self._torch = torch
            self._device = torch.device("cuda")
        self._grid: int | None = None
        self._cell: float | None = None
        self._occupied: int = 0
        self._particles: int = 0
        self._warned = False

    @property
    def grid(self) -> int:
        """Bok siatki faktycznie użyty; przed pierwszym użyciem 0 dla automatu."""
        return self._grid if self._grid is not None else self.requested_grid

    def describe(self) -> str:
        if self._cell is None:
            how = "automat" if self.requested_grid == 0 else f"{self.requested_grid}³"
            return f"chłodzenie {how} (jeszcze nie użyte)"
        auto = " (automat)" if self.requested_grid == 0 else ""
        where = "cuda" if self._torch is not None else "cpu"
        ppc = self.particles_per_cell
        note = "" if ppc >= _MIN_PARTICLES_PER_CELL else "  ⚠ za mało cząstek na komórkę"
        return (
            f"chłodzenie {self._grid}³{auto} ({where}), oczko {self._cell:.3g}, "
            f"{ppc:.1f} cząstek/komórkę{note}"
        )

    @property
    def particles_per_cell(self) -> float:
        return self._particles / max(self._occupied, 1)

    def _warn_if_undersampled(self) -> None:
        """Powiedz wprost, gdy siatka jest za gęsta na tyle cząstek.

        Tryb awarii jest tu cichy i dlatego groźny: chłodzenie po prostu przestaje
        działać, bieg kończy się bez błędu, a wniosek „dyssypacja nic nie zmienia"
        jest fałszywy. Lepiej to powiedzieć raz niż pozwolić na taki wynik.
        """
        if self._warned or self.particles_per_cell >= _MIN_PARTICLES_PER_CELL:
            return
        self._warned = True
        warnings.warn(
            f"siatka chłodzenia {self.grid}³ daje tylko {self.particles_per_cell:.1f} "
            f"cząstek na zajętą komórkę — lokalna dyspersja jest wtedy szumem "
            f"próbkowania, a chłodzenie prawie nie działa. Ustaw cooling_grid na 0 "
            f"(automat wybrałby {auto_grid(self._particles)}) albo zwiększ liczbę cząstek.",
            RuntimeWarning,
            stacklevel=3,
        )

    def apply(self, state: State, phys: PhysicsConfig, dt: float) -> float:
        """Ochłodź układ o jeden krok. Zwraca ilość odprowadzonej energii (≥ 0).

        Położenia nie są ruszane, więc energia potencjalna się nie zmienia i
        odprowadzona energia jest DOKŁADNIE ubytkiem energii kinetycznej — nie
        trzeba tu niczego szacować.
        """
        rate = float(phys.cooling_rate)
        if rate <= 0.0 or dt <= 0.0 or state.n < 2:
            return 0.0

        c = float(phys.c)
        masses = state.masses
        before = float(sr.kinetic_energy(masses, state.momenta, c).sum())

        velocities = sr.velocity(masses, state.momenta, c)
        bulk, sigma, density = self._local_fields(state.positions, masses, velocities)

        mean_mass = float(masses.mean())
        lam = rate * self._density_factor(density, masses, float(phys.cooling_density_power))
        lam *= _floor_factor(sigma, _minimum_dispersion(density, phys, c, mean_mass))

        # forma wykładnicza, nie (1 − λΔt): jest dokładnym rozwiązaniem równania
        # d(δv)/dt = −λ·δv, więc pozostaje stabilna i dodatnia przy dowolnie
        # dużym λΔt. Wariant liniowy przy λΔt > 1 odwracałby znak odchyłki.
        residual = velocities - bulk
        cooled = bulk + residual * np.exp(-lam * dt)[:, None]

        # |v_nowe| < c wynika z wypukłości: v_nowe leży na odcinku między v i
        # v_masowe, a v_masowe jest średnią ważoną prędkości podświetlnych.
        momenta = sr.momentum(masses, cooled, c)
        _restore_total_momentum(momenta, state.momenta, masses)
        state.momenta = momenta

        after = float(sr.kinetic_energy(masses, state.momenta, c).sum())
        # Nie zwracamy wartości ujemnej: poprawka pędu może w skrajnym przypadku
        # dodać znikomą ilość energii, a ujemne „wypromieniowanie" zepsułoby
        # bilans, którym mierzymy jakość całkowania.
        return max(before - after, 0.0)

    # ------------------------------------------------------------------ pola

    def _local_fields(
        self, positions: np.ndarray, masses: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lokalny przepływ masowy, dyspersja i gęstość — w położeniach cząstek.

        Uwaga na kolejność działań: z siatki odczytywane są SUMY (masa, pęd,
        masa·|v|²), a ilorazy liczone dopiero na cząstkach. Odczytanie gotowego
        ilorazu z siatki dawałoby średnią nieważoną masą i psuło wynik tam, gdzie
        masy w sąsiednich komórkach są różne.
        """
        if self._grid is None:
            self._grid = (
                auto_grid(positions.shape[0]) if self.requested_grid == 0 else self.requested_grid
            )
        box = fit_box(positions, self._grid, self.margin)

        # masa | masa·vx | masa·vy | masa·vz | masa·|v|²
        fields = np.empty((positions.shape[0], _N_FIELDS), dtype=np.float64)
        fields[:, 0] = masses
        fields[:, 1:4] = masses[:, None] * velocities
        fields[:, 4] = masses * np.einsum("ij,ij->i", velocities, velocities)

        if self._torch is not None:
            summed, occupied = _smooth_torch(self._torch, self._device, positions, fields, box)
        else:
            summed, occupied = _smooth_numpy(positions, fields, box)

        self._cell = box.h
        self._occupied = occupied
        self._particles = int(positions.shape[0])
        self._warn_if_undersampled()

        mass_at = np.maximum(summed[:, 0], 1e-300)
        bulk = summed[:, 1:4] / mass_at[:, None]
        mean_square = summed[:, 4] / mass_at

        # dyspersja z twierdzenia o rozkładzie: ⟨v²⟩ − |⟨v⟩|². Różnica dwóch
        # bliskich liczb, więc przy jednej cząstce w komórce wychodzi drobne
        # ujemne zero — obcięcie jest konieczne, nie kosmetyczne.
        variance = np.maximum(mean_square - np.einsum("ij,ij->i", bulk, bulk), 0.0)
        density = mass_at / (box.h**3)
        return bulk, np.sqrt(variance), density

    @staticmethod
    def _density_factor(density: np.ndarray, masses: np.ndarray, power: float) -> np.ndarray:
        """(ρ/ρ_odniesienia)^power, z ρ_odniesienia = średnia ważona masą.

        Odniesieniem jest bieżąca średnia, nie wartość z chwili startu. Dzięki
        temu ``cooling_rate`` zawsze znaczy „tempo przy średniej gęstości układu"
        i nie trzeba go przeliczać przy zmianie masy czy promienia. Cena: gdy
        cały układ się zagęszcza, tempo przy danej gęstości maleje. Wybór jest
        świadomy — parametr o stałym znaczeniu jest tu wart więcej niż parametr
        o stałej wartości bezwzględnej.
        """
        if abs(power) < 1e-12:
            return np.ones_like(density)
        reference = float(np.dot(masses, density) / max(masses.sum(), 1e-300))
        ratio = density / max(reference, 1e-300)
        return np.minimum(ratio**power, _MAX_DENSITY_BOOST)


def _smooth_numpy(
    positions: np.ndarray, fields: np.ndarray, box: Box
) -> tuple[np.ndarray, int]:
    """Rozłóż pola na siatkę i odczytaj je z powrotem w położeniach cząstek.

    Wszystkie pola idą jednym scatterem i jednym gatherem. Rozdzielenie ich na
    pięć osobnych przebiegów `bincount` było mierzalnie wolniejsze niż całe
    rozwiązanie równania Poissona, co dla operatora pomocniczego jest absurdem.
    """
    idx, wgt = cic_weights(positions, box)
    flat = idx.ravel()
    cells = box.ng**3

    # scatter: indeks złożony (komórka, pole) pozwala zrobić jedno bincount
    composite = (flat[:, None] * _N_FIELDS + np.arange(_N_FIELDS)).ravel()
    weights = (wgt[:, :, None] * fields[:, None, :]).ravel()
    grid_sums = np.bincount(composite, weights=weights, minlength=cells * _N_FIELDS)
    grid_sums = grid_sums.reshape(cells, _N_FIELDS)

    gathered = np.einsum("nkf,nk->nf", grid_sums[flat].reshape(*idx.shape, _N_FIELDS), wgt)
    return gathered, int(np.count_nonzero(grid_sums[:, 0]))


def _smooth_torch(torch, device, positions: np.ndarray, fields: np.ndarray, box: Box):
    """Ta sama operacja na GPU. float32 wystarcza — patrz uwaga o dyspersji.

    Jedyne miejsce, w którym precyzja mogłaby zaboleć, to różnica ⟨v²⟩ − |⟨v⟩|²
    liczona później: przy dyspersji rzędu 0,6 i prędkości masowej rzędu 5 różnica
    wynosi ~0,4 przy wyrazach ~25, czyli sześć rzędów wielkości nad progiem
    float32. Zapas jest duży, ale nie nieskończony — przy prędkościach masowych
    o kolejne dwa rzędy większych od dyspersji ten wybór trzeba by zmienić.
    """
    dtype = torch.float32
    with torch.no_grad():
        pos = torch.as_tensor(positions, dtype=dtype, device=device)
        val = torch.as_tensor(fields, dtype=dtype, device=device)
        origin = torch.as_tensor(box.origin, dtype=dtype, device=device)

        ng = box.ng
        local = (pos - origin) / box.h - 0.5
        base = torch.floor(local)
        frac = local - base
        base = base.to(torch.int64).clamp_(0, ng - 2)

        n = pos.shape[0]
        idx = torch.empty((n, 8), dtype=torch.int64, device=device)
        wgt = torch.empty((n, 8), dtype=dtype, device=device)
        corner = 0
        for dx in (0, 1):
            wx = frac[:, 0] if dx else 1.0 - frac[:, 0]
            ix = base[:, 0] + dx
            for dy in (0, 1):
                wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
                iy = base[:, 1] + dy
                for dz in (0, 1):
                    wz = frac[:, 2] if dz else 1.0 - frac[:, 2]
                    idx[:, corner] = (ix * ng + iy) * ng + base[:, 2] + dz
                    wgt[:, corner] = wx * wy * wz
                    corner += 1

        grid_sums = torch.zeros((ng**3, _N_FIELDS), dtype=dtype, device=device)
        contrib = wgt.unsqueeze(-1) * val.unsqueeze(1)  # (N,8,5)
        grid_sums.index_add_(0, idx.reshape(-1), contrib.reshape(-1, _N_FIELDS))

        gathered = (grid_sums[idx.reshape(-1)].reshape(n, 8, _N_FIELDS) * wgt.unsqueeze(-1)).sum(1)
        occupied = int(torch.count_nonzero(grid_sums[:, 0]).item())
        return gathered.to(torch.float64).cpu().numpy(), occupied


def _minimum_dispersion(
    density: np.ndarray, phys: PhysicsConfig, c: float, mean_mass: float
) -> np.ndarray:
    """Dyspersja (trójwymiarowa), poniżej której chłodzenie nie ma prawa zejść.

    Dwie podłogi, brana większa.

    FIZYCZNA — ``cooling_floor``·c, odpowiednik temperatury, poniżej której
    ośrodek przestaje promieniować. Stała, więc ustala NAJWIĘKSZĄ skalę fragmentu.

    NUMERYCZNA — z warunku, żeby masa Jeansa liczyła co najmniej N cząstek. Przy
    M_J = ρλ³ i λ = σ₁ᴰ√(π/Gρ) wychodzi

        M_J = σ₁ᴰ³·π^{3/2}/(G^{3/2}·√ρ)  ⟹  σ₁ᴰ ≥ [N·m̄·G^{3/2}·√ρ/π^{3/2}]^{1/3}

    Istotny jest wykładnik przy gęstości: ρ^{1/6}. Gęstość pochodzi ze zgrubnej
    siatki chłodzenia i dla zwartych zgęstek jest zaniżona nawet sześćdziesięciokrotnie,
    ale ρ^{1/6} tłumi ten błąd do czynnika dwa. Wariant z progiem na samą długość
    Jeansa zależy od gęstości jak √ρ i dziedziczyłby go niemal w całości — został
    zmierzony i faktycznie nie zadziałał.

    Uwaga na definicję σ: pola z siatki dają dyspersję TRÓJWYMIAROWĄ, a wzory
    Jeansa operują na jednej składowej, stąd dzielenie i mnożenie przez √3.
    """
    physical = float(phys.cooling_floor) * c
    min_particles = float(phys.cooling_min_particles)
    if min_particles <= 0.0 or phys.G <= 0.0 or mean_mass <= 0.0:
        return np.full_like(density, physical)
    jeans_mass = min_particles * mean_mass
    sigma_1d = (
        jeans_mass * phys.G**1.5 * np.sqrt(np.maximum(density, 0.0)) / np.pi**1.5
    ) ** (1.0 / 3.0)
    return np.maximum(sigma_1d * 3.0**0.5, physical)


def _floor_factor(sigma: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Wygaszenie tempa przy dyspersji zbliżającej się do progu ``floor``.

    Czynnik 1 − (σ_min/σ)² jest gładki, znika dokładnie przy σ = σ_min i dąży do
    1 przy σ ≫ σ_min, więc nie wprowadza progu skokowego — chłodzenie zwalnia
    w miarę zbliżania się do podłogi, zamiast wyłączać się nagle.
    """
    # obcięcie od dołu progiem, nie zerem: iloraz jest wtedy zawsze ≤ 1, więc
    # kwadrat nie ma jak się przepełnić, a wynik dla σ ≤ σ_min wychodzi dokładnie 0
    safe = np.maximum(sigma, floor)
    return np.where(floor > 0.0, 1.0 - (floor / np.maximum(safe, 1e-300)) ** 2, 1.0)


def _restore_total_momentum(
    momenta: np.ndarray, reference: np.ndarray, masses: np.ndarray
) -> None:
    """Wymuś, żeby chłodzenie nie zmieniło całkowitego pędu (in place).

    Tłumienie odchyłek od lokalnej średniej powinno zachowywać pęd samo z siebie,
    ale interpolacja CIC nie jest dokładnie tożsamościowa: średnia odczytana
    z siatki różni się od średniej prawdziwej o wielkość rzędu błędu interpolacji.
    Bez tej poprawki resztkowy pęd narastałby przez tysiące kroków i cały układ
    zaczynałby dryfować z kadru — czyli powstałby dokładnie ten artefakt, przed
    którym `spawn.make_state` zabezpiecza warunek początkowy.
    """
    drift = momenta.sum(axis=0) - reference.sum(axis=0)
    total = float(masses.sum())
    if total > 0.0:
        momenta -= np.outer(masses / total, drift)
