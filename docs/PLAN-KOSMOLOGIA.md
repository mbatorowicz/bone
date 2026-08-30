# Plan przebudowy: pełny model kosmologiczny ΛCDM

Rozbicie na małe kroki. Każdy krok ma **jeden cel**, **listę plików** i **kryterium
odbioru** — najczęściej test, który przed krokiem nie istnieje albo nie przechodzi,
a po kroku przechodzi.

Kolejność jest tak dobrana, żeby repozytorium było zielone jak najdłużej: najpierw
czyste dodatki (etapy 1–3, nic nie psują), potem jeden krótki przełącznik rdzenia
(etap 4), potem sprzątanie (etap 7).

---

## Model docelowy — dla przypomnienia

Współrzędne współporuszające `x`, pęd kanoniczny `p = a² ẋ`, potencjał pekuliarny:

```
dx/dt = p / a²
dp/dt = -∇Φ,        ∇²Φ = (4πG ρ̄₀ / a) δ
```

Faktoryzacja `Φ = Φ̃/a`, gdzie `∇²Φ̃ = 4πG ρ̄₀ δ` nie zależy od `a`. Leapfrog KDK
po `ln a`:

- drift: `Δx = p · ∫ da / (a³ H(a))`
- kick: `Δp = -∇Φ̃ · ∫ da / (a² H(a))`

Jednostki GADGET: długość Mpc/h, masa 10¹⁰ M☉/h, prędkość km/s. Stąd `G = 43,0071`,
`H₀ = 100`, jednostka czasu `(Mpc/h)/(km/s)` ≈ 977,8 Gyr/h — czyli czas Hubble'a
`1/H₀` to 9,78 h⁻¹ Gyr. (Wartość 977,8 **Myr** dotyczy jednostek `kpc/h`, których
GADGET używa domyślnie; tu pudła liczymy w Mpc/h.)

---

## Etap 1 — Fundamenty kosmologiczne

Czyste dodatki. Nic z istniejącego kodu nie jest ruszane, stare testy przechodzą.

### 1.1 Zależności i metadane
- Pliki: `pyproject.toml`
- Dodać `scipy>=1.11` (KD-tree do FOF, union-find, interpolacja tabel transferu).
  Zaktualizować `description` i `keywords`.
- Odbiór: `pip install -e ".[dev]"` przechodzi, `import scipy` działa.

### 1.2 Układ jednostek
- Pliki: `src/bone/units.py` (nowy), `tests/test_units.py` (nowy)
- `G = 43.0071`, `H100 = 100.0`, `MPC_KM`, `GYR_PER_CODE_TIME`, `critical_density_0(h)`.
- Odbiór: `ρ_crit,0 = 2,7754·10¹` w jednostkach kodu (czyli 2,7754·10¹¹ M☉/h/Mpc³)
  z błędem względnym poniżej 10⁻⁴.

### 1.3 Tło: dataclassa i E(a)
- Pliki: `src/bone/cosmology.py` (nowy), `tests/test_cosmology.py` (nowy)
- `@dataclass(frozen=True) Cosmology`: `h`, `Omega_m`, `Omega_b`, `Omega_L`, `n_s`,
  `sigma8`, `T_cmb`. Pola pochodne: `Omega_r` z `T_cmb`, `Omega_k` z warunku sumy.
  Metody `E(a)`, `H(a)`.
- Odbiór: `E(1) == 1` dokładnie; dla EdS (`Omega_m=1, Omega_L=0, Omega_r=0`)
  `H(a) = H₀ a^{-3/2}` do 10⁻¹²; ujemne `Omega_m` odrzucone wyjątkiem.

### 1.4 Czynnik wzrostu D(a) i tempo f(a)
- Pliki: `src/bone/cosmology.py`, `tests/test_cosmology.py`
- `D(a) ∝ H(a) ∫₀^a da'/(a' H(a'))³`, normalizacja `D(1) = 1`.
  `f(a) = dlnD/dlna` liczone różnicą centralną po `ln a`.
- Odbiór: dla EdS `D(a)/D(1) = a` do 10⁻⁶ i `f = 1` do 10⁻⁵; dla Planck18
  `f(1) ≈ Omega_m(1)^0,55` do 2%.

### 1.5 Wiek wszechświata
- Pliki: `src/bone/cosmology.py`, `tests/test_cosmology.py`
- `age(a) = ∫₀^a da'/(a' H(a'))` przeliczone na Gyr.
- Odbiór: dla Planck18 (`h=0,6736`, `Omega_m=0,3153`) wiek wychodzi 13,80 Gyr ±0,02.

### 1.6 Czynniki drift i kick
- Pliki: `src/bone/cosmology.py`, `tests/test_cosmology.py`
- `drift_factor(a1, a2) = ∫ da/(a³H)`, `kick_factor(a1, a2) = ∫ da/(a²H)`,
  kwadratura Gaussa-Legendre'a stałego rzędu (bez zależności od scipy w gorącej pętli).
- Odbiór: zgodność z gęstą kwadraturą trapezową (10⁵ punktów) do 10⁻¹⁰ na
  przedziale `a ∈ [0,02, 1]`; addytywność `f(a1,a2) + f(a2,a3) = f(a1,a3)`.

### 1.7 Transfer function Eisenstein & Hu 1998
- Pliki: `src/bone/power.py` (nowy), `tests/test_power.py` (nowy)
- Pełna formuła EH98 z oscylacjami barionowymi oraz wariant `nowiggle`.
- Odbiór: `T(k → 0) → 1` (dla `k = 10⁻⁵` różnica poniżej 10⁻³); `T` maleje
  monotonicznie dla `k > 0,5 h/Mpc`; wersja pełna i `nowiggle` zgadzają się co do
  obwiedni w granicach 15%, a różnią oscylacyjnie w okolicy `k ≈ 0,07 h/Mpc`.

### 1.8 Normalizacja przez σ₈
- Pliki: `src/bone/power.py`, `tests/test_power.py`
- `sigma_R(R)` z oknem top-hat w przestrzeni Fouriera, wyznaczenie amplitudy `A`
  tak, by `sigma_R(8) = sigma8`.
- Odbiór: po normalizacji `sigma_R(8)` odtwarza zadane `sigma8` do 10⁻⁶;
  `sigma_R` maleje monotonicznie z `R`.

### 1.9 Widmo z tabeli CAMB/CLASS
- Pliki: `src/bone/power.py`, `tests/test_power.py`
- Klasa `PowerSpectrum` z dwoma źródłami: analityczne EH98 albo tabela `k, P(k)`
  z pliku. Interpolacja log-log, ekstrapolacja `k^{n_s}` poniżej zakresu i `k^{-3}`
  powyżej. Wspólny interfejs `__call__(k, a)` zawierający `D(a)²`.
- Odbiór: zapis widma EH98 do pliku i wczytanie z powrotem daje ten sam `σ₈`
  z dokładnością 0,5%; ekstrapolacja nie produkuje `NaN` ani wartości ujemnych.

---

## Etap 2 — Periodyczna geometria i solver PM

Nadal dodatki: nowy backend `pm` powstaje **obok** istniejącego `mesh`.

### 2.1 Periodyczne pudło i CIC z zawijaniem
- Pliki: `src/bone/grid.py`, `tests/test_grid.py` (nowy)
- Klasa `PeriodicBox(length, ng)` z `h = L/ng`, metodą `wrap(positions)` oraz
  `cic_weights` używającym indeksów modulo `ng`. Stara `Box` zostaje na razie
  nietknięta — usunięta w kroku 7.2.
- Odbiór: wagi CIC sumują się do 1 dla każdej cząstki; masa złożona na siatkę
  równa się sumie mas do 10⁻¹²; cząstka w `x = L - h/4` rozkłada masę na komórki
  po obu stronach ściany.

### 2.2 Periodyczny PM — rdzeń numpy
- Pliki: `src/bone/backends/pm.py` (nowy), `tests/test_pm.py` (nowy)
- CIC na stałą siatkę, `rfftn`, funkcja Greena `-4πGρ̄₀/k²` z wyzerowanym modem
  `k = 0`, dekonwolucja CIC (przeniesiona z `mesh.py`, bez zmian), gradient
  różnicą centralną 4. rzędu przez `np.roll` — zawijanie jest teraz poprawne.
- Odbiór: `ΣF = 0` do 10⁻¹⁰ względnie; średnia gęstość nie generuje siły
  (jednorodna siatka daje `F = 0`); dwie cząstki w środku pudła w odległości
  `10h` dają siłę zgodną z `1/r²` w granicach 3%.

### 2.3 Periodyczny PM — ścieżka torch
- Pliki: `src/bone/backends/pm.py`, `tests/test_pm.py`
- Odbiór: parzystość CPU–GPU do 10⁻⁴ względnie (test pomijany bez CUDA).

### 2.4 Suma Ewalda — postać bezpośrednia
- Pliki: `src/bone/backends/ewald.py` (nowy), `tests/test_ewald.py` (nowy)
- Suma po przestrzeni rzeczywistej (człon `erfc`) i po modach `k`, parametr
  `α = 2/L`, obcięcia dobrane tak, żeby oba człony były zbieżne.
- Odbiór: zgodność z jawnym sumowaniem obrazów `8³` przy dużym obcięciu do 10⁻⁶;
  cząstka w środku prostej sieci sześciennej ma siłę zero do 10⁻¹⁰.

### 2.5 Suma Ewalda — tablicowanie
- Pliki: `src/bone/backends/ewald.py`, `tests/test_ewald.py`
- Poprawka `F_Ewald − F_najbliższy_obraz` stablicowana na siatce 64³ w jednej
  ósemce pudła, interpolacja trójliniowa, rozwinięcie przez symetrie.
- Odbiór: interpolacja zgadza się z dokładną sumą Ewalda do 10⁻³ względnie na
  1000 losowych separacji; tablica liczy się poniżej 10 s i jest cache'owana.

### 2.6 Wzorzec dla próbki cząstek
- Pliki: `src/bone/backends/ewald.py`, `tests/test_ewald.py`
- `ewald_forces_for(positions, masses, rows, box, G)` — odpowiednik dzisiejszego
  `exact_forces_for`, ale periodyczny.
- Odbiór: dla `N = 200` zgodność z pełnym `O(N²)` liczonym wprost sumą Ewalda
  do 10⁻⁴ względnie.

### 2.7 Zmierzony błąd PM
- Pliki: `src/bone/diagnostics.py`, `tests/test_pm.py`
- Podmiana wzorca w `measure_force_error` na Ewalda.
- Odbiór: na polu kosmologicznym z etapu 3 błąd RMS PM poniżej 5% dla siatki PM
  dwukrotnie gęstszej niż siatka cząstek — i **rośnie**, gdy siatkę rozrzedzić
  (czyli pomiar naprawdę mierzy metodę, a nie szum).

---

## Etap 3 — Warunki początkowe

### 3.1 Gaussowskie pole gęstości
- Pliki: `src/bone/ics.py` (nowy), `tests/test_ics.py` (nowy)
- Biały szum w przestrzeni rzeczywistej → `rfftn` → mnożenie przez `√(P(k)·V)`.
  Hermitowskość wychodzi automatycznie, bez ręcznego sklejania płaszczyzn Nyquista.
- Odbiór: zmierzone `P(k)` z pola zgadza się z zadanym w granicach szumu
  próbkowania (test χ² po binach, poziom 5%); pole po `irfftn` jest rzeczywiste;
  `⟨δ⟩ = 0` do 10⁻¹²; ten sam `seed` daje ten sam wynik.

### 3.2 Zel'dovich (1LPT)
- Pliki: `src/bone/ics.py`, `tests/test_ics.py`
- Potencjał przemieszczeń `φ_k = -δ_k/k²`, `Ψ = ∇φ`.
- Odbiór: `∇·Ψ = -δ` na siatce z błędem na poziomie dyskretyzacji (poniżej 1%
  dla modów `k < k_Nyq/4`).

### 3.3 2LPT
- Pliki: `src/bone/ics.py`, `tests/test_ics.py`
- Drugi potencjał ze źródła `Σ_{i<j}(φ_,ii φ_,jj − φ_,ij²)`, sześć dodatkowych FFT.
- Odbiór: przy amplitudzie skalowanej do zera 2LPT zbiega do 1LPT (różnica
  przemieszczeń maleje liniowo z `D`); skośność pola gęstości dla 2LPT jest
  bliższa przewidywaniu teorii perturbacji (`S₃ = 34/7`) niż dla ZA.

### 3.4 Złożenie stanu początkowego
- Pliki: `src/bone/ics.py`, `tests/test_ics.py`
- `make_initial_state(cfg)`: siatka `q` (`N = n_grid³`), przemieszczenia,
  prędkości `v = a² f D H Ψ`, masa cząstki `m_p = Ω_m ρ_crit,0 L³ / N`.
- Odbiór: `m_p` zgodne ze wzorem do 10⁻¹²; `Σp = 0` do 10⁻¹⁰ względnie;
  wszystkie pozycje w `[0, L)`; brak `NaN`.

### 3.5 Weryfikacja końcowa IC
- Pliki: `tests/test_ics.py`
- Odbiór: `σ₈` zmierzone z pola liniowego ekstrapolowanego do `z = 0` zgadza się
  z zadanym w granicach 2% dla `L = 200 Mpc/h`, `n_grid = 128`.

---

## Etap 4 — Przełącznik rdzenia

Tu repozytorium przestaje być zgodne wstecz. Etap jest celowo krótki.

### 4.1 Stan współporuszający
- Pliki: `src/bone/state.py`
- `a` zamiast `time` (z właściwością `redshift`), `peculiar_velocity(a) = p/a`,
  `wrap(box)`. Usunięcie `gamma`, `speed_over_c`, `velocities(c)`,
  `center_of_mass`, `extent`.
- Odbiór: `State` importuje się bez `bone.relativity`; `wrap` jest idempotentne.

### 4.2 Całkowanie po ln a
- Pliki: `src/bone/integrator.py`
- KDK z czynnikami drift/kick z `cosmology.py`, stały `Δln a` plus opcjonalne
  kryterium przyspieszeniowe. Zachowana zasada „jedno liczenie sił na krok".
- Odbiór: liczba wywołań backendu równa liczbie kroków; przejście `a₀ → a₁`
  w jednym kroku i w dziesięciu daje ten sam `a₁` do 10⁻¹⁴.

### 4.3 Test wzrostu pojedynczego modu
- Pliki: `tests/test_growth.py` (nowy)
- Bieg z jednym modem sinusoidalnym o małej amplitudzie, `z = 49 → z = 0`.
- Odbiór: amplituda rośnie jak `D(a)` z błędem poniżej 2%.

### 4.4 Test integralny na pełnym polu
- Pliki: `tests/test_growth.py`
- Bieg `L = 200 Mpc/h`, `n_grid = 64`, `z = 49 → z = 0`.
- Odbiór: `P_zmierzone(k) / P_liniowe(k) = 1 ± 3%` dla `k < 0,1 h/Mpc`
  (skale wciąż liniowe), oraz **powyżej** 1 dla `k > 0,5 h/Mpc` (wzmocnienie
  nieliniowe — jeśli go nie ma, to znaczy, że symulacja nic nie robi).

---

## Etap 5 — Diagnostyka kosmologiczna

### 5.1 Energie w konwencji kosmologicznej
- Pliki: `src/bone/diagnostics.py`
- `T = Σ p²/(2 m a²)`, `U = ½ Σ m φ` z potencjałem pekuliarnym. Usunięcie
  wszystkich wielkości SR.
- Odbiór: dla stanu początkowego `T` i `U` mają właściwy rząd wielkości względem
  przewidywania liniowego (test rzędu, nie równości).

### 5.2 Residuum Layzera-Irvine'a
- Pliki: `src/bone/diagnostics.py`, `tests/test_layzer_irvine.py` (nowy)
- `ΔE_LI(a) = (T+U)|_a − (T+U)|_{a₀} + ∫_{a₀}^{a} (2T + U) da/a`, akumulowane
  trapezami krok po kroku. To zastępuje `E_drift` jako główny wskaźnik jakości —
  w rozszerzającym się wszechświecie energia po prostu nie jest zachowana.
- Odbiór: `|ΔE_LI| / (T + |U|) < 10⁻²` na biegu 200 kroków; wartość **maleje**
  przy zmniejszeniu `Δln a` dwukrotnie.

### 5.3 Zmierzony wzrost i σ₈
- Pliki: `src/bone/diagnostics.py`, `tests/test_growth.py`
- `D_zmierzone(a)` z amplitudy modów `k < 4·k_fundamental`, `σ₈(a)` z pola CIC.
- Odbiór: oba zgadzają się z teorią liniową do 3% w biegu o małej amplitudzie.

### 5.4 Podpięcie błędu siły
- Pliki: `src/bone/engine.py`, `src/bone/diagnostics.py`
- `error_check_every` używa wzorca Ewalda z kroku 2.6.
- Odbiór: raportowany błąd jest liczbą skończoną i rośnie po rozrzedzeniu siatki PM.

---

## Etap 6 — Konfiguracja, presety, silnik

### 6.1 Sekcje kosmologiczne
- Pliki: `src/bone/config.py`
- `CosmologyConfig` (h, Omega_m, Omega_b, Omega_L, n_s, sigma8, T_cmb),
  `ICConfig` (box_size, n_grid, z_start, lpt_order, transfer, transfer_file, seed).
- Odbiór: `to_flat` / `from_flat` round-trip bezstratny.

### 6.2 Pozostałe sekcje
- Pliki: `src/bone/config.py`
- `SolverConfig` (pm_grid, device, error_check_every, error_check_sample),
  `TimeConfig` (z_end, dlna_max, accuracy, adaptive), `AnalysisConfig`
  (pk_bins, fof_linking, fof_min_particles, analyse_every), `RunConfig`
  (out_dir, snapshot_redshifts, diagnostics_every, live_every, point_stride).
- Odbiór: żadne pole z modelu STW/kształtów nie zostaje.

### 6.3 Schemat UI
- Pliki: `src/bone/config.py`
- Nowe grupy: „Kosmologia", „Pudło i warunki początkowe", „Czas", „Solver",
  „Analiza", „Widok i zapis". Podział `RUNTIME_KEYS` / `STARTUP_KEYS` —
  w kosmologii prawie wszystko jest startowe, bo `G` przestaje być suwakiem.
- Odbiór: istniejący test „każda kontrolka UI ma pole w configu" przechodzi
  po aktualizacji.

### 6.4 Presety
- Pliki: `src/bone/config.py`
- `planck18_small` (L = 100 Mpc/h, 128³), `planck18_medium` (L = 200, 256³),
  `bao` (L = 1000 — pik BAO widoczny w ξ(r)), `linear_growth` (mała amplituda,
  test wzrostu), `precision` (małe pudło, gęsta siatka PM, mały krok).
- Odbiór: każdy preset buduje silnik i wykonuje 3 kroki bez wyjątku.

### 6.5 Silnik
- Pliki: `src/bone/engine.py`
- Przepięcie na `ics.make_initial_state`, usunięcie `Cooling`, przekazywanie
  `Cosmology` do backendu i integratora.
- Odbiór: `Engine().advance(10)` działa dla domyślnej konfiguracji.

---

## Etap 7 — Sprzątanie

### 7.1 Usunięcie warstwy STW i chłodzenia
- Usunąć: `src/bone/relativity.py`, `tests/test_relativity.py`,
  `src/bone/cooling.py`, `tests/test_cooling.py`, `src/bone/spawn.py`,
  `tests/test_shapes.py`.
- Odbiór: `rg -n "relativity|cooling|spawn|gamma|beta_max"` w `src/` nie zwraca nic.

### 7.2 Usunięcie backendów nieperiodycznych
- Usunąć: `src/bone/backends/mesh.py`, `src/bone/backends/exact.py`, starą klasę
  `Box` i `fit_box` z `grid.py`. Zaktualizować `backends/__init__.py`
  (`make_backend` zwraca `pm`, wzorcem jest `ewald`). Przepisać
  `tests/test_backends.py` — w szczególności `test_mesh_is_not_periodic`
  zamienia się w swoje przeciwieństwo.
- Odbiór: `pytest tests/test_backends.py` na zielono.

### 7.3 Usunięcie martwych zasobów
- Usunąć: `legacy/`, `runs/*`, `scripts/plot_shapes.py`, `scripts/plot_filament.py`,
  `scripts/find_clumps.py`, `scripts/run_sim.py`, obrazki `docs/frag_*.png`
  i `docs/shapes*.png`.
- Odbiór: `git status` czysty po commicie, README nie odsyła do nieistniejących plików.

### 7.4 Przepisanie pozostałych testów
- Pliki: `tests/test_config_and_io.py`, `tests/test_conservation.py`
- `test_conservation.py` → `test_integration.py`: zamiast zachowania energii
  i momentu pędu sprawdza residuum LI, zbieżność z krokiem i wykrywanie `NaN`.
- Odbiór: pełny `pytest` na zielono.

### 7.5 Bramka jakości
- Odbiór: `ruff check` bez uwag, `pytest -q` bez błędów i bez pominiętych testów
  innych niż CUDA.

---

## Etap 8 — Warstwa analizy

### 8.1 Pomiar widma mocy
- Pliki: `src/bone/analysis/__init__.py`, `src/bone/analysis/power_measure.py`
  (nowe), `tests/test_analysis.py` (nowy)
- CIC → `rfftn` → uśrednianie po powłokach `k`, dekonwolucja okna CIC, odjęcie
  szumu śrutowego `1/n̄`.
- Odbiór: dla jednorodnego procesu Poissona zmierzone `P(k)` po **nie**odjęciu
  szumu równa się `1/n̄` w granicach 5%, a po odjęciu jest zgodne z zerem.

### 8.2 Funkcja korelacji
- Pliki: `src/bone/analysis/correlation.py`, `tests/test_analysis.py`
- `ξ(r)` transformatą Hankela/FFT z `P(k)`.
- Odbiór: dla analitycznego `P(k) ∝ exp(-k²σ²)` odtwarza znany wynik do 1%;
  na presecie `bao` widoczny pik w okolicy `r ≈ 105 Mpc/h`.

### 8.3 Halo finder FOF
- Pliki: `src/bone/analysis/halos.py`, `tests/test_analysis.py`
- `b = 0,2` średniej separacji międzycząstkowej, `scipy.spatial.cKDTree` +
  `scipy.sparse.csgraph.connected_components`, obsługa periodyczności przez
  `boxsize` w KD-tree.
- Odbiór: na sztucznym polu z 10 rozłącznymi kulami po 500 cząstek FOF znajduje
  dokładnie 10 halo o właściwej masie; dwie kule sklejone przez ścianę pudła
  są rozpoznane jako jedno halo.

### 8.4 Funkcja masy halo
- Pliki: `src/bone/analysis/mass_function.py`, `tests/test_analysis.py`
- Przewidywania Press-Schechter i Sheth-Tormen z `σ(M)` liczonego z `power.py`.
- Odbiór: całka `∫ (M/ρ̄) dn/dlnM dlnM` dla Press-Schechter wynosi 1 do 2%
  (cała masa w halo — warunek normalizacyjny formalizmu).

### 8.5 Porównanie symulacji z teorią
- Pliki: `tests/test_analysis.py`
- Odbiór: na presecie `planck18_small` zmierzona funkcja masy zgadza się z
  Sheth-Tormen co do rzędu wielkości w zakresie mas rozdzielonych (powyżej
  100 cząstek na halo). Test dokumentuje rozbieżność, nie udaje jej braku.

---

## Etap 9 — Zapis i wiersz poleceń

### 9.1 Formaty zapisu
- Pliki: `src/bone/io/checkpoint.py`, `src/bone/io/trajectory.py`,
  `src/bone/io/live.py`
- `a` zamiast `time`, zapis `Cosmology` i `box_size` w `config.json`,
  `shade` liczone z lokalnej gęstości zamiast z `β`.
- Odbiór: round-trip checkpointu odtwarza `a`, pudło i kosmologię; klatka
  trajektorii wczytuje się po indeksie.

### 9.2 Komenda `run`
- Pliki: `src/bone/cli.py`
- Argumenty: `--box`, `--ngrid`, `--pm-grid`, `--z-start`, `--z-end`,
  `--omega-m`, `--omega-b`, `--sigma8`, `--h`, `--ns`, `--lpt`, `--transfer`,
  `--seed`, `--steps`, `--out`. Log konsolowy: `z`, `a`, `D(a)`, `σ₈`, `ΔE_LI`.
- Odbiór: `python -m bone run --preset planck18_small --steps 20` kończy się
  kodem 0 i zapisuje snapshot.

### 9.3 Komenda `analyse`
- Pliki: `src/bone/cli.py`
- Liczy `P(k)`, `ξ(r)`, FOF i funkcję masy ze snapshotu, zapisuje do `.npz`
  i wypisuje podsumowanie.
- Odbiór: `python -m bone analyse runs/latest` produkuje plik z widmem
  i listą halo.

### 9.4 Komenda `bench`
- Pliki: `src/bone/cli.py`
- Skalowanie PM po `n_grid` i `pm_grid`, CPU vs CUDA.
- Odbiór: tabela czasów bez wyjątków dla trzech rozmiarów.

---

## Etap 10 — Studio

### 10.1 Status serwera
- Pliki: `src/bone/studio/server.py`
- Do `/api/status` dochodzą: `z`, `a`, `H_over_H0`, `age_gyr`, `D_measured`,
  `D_theory`, `sigma8_measured`, `li_residual`, `n_halos`, `m_largest`.
- Odbiór: `curl /api/status` zwraca komplet pól w trakcie biegu.

### 10.2 Pakiet live
- Pliki: `src/bone/io/live.py`, `src/bone/studio/server.py`
- Nagłówek niesie `box_size` i `a` zamiast `half` i `time`; `shade` to
  `log₁₀(1+δ)` odczytane z siatki CIC.
- Odbiór: test układu bufora (rozszerzony `test_live_buffer_layout`) przechodzi.

### 10.3 HUD
- Pliki: `src/bone/studio/web/index.html`, `app.js`, `styles.css`
- Zamiana `⟨γ⟩`, `γ_max`, `β_max`, `wirial`, `r½/r₀` na `z`, `a`, `H/H₀`,
  wiek [Gyr], `D` zmierzone kontra teoria, `σ₈`, residuum LI, liczba halo.
  Sparkline przepięty na residuum LI.
- Odbiór: HUD nie pokazuje `NaN` ani `undefined` na świeżym biegu.

### 10.4 Scena
- Pliki: `src/bone/studio/web/app.js`
- Stałe pudło z ramką, bez auto-dopasowania rozmiaru; mapa barw przepięta
  z „zimno-gorąco po β" na skalę gęstości pokazującą sieć kosmiczną.
- Odbiór: kamera nie skacze między klatkami; przy `z = 0` widać włókna i pustki.

---

## Etap 11 — Dokumentacja i skrypty

### 11.1 `scripts/plot_power.py`
Zmierzone `P(k)` w kilku redshiftach na tle teorii liniowej. Odbiór: wykres
powstaje ze snapshotu bez ręcznych parametrów.

### 11.2 `scripts/plot_slice.py`
Plaster pudła o zadanej grubości, gęstość w skali logarytmicznej. Odbiór:
na presecie `planck18_medium` przy `z = 0` widać sieć kosmiczną.

### 11.3 `scripts/plot_mass_function.py`
Zmierzona funkcja masy kontra Press-Schechter i Sheth-Tormen, z zaznaczoną
granicą rozdzielczości. Odbiór: granica jest na wykresie, nie w domyśle.

### 11.4 README
Przepisany pod nowy model: równania ruchu, układ jednostek, skąd bierze się masa
cząstki, czym jest residuum Layzera-Irvine'a, gdzie PM kłamie (rozdzielczość
oczka, brak sił krótkozasięgowych), czego model nie zawiera (barionów, promieniowania
jako składnika dynamicznego, niegaussowskości, masywnych neutrin).
Odbiór: żadne odesłanie w README nie prowadzi do nieistniejącego pliku.

### 11.5 `docs/IDEA.md`
Nowy kanon zastępujący „Bone SR". Odbiór: dokument nie wspomina o STW ani
o dziesięciu bryłach startowych.

---

## Czego ten model nadal nie będzie zawierał

Warto to zapisać teraz, żeby nie trzeba było tego odkrywać później:

- **Brak sił krótkozasięgowych.** Czysty PM rozdziela grawitację tylko do oczka
  siatki. Wewnętrzna struktura halo jest zaniżona. Naturalnym rozszerzeniem jest
  P³M albo TreePM.
- **Brak barionów.** To symulacja samej ciemnej materii. Gaz, chłodzenie
  radiacyjne i tworzenie gwiazd wymagałyby SPH albo siatki hydrodynamicznej.
- **Brak masywnych neutrin i dynamicznej ciemnej energii.** `Omega_r` wchodzi do
  `E(a)`, ale nie jest składnikiem dynamicznym; `w = -1` na sztywno.
- **Warunki początkowe gaussowskie.** Brak niegaussowskości typu `f_NL`.
- **Jeden poziom rozdzielczości.** Brak zagnieżdżonych warunków początkowych
  (zoom-in).
