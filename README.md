# Bone

Grawitacja N ciał, gaz cząstek i atomy na jednym komputerze. Cztery modele,
jeden silnik, jedna aplikacja w Ruście — okno z panelem albo bieg wsadowy
z wiersza poleceń.

- **`sr`** — odosobniona chmura cząstek. Grawitacja newtonowska, kinematyka
  szczególnej teorii względności, dyssypacja zależna od gęstości.
- **`lcdm`** — próbka materii w modelu standardowym kosmologii (Planck 2018).
  Warunki początkowe z widma mocy, całkowanie po `ln a` od `z = 49` do dziś.
- **`sm`** — Model Standardowy jako klasyczny gaz cząstek. Siedemnaście gatunków
  elementarnych plus hadrony, cztery oddziaływania, rozpady i anihilacja.
  To nie jest kwantowa teoria pola.
- **`qm`** — atomy i orbitale. Wodór i jony wodoropodobne są dokładnym
  rozwiązaniem Schrödingera (`ψ_{nlm} = R_{nl} Y_{lm}`). Atomy wieloelektronowe
  to niezależne elektrony z `Z_eff` Slatera; błąd wobec tablic jonizacji jest
  mierzony i pokazywany. Chmura na ekranie to próbka `|ψ|²`, nie zbiór elektronów.

Błąd przybliżenia jest **mierzony i pokazywany**, nie zakładany. To jedyna
liczba, która odróżnia przybliżenie od usterki.

## Budowanie i uruchamianie

```bash
cd cosmo
cargo build --release          # wynik: target/release/BoneCosmo
cargo test --workspace         # 447 testów
cargo clippy --workspace --all-targets -- -D warnings
```

```bash
BoneCosmo                                        # okno z panelem
BoneCosmo presety                                # nazwy zestawów nastaw SR, SM, QM
BoneCosmo sr   --zestaw fragmentation --kroki 2000 --do runs/frag
BoneCosmo lcdm --zestaw struktury --do runs/lss
BoneCosmo sm   --zestaw plazma --kroki 400 --do runs/plazma
BoneCosmo qm   --zestaw superpozycja --kroki 80 --do runs/atom
BoneCosmo sr   --wznow --do runs/frag            # dalej z checkpointu
BoneCosmo lcdm --wznow --do runs/lss             # to samo dla ΛCDM
BoneCosmo sm   --wznow --do runs/plazma
BoneCosmo qm   --wznow --do runs/atom
BoneCosmo --pomoc
```

Bieg wsadowy zostawia w katalogu wyjściowym `checkpoint.bin` (pełny stan w `f64`,
do wznowienia), `config.json` (użyte parametry) i `frames/` z indeksem
`trajectory.json` (klatki w `f32`, do oglądania).

## Model SR: co to liczy

Zmienną stanu jest **pęd**, nie prędkość:

```
dx/dt = p c² / E,        E = √((pc)² + (mc²)²)
dp/dt = F
```

Prędkość wynika z pędu, więc `|v| < c` jest spełnione **tożsamościowo** — nie ma
obcinania prędkości ani sprawdzania warunków. Nawet dla pędu 10³⁰⁰ nie powstaje `NaN`.

Siła jest newtonowska, ze zmiękczeniem Plummera:

```
F_i = −G mᵢ Σⱼ mⱼ (xᵢ − xⱼ) / (|xᵢ − xⱼ|² + ε²)^{3/2}
```

Całkowanie: leapfrog KDK na pędzie, krok adaptacyjny `dt ≤ η√(ε/a_max)`.

### Czego to NIE jest

To model „kinematyka SR + siła Newtona", a nie ogólna teoria względności:

- grawitacja jest natychmiastowa — bez opóźnienia, fal grawitacyjnych i pędu pola;
- źródłem grawitacji jest masa spoczynkowa, nie pełny tensor energii-pędu;
- dlatego środek masy spoczynkowej powoli wędruje, mimo że `Σp = 0` jest zachowane
  dokładnie. Nie jest to usterka — dryf zgadza się z przewidywaniem kinematycznym
  co do rzędu wielkości;
- brak horyzontów zdarzeń i precesji peryhelium.

## Model ΛCDM: co to liczy

Tło z parametrów Plancka 2018, widmo mocy Eisensteina i Hu (wariant bez oscylacji
barionowych) znormalizowane przez `σ₈`, przesunięcia Zel'dovicha jako warunek
początkowy, leapfrog KDK po `ln a`.

Ograniczenie, które trzeba wypowiedzieć wprost: **brzegi są izolowane, nie
periodyczne**. Liczona jest odosobniona próbka materii w pustej przestrzeni, a nie
kawałek jednorodnego wszechświata z nieskończonym ciągiem kopii. Na brzegu próbki
brakuje przyciągania z zewnątrz, więc krawędź rusza się wolniej od środka. Za to nic
nie zawija się przez ścianę i chmura może się swobodnie zapadać.

## Kształty startowe (SR)

Dziesięć rozkładów: `ball`, `cube`, `cylinder`, `disk`, `torus`, `sphere_shell`,
`filament`, `gaussian`, `two_clumps`, `plummer`. Proporcje ustawiają dwa pokrętła:

- **`thickness`** — przekrój poprzeczny, dla kształtów, które go mają (dysk, torus,
  włókno). To on, a nie promień, wyznacza długość fali fragmentacji (λ ≈ 3,6·σ).
- **`flatten`** — mnożnik osi z, działa na **każdy** kształt. Rozmiar (`radius`) jest
  przez to oddzielony od proporcji, więc zmiana kształtu nie zmienia przy okazji skali.

![Kształty](docs/shapes.png)

Te same dziesięć kształtów po spłaszczeniu (`flatten = 0,25`) i w rzucie z góry:
[`docs/shapes_flat.png`](docs/shapes_flat.png),
[`docs/shapes_top.png`](docs/shapes_top.png).

### Dlaczego warunek startowy zadaje się wiriałem, a nie temperaturą

Ta sama dyspersja prędkości **nie jest porównywalna między kształtami**. Zmierzone
2K/|U| przy identycznej masie, promieniu, G i dyspersji:

| kształt | kostka | powłoka | walec | kula | chmura | torus | dysk | Plummer | włókno | dwie gromady |
|---|---|---|---|---|---|---|---|---|---|---|
| 2K/\|U\| | 1,33 | 1,27 | 1,21 | 1,03 | 0,90 | 0,93 | 0,88 | 0,70 | 0,63 | 0,57 |

Rozrzut jest 2,3-krotny, a granica stabilności leży w środku tego przedziału. Bieg
z ustaloną temperaturą miesza więc wpływ geometrii z wpływem tego, jak daleko od
równowagi kształt wystartował — i nie pozwala rozstrzygnąć, co spowodowało wynik.

Dlatego istnieje parametr **`virial`**: podaje się docelowe 2K/|U|, a dyspersja jest
dobierana do energii potencjalnej *tego* kształtu (bisekcja po relatywistycznej energii
kinetycznej, |U| z podpróbki liczonej dokładnie). `virial = 0` oddaje kontrolę suwakowi
temperatury.

## Dwa solwery grawitacji

| | `exact` | `mesh` |
|---|---|---|
| metoda | dokładne sumowanie par | siatka cząstek (PM) z FFT |
| koszt | `O(N²)` | `O(N + M log M)`, `M = (2·grid)³` |
| błąd siły | zero (definicja) | mierzony: 0,3–1% na gładkiej chmurze, do 15% po powstaniu zgęstek |
| precyzja | `f64` | `f64` na cząstkach, `f32` na siatce |

Zmierzone na 22 rdzeniach (wydanie `release`, milisekundy na krok):

| solver | siatka | N = 4 000 | N = 20 000 | N = 120 000 |
|---|---|---|---|---|
| `exact` | — | 9,5 | 408 | ~15 000 (ekstrapolacja) |
| `mesh` | 48 | 16 | 16 | 16 |
| `mesh` | 96 | 90 | 90 | 90 |
| `mesh` | 192 | 1 480 | 1 480 | 1 480 |

Koszt siatki nie zależy od liczby cząstek, tylko od siatki — i między siatką 64 a 192
różni się czterdziestokrotnie. Dlatego `auto` **nie** ma stałego progu w cząstkach:
porównuje `N²` z `(2·grid)³·log₂(2·grid)` i wybiera tańszy solver. Przy siatce 64
granica wypada w okolicy 8 tys. cząstek, przy 192 — ponad 40 tys.

### Jak działa `mesh` i gdzie kłamie

Pudło jest **zerowo dopełnione** do podwojonego rozmiaru, a jądro grawitacyjne liczone
metodą Hockneya. Dzięki temu brzegi są izolowane, a nie periodyczne — chmura nie
oddziałuje ze swoimi kopiami, co jest typowym błędem naiwnej implementacji na FFT.
Masa jest rozkładana schematem CIC, a jego wygładzanie odkręcane w przestrzeni
Fouriera; bez tej korekty błąd siły wynosił 5–9% zamiast 1–2%.

Dwa ograniczenia, o których warto wiedzieć:

1. **PM rozdziela grawitację tylko do rozmiaru oczka.** Jeśli poprosisz o `ε` mniejsze
   niż komórka, solver podniesie je do rozmiaru komórki i **powie o tym**. Alternatywa
   — udawać, że liczy z zamówionym `ε` — dawałaby ładniejszy komunikat i gorszą fizykę.
2. **Błąd rośnie, gdy układ wytworzy strukturę drobniejszą od oczka**, i rośnie mocno.
   Zmierzone na presecie `dissipation` (120 tys. cząstek, siatka 128, chłodzenie
   zagęszczające materię):

   | krok | błąd siły | dryf energii |
   |---|---|---|
   | 200 | 0,30% | 0,008 |
   | 400 | 10,3% | 0,000 |
   | 800 | 14,3% | 0,45 |
   | 1000 | 14,7% | 0,40 |

   Te dwie kolumny rosną razem i to nie jest zbieg okoliczności: dryf energii jest
   skutkiem błędu siły. Późna faza tego biegu **nie jest wynikiem ilościowym** — mówi
   „chłodzenie prowadzi do fragmentacji", a nie „fragmenty mają taką masę". Panel
   i bieg wsadowy pokazują wtedy podpowiedź „zagęść siatkę", więc nie da się tego
   przeoczyć. Właściwym lekarstwem byłoby dołożenie sumowania bliskiego zasięgu
   (P³M/TreePM); tego nie ma.

   Gładka chmura zachowuje się inaczej: preset `fragmentation` (120 tys. cząstek,
   pierścień, siatka 192) trzyma 0,95% błędu siły i dryf 2·10⁻⁵.

![Fragmentacja pierścienia](docs/frag_ring.png)

Ten sam pierścień policzony dokładnym `O(N²)`
([`docs/frag_ring_exact.png`](docs/frag_ring_exact.png)) daje ten sam obraz zgęstek —
to jest sprawdzenie, że fragmentacja jest fizyką, a nie artefaktem siatki. Wpływ
gęstości siatki i grubości przekroju: [`docs/frag_g192.png`](docs/frag_g192.png),
[`docs/frag_thin.png`](docs/frag_thin.png).

## Model cząstek (SM): co to liczy

Klasyczny gaz punktów o tożsamości z Modelu Standardowego. Każda cząstka ma
gatunek z tablicy PDG (masa, ładunek w trzecich `e`, spin, kolor, czas życia).
Na cząstki działają cztery oddziaływania — Coulomb i grawitacja na tym samym
solverze co modele grawitacyjne, silne i słabe pętlą po parach — a nad tym
siedzi warstwa stochastyczna: rozpady i anihilacja.

Zmienną stanu jest pęd, tak samo jak w SR, i z dodatkowego powodu: cząstka
bezmasowa (foton, gluon, neutrino w tym przybliżeniu) nie ma prędkości jako
zmiennej stanu — jej prędkość jest zawsze `c`.

### Czego to NIE jest

To nie jest kwantowa teoria pola. Brakuje stanów związanych (atom wodoru nie
istnieje — zmiękczenie `ε` jest protezą za kwantowanie), amplitud i interferencji,
hadronizacji oraz twardych zderzeń. Ładunek, liczba barionowa i liczby leptonowe
są zachowywane **dokładnie** (są liczbami całkowitymi). Siły i rozpady żyją na
skalach, które się nie spotykają: mion żyje `6,6·10¹⁷ fm/c`, a oddziaływanie
działa na `~1 fm/c`. Zestaw nastaw wybiera jedną z tych skal.

### Zestawy nastaw

| zestaw | co pokazuje |
|---|---|
| `plazma` | gaz e⁻+p, ekranowanie Coulomba |
| `para` | elektron i proton naprzeciw siebie |
| `miony` | wiązka μ⁻, rozpad z dylatacją czasu |
| `anihilacja` | chmura e⁺e⁻ → γγ |
| `piony` | π⁰ → γγ, masa spoczynkowa w ruch |
| `uwiezienie` | para uū, potencjał Cornella |

## Model atomów (QM): co to liczy

Stany związane atomu wodoropodobnego są tu **dokładnym** rozwiązaniem równania
Schrödingera, nie modelem Bohra:

```
ψ_{nlm}(r,θ,φ) = R_{nl}(r) Y_{lm}(θ,φ)
E_n = −μ Z² / (2 n²) hartree
```

Chmura na ekranie to próbka `|ψ|²` (Metropolis–Hastings), nie zbiór elektronów.
Jeden elektron w `1s` jest tysiącami punktów, bo inaczej orbitalu nie widać.
Odcień pokazuje znak funkcji falowej (dwa płaty `2p_z` mają przeciwne znaki)
albo numer powłoki w atomie wieloelektronowym.

Superpozycja stanów o różnych `n` ewoluuje fazą `e^{−iEt/ħ}`. Gęstość bije —
preset `superpozycja` (`1s + 2p_z`) pokazuje ten ruch. To nie jest klasyczna
orbita.

Atom wieloelektronowy (C, Ne, Na, Fe, …) jest przybliżeniem Slatera: każdy
elektron w wodoropodobnym orbitalu z `Z_eff = Z − σ`. Diagnostyka porównuje
energię orbitalu walencyjnego z pierwszą jonizacją NIST i **podaje błąd**.
Na helu ten błąd jest duży (~60%) i to jest wynik, nie usterka.

Panel rysuje funkcje, nie tylko chmurę: `R_{nl}(r)`, `P(r) = r²R²`, `|Y_{lm}|²`
oraz drabinę poziomów i linie Rydberga (Lyman, Balmer, Paschen). Hα wychodzi
656 nm, bo masa jądra jest skończona.

### Czego to NIE jest

To nie jest QFT, QED ani pełny atom wieloelektronowy. Brak korelacji, wymienności
Hartree–Focka, struktury subtelnej jako dynamiki, cząsteczek i wiązań. Klasyczny
model `sm` nadal nie wiąże elektronu z protonem — ten moduł odpowiada na inne
pytanie.

### Zestawy nastaw

| zestaw | co pokazuje |
|---|---|
| `wodor` | H `1s`, rozwiązanie dokładne |
| `orbital_2p` | `2p_z`: dwa płaty, węzeł na równiku |
| `orbital_3d` | `3d_z²` |
| `rydberg` | `n=8`, elektron daleko od jądra |
| `superpozycja` | `1s+2p_z`, bicie gęstości |
| `hel_plus` | He⁺, nadal jeden elektron |
| `hel` / `wegiel` / `neon` / `sod` / `zelazo` | Slater, błąd IE w tabeli |

## Diagnostyka

Silnik liczy na bieżąco energię (kinetyczną relatywistyczną i potencjalną), pęd, moment
pędu, stosunek wirialny `2T/|U|`, promień połowy masy, statystyki `γ` i `β`, oraz
**zmierzony błąd siły** — przez porównanie z dokładnym `O(N²)` na losowej próbce.

Wielkość, na którą warto patrzeć, to dryf energii. Zmierzony na presecie `precision`
(2 tys. cząstek, solver dokładny, 200 kroków): **1,3·10⁻⁶**, oscylujący wokół zera,
a nie narastający — sygnatura poprawnego całkowania, a nie tłumienia. W biegu
z chłodzeniem wielkością zachowaną jest `E + energia odprowadzona`, i to jej dryf
jest raportowany.

Dryf rzędu 10⁻⁶ dotyczy solvera dokładnego. Na siatce jest ograniczony przez błąd
siły — patrz tabela wyżej.

Dla ΛCDM analogiczną miarą jest reszta równania Layzera–Irvine'a; maleje z krokiem,
co sprawdza osobny test.

## Dobór parametrów

Dwie rzeczy, na które łatwo się nadziać:

**Masa jest masą całego układu**, nie jednej cząstki. Dlatego przesunięcie suwaka
liczby cząstek zmienia rozdzielczość, a nie badany obiekt — energia i `β_max` zostają
takie same od tysiąca do 120 tys. cząstek.

**Za duże `G` przy danym `c` czyni orbitę kołową niespełnialną.** Prędkości startowe
trafiłyby wtedy na limit, układ wystartowałby z dodatnią energią i rozleciał się. Kod
to wykrywa i ostrzega, zamiast po cichu przyciąć. Żeby dobrać `G` świadomie, jest
`sr::config::gravity_for_beta(total_mass, radius, c, beta)`.

## Struktura

```
cosmo/
  core/          bone-core — fizyka, I/O, sesja, CLI
    vec3, rng, fft, grid, mesh
    sr/          relativity, state, config, presets, spawn,
                 backends/exact, integrator, cooling, diagnostics, engine
    lcdm/        units, cosmology, power, ics, engine, presets
    sm/          particles, units, kinematics, forces, decays, spawn,
                 diagnostics, engine, presets
    qm/          units, hydrogen, elements, plot, sample, engine, presets
    io/          binary, checkpoint, trajectory
    session.rs   wspólna pętla: krok, diagnostyka, zapis
    cli.rs       bieg wsadowy
  ui/            bone-ui — kamera, renderer, panel, odtwarzacz
  app/           binarka BoneCosmo
```

Modele `sr` i `lcdm` różnią się kinematyką i warunkami początkowymi, nie sposobem
liczenia grawitacji — dlatego `mesh`, `grid`, `fft`, `vec3` i `rng` są wspólne.
`sm` bierze z tego kinematykę relatywistyczną i solver dalekozasięgowy: Coulomb
to to samo równanie co grawitacja, z innym ładunkiem. `qm` nie liczy sił —
chmura jest próbką `|ψ|²`, a czas jest fazą superpozycji.

Strona z opisem i odnośnikiem do wydania leży w `www/` (statyczna, nic nie liczy).
