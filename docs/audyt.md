# Audyt naukowy stałych

9 września 2026. Cztery modele, jeden katalog. Żaden wzór nie był zły —
problem był organizacyjny: ta sama stała żyła w dwóch plikach, a wielkości
pochodne były wpisane osobno zamiast wynikać ze źródła.

Źródła porównania: CODATA 2022, PDG 2025 (Navas i in., Phys. Rev. D 110,
030001 (2024) i aktualizacja 2025), Planck 2018 Table 2 (Aghanim i in.,
A&A 641, A6 (2020)), definicje SI 2019. Poprzednie wpisy w kodzie brały
CODATA 2018 i PDG 2024.

Katalog po audycie leży w [`cosmo/core/src/constants.rs`](../cosmo/core/src/constants.rs).
`sm` i `qm` go reeksportują. Test łapie, gdyby któryś moduł wrócił do
lokalnego wpisu.

## Werdykt

| | |
|---|---|
| Błędy modelu (zły wzór, zła fizyka) | zero |
| Stałe w `constants.rs` | 22 |
| Zaktualizowanych mas | 13 |
| Czasy życia rezonansów wyprowadzone `τ = ħ/Γ` zamiast wpisu | 4 |

Granice opisu — Newton zamiast OTW, izolowany ΛCDM, klasyczny gaz SM,
Slater zamiast Hartree–Focka — zostają wypowiedziane wprost. Audyt ich
nie rusza. Normalizacja dotyczy liczb katalogowych, nie przybliżeń dynamiki.

## Co się zmieniło

### Stałe katalogowe

| Stała | Było | Jest | Źródło |
|---|---|---|---|
| α | 7,2973525693·10⁻³ | 7,2973525643·10⁻³ | CODATA 2022 |
| E_h [eV] | 27,211386245988 | 27,211386245981 | CODATA 2022 |
| a₀ [m] | 5,29177210903·10⁻¹¹ | 5,29177210544·10⁻¹¹ | CODATA 2022 |
| ħ/E_h [s] | 2,4188843265857·10⁻¹⁷ | 2,4188843265864·10⁻¹⁷ | CODATA 2022 |
| m_p/m_e | 1836,15267343 | 1836,152673426 | CODATA 2022 |
| m_e [MeV] | 0,51099895000 | 0,51099895069 | CODATA 2022 |
| m_p [MeV] | 938,27208816 | 938,27208943 | CODATA 2022 |
| m_n [MeV] | 939,56542052 | 939,56542194 | CODATA 2022 |
| hc [eV·nm] | 1239,84193 | 1239,841984332 | SI 2019, dokładne |
| sin²θ_W | 0,23121 | 0,23122 | PDG 2025, MSbar(M_Z) |

`ħc = 197,3269804 MeV·fm` nie zmieniło się: jest dokładne z definicji `ħ` i `c`.

Wartości pochodne **nie są wpisane**. Osobny wpis rozjechałby się z resztą
tablicy przy pierwszej poprawce z CODATA:

```
COULOMB  = α · ħc
r_e      = α ħc / m_e
Rydberg  = E_h / 2
λ_W      = ħc / M_W
G        = ħc / M_Pl²          (w jednostkach cząstek)
Ω_Λ      = 1 − Ω_m − Ω_r
```

`r_e` wychodzi 2,8179403205 fm i zgadza się z CODATA 2022. Jonizacja wodoru
z zredukowaną masą nadal trafia w NIST: 13,59844 eV. Hα nadal 656,3 nm.

### Tablica cząstek

Kwarki lekkie to masy MS-bar, nie masy biegunowe — to jest w komentarzu
tablicy od początku. Zmiana jest w trzeciej cyfrze, ale jedna cyfra w dwóch
plikach to dwa źródła prawdy.

| Gatunek | Było [MeV] | Jest [MeV] | Uwaga |
|---|---|---|---|
| d | 4,67 | 4,70 | PDG 2025, MS 2 GeV |
| s | 93,4 | 93,5 | PDG 2025, MS 2 GeV |
| c | 1270 | 1273,0 | PDG 2025, MS(m) |
| b | 4180 | 4183 | PDG 2025, MS(m) |
| t | 172690 | 172560 | PDG 2025, pomiar bezpośredni |
| τ | 1776,86 | 1776,93 | PDG 2025 |
| W | 80377 | 80369,2 | PDG 2025 |
| Z | 91187,6 | 91188,0 | PDG 2025 |
| H | 125250 | 125200 | PDG 2025 |

Elektron, proton i neutron biorą masy z CODATA 2022, nie z osobnego wiersza
PDG. PDG 2025 nadal cytuje CODATA 2018 dla nukleonów; katalog idzie za
nowszym CODATA, bo to jest pomiar stałej, a nie cząstki złożonej.

Masy π, μ oraz czasy życia μ, τ, n, π są zgodne z PDG 2025 i nie wymagały
zmiany.

### Czasy życia rezonansów

W, Z, H i t nie mają mierzonego `τ` niezależnego od szerokości. Wpisanie
obu liczb pozwalało im się rozjechać. Teraz `τ = ħ/Γ`.

| Gatunek | Było | Jest | Skutek |
|---|---|---|---|
| t | 4,6·10⁻²⁵ s | ħ/Γ, Γ = 1,42 GeV | było wpisane, jest wyprowadzone |
| W | 3,157·10⁻²⁵ s | ħ/Γ, Γ = 2,14 GeV | szerokość PDG 2025 zmieniła τ |
| Z | 2,638·10⁻²⁵ s | ħ/Γ, Γ = 2,4955 GeV | ta sama liczba, inne źródło |
| H | 2,06·10⁻²² s | ħ/Γ, Γ_SM = 4,07 MeV | stare 3,2 MeV → LHCHXSWG |

Szerokość Higgsa jest przewidywaniem SM, nie pomiarem. Doświadczalna
szerokość jest rzędu kilku MeV z niepewnością większą niż sama wartość;
wpisanie pomiaru udawałoby dokładność, której nie ma.

### Kanały rozpadu τ

PDG 2025: BR(e) 17,82% → 17,85%, BR(μ) 17,39% → 17,37%. Kanał pionowy
zostaje 10,82%. Reszta (kanały wielociałowe, których model nie rozgrywa)
jest doważona do jedności: 53,96%. Suma nadal wynosi dokładnie jeden.

## Zostawione świadomie

| Parametr | Wartość | Dlaczego bez zmiany |
|---|---|---|
| ΛCDM Planck 2018 | h = 0,6736, Ω_m = 0,3153, Ω_b = 0,0493, n_s = 0,9649, σ₈ = 0,8111 | Zestaw TT,TE,EE+lowE+lensing+BAO. To katalog obserwacji, nie stała do „aktualizacji". |
| N_eff | 3,046 | Wartość, przy której Planck wyciągał parametry (Mangano i in.). Nowsze rachunki dają 3,044; podstawienie nowszej liczby do parametrów wyciągniętych przy starej psułoby spójność zestawu. |
| T_CMB | 2,7255 K | Fixsen 2009, ta sama liczba, której używał Planck. Teraz z `constants.rs`. |
| G (ΛCDM) | 43,0071 | Jednostki GADGET: Mpc/h, 10¹⁰ M☉/h, km/s. To konwencja układu, nie Newtonowskie G z SI. |
| σ, α_s | 0,18 GeV², 0,30 | Potencjał Cornella, skala ~1 GeV. Fenomenologia, nie stała katalogowa. |
| Neutrina | m = 0 | Znane są różnice kwadratów i górne ograniczenie, nie masa. Zero jest jawnym przybliżeniem. |

## Granice modeli — to nie jest usterka

Każdy moduł już wypowiada, czym nie jest. Audyt to potwierdza, nie poprawia.

- **SR** — kinematyka szczególnej teorii względności plus siła Newtona.
  Grawitacja jest natychmiastowa, źródłem jest masa spoczynkowa. Brak
  horyzontów i precesji peryhelium.
- **ΛCDM** — brzegi izolowane (Hockney), nie periodyczne. Odosobniona
  próbka materii, nie kawałek jednorodnego wszechświata z kopiami.
- **SM** — klasyczny gaz punktów o tożsamości z Modelem Standardowym.
  Nie QFT: brak stanów związanych, amplitud, hadronizacji i twardych
  zderzeń. `ε` jest protezą za kwantowanie.
- **QM** — wodór i jony wodoropodobne są dokładnym Schrödingera. Atom
  wieloelektronowy to Slater; diagnostyka **mierzy** błąd IE wobec NIST
  i go pokazuje. Na helu ten błąd jest duży (~60%) i to jest wynik.

## Skąd co pochodzi po normalizacji

```
SI 2019          c, h, e, ħ           dokładne z definicji
CODATA 2022      α, E_h, a₀, m_e, m_p, m_n, m_p/m_e
PDG 2025         masy i Γ rezonansów, sin²θ_W, kanały τ
Planck 2018      h, Ω_m, Ω_b, n_s, σ₈, T_CMB
wyprowadzone     COULOMB, r_e, Rydberg, λ_W, τ_W/Z/H/t, Ω_Λ
fenomenologia    α_s, napięcie struny
```

Jedna poprawka z następnym CODATA rusza wszystkie pochodne naraz.
To był cały punkt audytu.
