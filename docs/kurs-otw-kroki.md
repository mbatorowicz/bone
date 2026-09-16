# Kurs OTW — kroki na jedno okno czatu

Jeden krok = jeden chat = jeden commit (potem `git push`). Nie łącz kroków. W nowym oknie wklej tylko **Prompt** z karty.

Źródło architektury: plan *Kurs OTW rdzeń*. Ten plik jest listą wykonawczą.

Reguły:

- nie czytaj całego repo od zera — punkt startu jest w karcie,
- nie ruszaj plików spoza listy, chyba że kompilacja tego wymaga,
- jeśli krok puchnie, dokończ go i zatrzymaj się,
- UI: weryfikacja w oknie BoneCosmo, nie na URL gminy,
- `cargo test` zielone przed commitem.

---

## Krok 1 — Powłoka akademii, stare laby żyją

**Cel:** Aplikacja otwiera się na mapie kursu. Wejście w N-ciała / kosmologię / cząstki / atomy działa jak dziś.

**Zależności:** brak.

**Start:** `cosmo/ui/src/lib.rs`, `cosmo/ui/src/panels.rs`, `cosmo/ui/src/simulation.rs`.

**Zrób:** enum `Screen { Map, Lesson(LessonId), Lab(LabId) }`. Mapa: trzy ścieżki (karty-stuby), na dole cztery laboratoria. `Lesson` — placeholder z wstecz. `Lab` reuse `panels` + `draw_cloud`. Tytuł: „Bone — czasoprzestrzeń”. Bez nowej fizyki.

**Gotowe gdy:** `cargo test --workspace` zielone; z mapy da się otworzyć każde z 4 laboratoriów i wrócić.

**Prompt:**

```
Wykonaj krok 1 z docs/kurs-otw-kroki.md. Tylko powłoka UI: Screen::Map | Lesson | Lab. Mapa kursu na starcie, stare 4 laboratoria działają z mapy bez zmiany fizyki. Nie pisz lekcji ani modułu gr.
```

---

## Krok 2 — Ekran lekcji, parser Markdown, stuby

**Cel:** Każda lekcja rdzenia ma ID, tytuł, stub i układ 60/40. Placeholder-animacja z play/pauza.

**Zależności:** krok 1.

**Start:** `cosmo/ui/src/lesson.rs`, katalog `cosmo/lessons/`.

**Zrób:** parser nagłówków / akapitów / `> callout` / blok kodu do egui. Stuby: `stw/01`–`07`, `geo/01`–`06`, `bh/01`–`04`. Mapa otwiera lekcję. Wstecz/Dalej. Pętla `request_repaint`.

**Gotowe gdy:** przejście A1→A7→mapa; parser nie wywala UI.

**Prompt:**

```
Wykonaj krok 2 z docs/kurs-otw-kroki.md. Ekran lekcji 60/40, parser Markdown, stuby wszystkich lekcji STW/geo/BH, placeholder-animacja. Bez prawdziwej fizyki GR i bez pełnych tekstów.
```

---

## Krok 3 — Teksty STW dla laika

**Cel:** Ścieżka A kompletna tekstowo (analogia → obraz → wzór → „w kodzie”).

**Zależności:** krok 2.

**Pliki:** `cosmo/lessons/stw/*.md` (ew. drobny parser).

**Zrób:** lekcje 1–7: błyskawica/jednoczesność; stożek świetlny; Lorentz jako przechylanie osi; dylatacja; kontrakcja; 4-wektor jako wydarzenie; zapowiedź N-ciał. Po polsku, dla laika.

**Gotowe gdy:** każda lekcja STW to kompletna strona w UI.

**Prompt:**

```
Wykonaj krok 3 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki STW w cosmo/lessons/stw/. Nie zmieniaj silnika ani wizualizacji.
```

---

## Krok 4 — `gr::lorentz`

**Cel:** Liczby STW w `bone-core`, testowane, bez UI.

**Zależności:** workspace kompiluje się (krok 1).

**Pliki:** `cosmo/core/src/gr/mod.rs`, `lorentz.rs`; `pub mod gr` w `cosmo/core/src/lib.rs`.

**Zrób:** boost 1D/3D, γ, dylatacja, kontrakcja, interwał Minkowskiego, 4-wektor zdarzenia. Testy: złożenie boostów, interwał niezmienniczy, |v|<c. Nie mieszaj z `sr/relativity.rs`.

**Gotowe gdy:** testy `gr`/`lorentz` przechodzą, clippy czysty.

**Prompt:**

```
Wykonaj krok 4 z docs/kurs-otw-kroki.md. Nowy moduł bone-core gr::lorentz z testami. Bez UI, bez Schwarzschilda, bez RK4.
```

---

## Krok 5 — Animacje STW: Minkowski, błyskawica, boost

**Cel:** Lekcje 1–3 mają ruchomy obraz (suwak β / czas).

**Zależności:** kroki 2 i 4.

**Pliki:** `cosmo/ui/src/viz/mod.rs`, `viz/minkowski.rs`.

**Zrób:** diagram ct–x, światło 45°, światolinie, suwak β przechyla osie. Lekcja 1: błyskawica i dwaj obserwatorzy. Liczby z `gr::lorentz`.

**Gotowe gdy:** suwak widać od razu, γ zgodne z testami.

**Prompt:**

```
Wykonaj krok 5 z docs/kurs-otw-kroki.md. Animacje lekcji STW 1–3 (błyskawica, Minkowski, boost). Liczby z gr::lorentz. Nie ruszaj zegarów ani N-ciał.
```

---

## Krok 6 — Zegary, kontrakcja, 4-wektor + drzwi do N-ciał

**Cel:** Lekcje 4–6 animowane; lekcja 7 otwiera laboratorium N-ciała.

**Zależności:** krok 5.

**Pliki:** `cosmo/ui/src/viz/clocks.rs`; mapowanie lekcji 7 → `LabId::Nbody`.

**Zrób:** dwa zegary, kurcząca się linijka, punkt `(ct,x)`. Z lekcji 7: preset N-ciał, Newton/SR, karta „siła nadal newtonowska — to nie OTW”.

**Gotowe gdy:** ścieżka A → chmura N-ciał → powrót na mapę.

**Prompt:**

```
Wykonaj krok 6 z docs/kurs-otw-kroki.md. Animacje lekcji STW 4–6 i wejście z lekcji 7 do laboratorium N-ciała. Bez geodezyjnej.
```

---

## Krok 7 — `gr::rk4` + `gr::metric`

**Cel:** Stepper RK4 i metryka Schwarzschilda jako liczby, bez geodezyjnej.

**Zależności:** krok 4.

**Pliki:** `cosmo/core/src/gr/rk4.rs`, `metric.rs`.

**Zrób:** RK4 na wektorze stanu. Schwarzschild `G=c=1`: `g_μν`, `r_h=2M`, sfera fotonowa `3M`, ISCO `6M`. Testy: `y'=y` vs `exp`; wartości `g_tt`, `g_rr`.

**Gotowe gdy:** testy przechodzą, UI nietknięte.

**Prompt:**

```
Wykonaj krok 7 z docs/kurs-otw-kroki.md. Tylko gr::rk4 i gr::metric z testami. Bez Γ, bez geodezyjnej, bez UI.
```

---

## Krok 8 — Christoffel + geodezyjna

**Cel:** Ruch w Schwarzschildu scałkowany; błąd E i L zmierzony.

**Zależności:** krok 7.

**Pliki:** `cosmo/core/src/gr/christoffel.rs`, `geodesic.rs`.

**Zrób:** analityczne Γ Schwarzschilda. Stan `(t,r,θ,φ)` + pochodne po λ. Testy: M→0 → prosta; zachowanie E, L; orbita kołowa `6M`; sfera fotonowa `3M`.

**Gotowe gdy:** testy z jawną tolerancją.

**Prompt:**

```
Wykonaj krok 8 z docs/kurs-otw-kroki.md. gr::christoffel i gr::geodesic z testami E/L i orbit. Bez raytracera i bez UI.
```

---

## Krok 9 — Teksty geodezyjna

**Cel:** Ścieżka B kompletna tekstowo.

**Zależności:** krok 2.

**Pliki:** `cosmo/lessons/geo/*.md`.

**Zrób:** lekcje 1–6: kula; metryka-linijka; Schwarzschild słowami; geodezyjna; RK4; zapowiedź labu. Laik, jeden wzór na lekcję.

**Gotowe gdy:** teksty renderują się w ekranie lekcji.

**Prompt:**

```
Wykonaj krok 9 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki geodezyjna. Bez nowych wizualizacji.
```

---

## Krok 10 — Animacje geodezyjna (kula, siatka, film RK4)

**Cel:** Lekcje B1–B5 mają ruchomy obraz.

**Zależności:** kroki 8 i 9.

**Pliki:** `cosmo/ui/src/viz/sphere.rs`, `viz/metric_grid.rs`, `viz/rk4_film.rs`.

**Zrób:** meridian vs „idź prosto”; gumowa siatka (suwak M); Euler vs RK4 z błędem na ekranie.

**Gotowe gdy:** suwaki i play/pauza działają; labu zrzucania jeszcze nie ma.

**Prompt:**

```
Wykonaj krok 10 z docs/kurs-otw-kroki.md. Animacje lekcji geodezyjna 1–5 (kula, siatka, RK4 vs Euler). Laboratorium zrzucania zostaw na krok 11.
```

---

## Krok 11 — Laboratorium geodezyjnych

**Cel:** Zrzucasz foton albo cząstkę: orbita / wychwyt / ucieczka w równiku.

**Zależności:** krok 8, powłoka labu (krok 1).

**Pliki:** `LabId::Geodesics`; `cosmo/ui/src/viz/geodesics.rs`.

**Zrób:** widok równikowy, horyzont, kilka krzywych, suwaki parametru zderzenia / pędu, null vs timelike. Wejście z lekcji B6 i z mapy.

**Gotowe gdy:** mały parametr → wychwyt, duży → ucieczka; okolice `3M`/`6M` zgodne z testami.

**Prompt:**

```
Wykonaj krok 11 z docs/kurs-otw-kroki.md. Laboratorium zrzucania geodezyjnych w równiku Schwarzschilda. Bez raytracera pikseli.
```

---

## Krok 12 — `gr::raytrace` (silnik, bez UI)

**Cel:** Funkcja obrazu zwraca bufor pikseli; testy bez okna.

**Zależności:** krok 8.

**Pliki:** `cosmo/core/src/gr/raytrace.rs`.

**Zrób:** kamera, piksel = geodezyjna zerowa wstecz; horyzont / dysk / ucieczka. ~320×180 CPU. Testy na małym obrazie (np. 32×18): centrum czarne, duży parametr ≠ horyzont.

**Gotowe gdy:** test jednostkowy kończy się w rozsądnym czasie.

**Prompt:**

```
Wykonaj krok 12 z docs/kurs-otw-kroki.md. Tylko gr::raytrace + testy. Bez podpinania do okna, bez wątku tła.
```

---

## Krok 13 — Teksty BH + animacje 2D

**Cel:** Ścieżka C uczona ruchem, zanim wejdzie ciężki raytracer.

**Zależności:** kroki 2, 8, 11.

**Pliki:** `cosmo/lessons/bh/*.md`; `cosmo/ui/src/viz/rings.rs`.

**Zrób:** lekcje C1–C4. Animowane pierścienie `2M`/`3M`/`6M`, pęk promieni, Einstein ring w 2D.

**Gotowe gdy:** ścieżka C czytelna i ruchoma bez klatki 320×180.

**Prompt:**

```
Wykonaj krok 13 z docs/kurs-otw-kroki.md. Teksty czarnej dziury i animacje 2D pierścieni/soczewkowania. Laboratorium raytracera zostaw na krok 14.
```

---

## Krok 14 — Laboratorium raytracera w oknie

**Cel:** Obraz dysku + soczewkowanie; UI nie zamarza.

**Zależności:** krok 12.

**Pliki:** `LabId::BlackHole`; `cosmo/ui/src/viz/blackhole.rs`.

**Zrób:** `ColorImage`, overlay `2M`/`3M`/`6M`, suwaki masy / nachylenia / odległości. Spin = 0. Liczenie w tle. Wejście z C3/C4 i z mapy.

**Gotowe gdy:** suwak pokazuje „liczę…” i wstawia obraz; okno reaguje w trakcie.

**Prompt:**

```
Wykonaj krok 14 z docs/kurs-otw-kroki.md. Laboratorium raytracera Schwarzschilda w UI: tło, suwaki, overlay 2M/3M/6M. Nie dodawaj Kerra ani PINN.
```

---

## Krok 15 — Dokumentacja, siatka, push

**Cel:** README i landing mówią: kurs STW → geodezyjna → czarna dziura; `sr` to nadal nie OTW.

**Zależności:** kroki 1–14.

**Pliki:** `README.md`, `cosmo/README.md`, `www/index.html`.

**Zrób:** `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`. Ręcznie: A→B→C, N-ciała z lekcji 7, raytracer, trzy pozostałe laby z mapy. Commit + push. Bez force push.

**Gotowe gdy:** testy i clippy czyste.

**Prompt:**

```
Wykonaj krok 15 z docs/kurs-otw-kroki.md. README + landing, pełne testy i clippy, commit i push. Bez nowej fizyki.
```

---

Rdzeń (kroki 1–15) jest na mapie. Fala 2: tensory → równania Einsteina → PINN.
Fala 3: animacje ten/ein/pinn. Fala 4: Kerr i siatka PDE (kroki 26–36).

## Krok 16 — Powłoka: tensory, Einstein, PINN

**Cel:** Na mapie widać trzy następne ścieżki. Wejście w STW / geo / BH i laboratoria działa jak dziś.

**Zależności:** krok 15.

**Start:** `cosmo/ui/src/screen.rs`, `cosmo/ui/src/lesson.rs`, katalog `cosmo/lessons/`.

**Zrób:** `Track::{Tensor, Einstein, Pinn}` obok rdzenia. Stuby: `ten/01`–`05`, `ein/01`–`04`, `pinn/01`–`04`. Mapa: sekcja „Następne działy”. Lekcja — istniejący 60/40 i placeholder-animacja. Ostatnia lekcja wraca na mapę, bez labu. Bez algebry tensorów w `gr`, bez PINN, bez Kerra.

**Gotowe gdy:** `cargo test --workspace` zielone; z mapy da się otworzyć każdą nową lekcję i wrócić; rdzeń nietknięty.

**Prompt:**

```
Wykonaj krok 16 z docs/kurs-otw-kroki.md. Powłoka trzech ścieżek: tensory, Einstein, PINN. Stuby i placeholder. Bez nowej fizyki, bez Kerra, bez sieci.
```

## Krok 17 — Teksty tensory dla laika

**Cel:** Ścieżka `ten` kompletna tekstowo.

**Pliki:** `cosmo/lessons/ten/*.md`.

**Zrób:** lekcje 1–5: liczba vs wektor; macierz jako maszyna; tensor; metryka którą już znasz; wskaźniki w górę i w dół. Laik, analogia → obraz → wzór → kod.

**Prompt:**

```
Wykonaj krok 17 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki tensory. Bez silnika i bez nowych animacji.
```

## Krok 18 — `gr::tensor`

**Cel:** Algebra 4D w `bone-core`, testowana, bez UI.

**Pliki:** `cosmo/core/src/gr/tensor.rs`.

**Zrób:** wektor, kowektor, tensor (1,1) i (0,2), skurcz, `η_μν`, podnoszenie/opuszczanie Minkowskiego. Testy: skurcz, ηη⁻¹ = 1, |v|<c nie tu. Bez Riemann, bez Einstein.

**Prompt:**

```
Wykonaj krok 18 z docs/kurs-otw-kroki.md. Tylko gr::tensor z testami. Bez UI, bez równań Einsteina, bez PINN.
```

## Krok 19 — Teksty Einstein dla laika

**Cel:** Ścieżka `ein` kompletna tekstowo.

**Pliki:** `cosmo/lessons/ein/*.md`.

**Zrób:** lekcje 1–4: masa zgina przestrzeń; lewa strona (krzywizna); prawa strona (energia); próżnia i Schwarzschild. Laik, analogia → obraz → wzór → kod. Bez Kerra, bez PDE na siatce, bez nowych animacji.

**Prompt:**

```
Wykonaj krok 19 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki Einstein. Bez silnika i bez nowych animacji.
```

## Krok 20 — `gr::einstein`

**Cel:** Krzywizna i równanie pola na Schwarzschildu, testowane, bez UI.

**Pliki:** `cosmo/core/src/gr/einstein.rs`.

**Zrób:** Riemann z Γ, Ricci, skalar, `G_μν`, `T_μν` (próżnia i pył). Testy: Minkowski Riemann = 0; Kretschmann `48 M²/r⁶`; `G_μν = 0` poza horyzontem; `G_μν = 8π T_μν` w próżni. Bez Kerra, bez PDE na siatce, bez PINN.

**Prompt:**

```
Wykonaj krok 20 z docs/kurs-otw-kroki.md. Tylko gr::einstein z testami. Bez UI, bez Kerra, bez sieci.
```

## Krok 21 — Teksty PINN dla laika

**Cel:** Ścieżka `pinn` kompletna tekstowo.

**Pliki:** `cosmo/lessons/pinn/*.md`.

**Zrób:** lekcje 1–4: sieć zgaduje funkcję; residual nie etykieta; ciepło i fala; dlaczego Einstein jest drogi. Laik, analogia → obraz → wzór → kod. Bez PyTorcha w tekście jako zależności, bez Kerra.

**Prompt:**

```
Wykonaj krok 21 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki PINN. Bez silnika i bez nowych animacji.
```

## Krok 22 — `gr::pinn`

**Cel:** Residual PDE i mała sieć w `bone-core`, testowana, bez UI.

**Pliki:** `cosmo/core/src/gr/pinn.rs`.

**Zrób:** residual ciepła `u_t − k u_xx` i fali `u_tt − c² u_xx`; maleńka sieć 2→N→1; kilka kroków spadku. Testy: residual analitycznego ciepła ≈ 0; spadek błędu po kroku. Bez PyTorcha, bez Kerra, bez równań pola na siatce.

**Prompt:**

```
Wykonaj krok 22 z docs/kurs-otw-kroki.md. Tylko gr::pinn z testami. Bez UI, bez Kerra, bez CUDA.
```

## Krok 23 — Animacje tensory

**Cel:** Ścieżka `ten` ma ruchomy obraz; liczby z `gr::tensor`.

**Zależności:** kroki 17 i 18.

**Pliki:** `cosmo/ui/src/viz/tensors.rs`; podpięcie w `viz/mod.rs`; `cosmo/lessons/ten/*.md` (akapit „Co widać”).

**Zrób:** lekcje 1–5. Kartka i strzałka (skalar vs wektor); maszyna (1,1); skrzynka η(u,v); g_tt / g_rr vs η; podnieś / opuść. Suwak kąta, play/pauza. Bez Einsteina, bez PINN, bez Kerra.

**Gotowe gdy:** suwak widać od razu; długość strzałki i ηη⁻¹ zgadzają się z testami; mapa nie pisze „stub”.

**Prompt:**

```
Wykonaj krok 23 z docs/kurs-otw-kroki.md. Animacje lekcji tensory 1–5. Liczby z gr::tensor. Bez Einsteina na ekranie i bez PINN.
```

## Krok 24 — Animacje Einstein

**Cel:** Ścieżka `ein` ma ruchomy obraz; liczby z `gr::einstein`.

**Zależności:** kroki 19, 20 i 23.

**Pliki:** `cosmo/ui/src/viz/einstein.rs`; `cosmo/lessons/ein/*.md` (akapit „Co widać”).

**Zrób:** lekcje 1–4. Trampolina z suwakiem M; pętla / Kretschmann; szalki G vs 8πT (próżnia / pył); łąka 2M/3M/6M z G = 0. Bez Kerra, bez PDE na siatce, bez PINN.

**Gotowe gdy:** Kretschmann i G na łące zgadzają się z testami silnika.

**Prompt:**

```
Wykonaj krok 24 z docs/kurs-otw-kroki.md. Animacje lekcji Einstein 1–4. Liczby z gr::einstein. Bez Kerra i bez sieci.
```

## Krok 25 — Animacje PINN, mapa, siatka

**Cel:** Ścieżka `pinn` ma ruchomy obraz; mapa nie kłamie, że to stub.

**Zależności:** kroki 21, 22 i 24.

**Pliki:** `cosmo/ui/src/viz/pinn.rs`; `cosmo/ui/src/screen.rs`; `cosmo/lessons/pinn/*.md`; README jeśli kłamie.

**Zrób:** lekcje 1–4. Sieć 2→8→1; termometr residualu; ciepło i fala z analitycznego wzoru; półka Schwarzschilda vs garnek. `cargo test --workspace`, clippy. Commit + push. Bez Kerra, bez CUDA, bez zgadywania metryki.

**Gotowe gdy:** residual analitycznego ciepła ≈ 0 na ekranie; następne działy nie piszą „stub”.

**Prompt:**

```
Wykonaj krok 25 z docs/kurs-otw-kroki.md. Animacje PINN, podpis mapy, testy i clippy, commit i push. Bez Kerra i bez CUDA.
```

---

Fala 3 (kroki 23–25) jest na mapie. Fala 4: Kerr → siatka PDE. Zderzenia, CUDA i PINN na `g_μν` zostają później.

## Krok 26 — Powłoka: Kerr, siatka PDE

**Cel:** Na mapie widać dwie kolejne ścieżki. Wejście w STW / geo / BH / ten / ein / pinn i laboratoria działa jak dziś.

**Zależności:** krok 25.

**Start:** `cosmo/ui/src/screen.rs`, `cosmo/ui/src/lesson.rs`, katalog `cosmo/lessons/`.

**Zrób:** `Track::{Kerr, Pde}` jako `Track::LATER`. Slugi `kerr` / `pde`. Stuby: `kerr/01`–`04`, `pde/01`–`04`. Mapa: sekcja „Obrót i siatka”. Lekcja — istniejący 60/40 i placeholder-animacja. Ostatnia lekcja Kerr na razie wraca na mapę (drzwi do labu w kroku 31). Ostatnia lekcja PDE wraca na mapę, bez labu. Bez `gr::kerr`, bez `gr::fd`, bez suwaka `a`.

**Gotowe gdy:** `cargo test --workspace` zielone; z mapy da się otworzyć każdą nową lekcję i wrócić; raytracer nadal `spin = 0`.

**Prompt:**

```
Wykonaj krok 26 z docs/kurs-otw-kroki.md. Powłoka dwóch ścieżek: Kerr i siatka PDE. Stuby i placeholder. Bez gr::kerr, bez suwaka a, bez różniczek skończonych.
```

## Krok 27 — Teksty Kerr dla laika

**Cel:** Ścieżka `kerr` kompletna tekstowo.

**Zależności:** krok 26.

**Pliki:** `cosmo/lessons/kerr/*.md`.

**Zrób:** lekcje 1–4: wleczenie układu (karuzela, `g_tφ`, `a = 0` to stara mata); ergosphera vs horyzont `r+`; pierścienie pękają (`r±`, foton±, ISCO±); cień nie na środku. Laik, analogia → obraz → wzór → kod. Bez Kerra w silniku, bez nowych animacji, bez Cauchy horizon.

**Prompt:**

```
Wykonaj krok 27 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki Kerr. Bez silnika i bez nowych animacji.
```

## Krok 28 — `gr::kerr` metryka

**Cel:** Boyer-Lindquist jako liczby w `bone-core`, testowane, bez UI.

**Zależności:** krok 7 (`Schwarzschild` do testu `a = 0`).

**Pliki:** `cosmo/core/src/gr/kerr.rs`; `pub mod kerr` w `gr/mod.rs`.

**Zrób:** `Kerr { mass, spin }` z `|a| ≤ M`. `Δ`, `Σ`, `g_μν` (w tym `g_tφ`), `r+`, ergo(`θ`), ISCO±, foton±. `|a| > M` to błąd. Test: `a = 0` zgadza się z `Schwarzschild` (`g_tφ = 0`, `r+ = 2M`, ergo = `2M`). Bez Γ, bez geodezyjnej, bez raytracera.

**Prompt:**

```
Wykonaj krok 28 z docs/kurs-otw-kroki.md. Tylko gr::kerr metryka z testami. Bez Γ, bez UI, bez raytracera.
```

## Krok 29 — Γ + geodezyjna Kerra

**Cel:** Ruch w Kerrze da się scałkować; `a = 0` wraca do znanego koła `6M`.

**Zależności:** krok 28.

**Pliki:** `cosmo/core/src/gr/kerr.rs` (Γ i `rhs` w tym samym module).

**Zrób:** analityczne Γ Kerra — nowy typ, nie `Christoffel` Schwarzschilda (`accel` nie zna `g_tφ`). Ten sam stan 8 liczb i `gr::rk4`. Zachowanie `E` i `L_z`. Test: `a = 0` koło `6M` jak w `geodesic`; `a ≠ 0` ISCO współ < `6M`. Bez traitu `Spacetime` na Schwarzschildu. Bez raytracera.

**Prompt:**

```
Wykonaj krok 29 z docs/kurs-otw-kroki.md. Γ i geodezyjna Kerra z testami E/L i a = 0 → 6M. Bez raytracera i bez UI.
```

## Krok 30 — Raytrace ze spinem

**Cel:** Ten sam bufor pikseli, horyzont `r+`; `a = 0` nie psuje starych testów.

**Zależności:** krok 29.

**Pliki:** `cosmo/core/src/gr/raytrace.rs`.

**Zrób:** `Config.spin` (domyślnie 0). Przy zerze istniejące testy 32×18 zostają. Przy `a ≠ 0` geodezyjna z `gr::kerr`, horyzont `r+` nie `2M`. Test: `a = 0.9` cień niesymetryczny (losy lewo/prawo się różnią). Bez okna, bez drugiego labu.

**Prompt:**

```
Wykonaj krok 30 z docs/kurs-otw-kroki.md. Config.spin w gr::raytrace z testami a = 0 i cienia. Bez UI.
```

## Krok 31 — Animacje Kerr

**Cel:** Ścieżka `kerr` ma ruchomy obraz; liczby z `gr::kerr`.

**Zależności:** kroki 27 i 29.

**Pliki:** `cosmo/ui/src/viz/kerr.rs`; podpięcie w `viz/mod.rs`; `cosmo/lessons/kerr/*.md` (akapit „Co widać”).

**Zrób:** lekcje 1–4. Wleczenie (strzałki `g_tφ`); ergo vs `r+`; pęk pierścieni foton± / ISCO±; zapowiedź cienia. Suwak `a/M`, play/pauza. Lekcja 4: drzwi do `LabId::BlackHole`. Bez PDE, bez CUDA.

**Gotowe gdy:** suwak `a` widać od razu; `a = 0` wraca do kółek `2M` / `3M` / `6M`.

**Prompt:**

```
Wykonaj krok 31 z docs/kurs-otw-kroki.md. Animacje lekcji Kerr 1–4. Liczby z gr::kerr. Bez suwaka a w laboratorium raytracera.
```

## Krok 32 — Suwak `a` w raytracerze

**Cel:** Jeden stół raytracera; ścieżka C nadal startuje z `a = 0`.

**Zależności:** kroki 30 i 31.

**Pliki:** `cosmo/ui/src/viz/blackhole.rs`; `cosmo/ui/src/screen.rs` (podpis labu).

**Zrób:** suwak `a/M` ∈ [0, 0.998]. Tło 320×180 jak dziś. Overlay `r+` / ergo / ISCO±. Wejście z BH: `spin = 0`. Wejście z Kerr 4: wartość z lekcji. Test `spin = 0` zostaje prawdziwy dla ścieżki C. Bez drugiego `LabId`, bez zderzeń.

**Gotowe gdy:** zmiana `a` pokazuje „liczę…” i nowy cień; okno reaguje w trakcie.

**Prompt:**

```
Wykonaj krok 32 z docs/kurs-otw-kroki.md. Suwak a w istniejącym laboratorium raytracera. Ścieżka C startuje z zerem. Bez nowego LabId i bez PDE.
```

## Krok 33 — Teksty siatka PDE dla laika

**Cel:** Ścieżka `pde` kompletna tekstowo.

**Zależności:** krok 26.

**Pliki:** `cosmo/lessons/pde/*.md`.

**Zrób:** lekcje 1–4: węzły zamiast suwaków (ten sam residual co PINN, inna pamięć); ciepło 1D (FTCS, CFL); fala 1D (leapfrog); dlaczego Einstein na siatce to miliony węzłów × 10 kratek. Laik, analogia → obraz → wzór → kod. Bez `gr::fd`, bez CUDA, bez zgadywania `g_μν`. Ostatnia lekcja wraca na mapę.

**Prompt:**

```
Wykonaj krok 33 z docs/kurs-otw-kroki.md. Tylko pełne teksty Markdown ścieżki siatka PDE. Bez silnika i bez nowych animacji.
```

## Krok 34 — `gr::fd`

**Cel:** Residual PDE na węzłach 1D, testowany, bez UI.

**Zależności:** krok 22 (`heat_exact` / `wave_exact` do porównania).

**Pliki:** `cosmo/core/src/gr/fd.rs`.

**Zrób:** siatka 1D, krok ciepła (FTCS) i fali (leapfrog). Te same wzory analityczne co `gr::pinn`. Test: residual analitycznego `u` → 0 przy zagęszczaniu; CFL pilnowane. Nie ruszać `cosmo/core/src/grid.rs` (CIC N-ciał). Bez PyTorcha, bez `g_μν`, bez 2D.

**Prompt:**

```
Wykonaj krok 34 z docs/kurs-otw-kroki.md. Tylko gr::fd z testami. Bez UI, bez CUDA, bez ruszania grid.rs N-ciał.
```

## Krok 35 — Animacje siatka PDE

**Cel:** Ścieżka `pde` ma ruchomy obraz; liczby z `gr::fd`.

**Zależności:** kroki 33 i 34.

**Pliki:** `cosmo/ui/src/viz/fd.rs`; podpięcie w `viz/mod.rs`; `cosmo/lessons/pde/*.md` (akapit „Co widać”).

**Zrób:** lekcje 1–4. Węzły vs wzmacniacz PINN; garnek FTCS; struna leapfrog; półka Schwarzschilda vs siatka (strzałka przekreślona). Play/pauza. Bez labu. Bez zgadywania metryki.

**Gotowe gdy:** residual analitycznego ciepła na węzłach ≈ 0 na ekranie.

**Prompt:**

```
Wykonaj krok 35 z docs/kurs-otw-kroki.md. Animacje lekcji siatka PDE 1–4. Liczby z gr::fd. Bez CUDA i bez g_μν.
```

## Krok 36 — Mapa, README, siatka

**Cel:** Mapa nie pisze „Bez Kerra”; README zapowiada falę, nie obiecuje zderzeń.

**Zależności:** kroki 26–35.

**Pliki:** `cosmo/ui/src/screen.rs`; `README.md`; `cosmo/README.md`; ewentualnie `www/index.html`.

**Zrób:** podpisy sekcji „Obrót i siatka” bez stubu. `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`. Commit + push. Bez force push. Bez zderzeń, bez CUDA, bez PINN na metryce.

**Gotowe gdy:** testy i clippy czyste; z mapy Kerr 4 otwiera raytracer z `a`, PDE 4 wraca na mapę.

**Prompt:**

```
Wykonaj krok 36 z docs/kurs-otw-kroki.md. Podpisy mapy i README, pełne testy i clippy, commit i push. Bez zderzeń i bez CUDA.
```

Zderzenia czarnych dziur, CUDA/MPI i PINN na `g_μν` zostają później.
