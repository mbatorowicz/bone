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
Kerr i PDE na siatce zostają później.

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
