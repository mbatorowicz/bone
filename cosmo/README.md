# Bone Cosmo

Kurs w oknie: szczególna teoria względności → geodezyjna → czarna dziura.
Potem laboratoria. N-ciała (`sr`) to Newton + kinematyka SR. Nie OTW —
metryka i geodezyjna żyją w `gr`, nie w sile chmury.

- **STW** (`stw/01`–`07`) — Minkowski, Lorentz, zegary; lekcja 7 otwiera N-ciała.
- **Geodezyjna** (`geo/01`–`06`) — Schwarzschild, RK4; lekcja 6 otwiera zrzucanie.
- **Czarna dziura** (`bh/01`–`04`) — pierścienie `2M` / `3M` / `6M`; raytracer
  dysku, spin = 0, liczenie w tle.
- **Następne działy** — tensory, równania Einsteina, PINN: pełny tekst,
  silnik `gr` i animacja. Ostatnia lekcja wraca na mapę.
- **Kerr** (`kerr/01`–`04`) — suwak `a/M`, lekcja 4 otwiera raytracer.
- **Siatka PDE** (`pde/01`–`04`) — ciepło i fala 1D na węzłach, nie metryka.
  Lekcja 4 wraca na mapę.

Następna fala: PINN na metryce (`A(r)`) i kafelki (kroki 37–44).
Bez zderzeń czarnych dziur i bez crate’ów CUDA/MPI.

Cztery chmury zostają:

- **Kosmologia** (`lcdm`) — ΛCDM, Planck 2018, Particle-Mesh z izolowanymi
  brzegami (Hockney), warunki Zel'dovicha, całkowanie po `ln a`.
- **N-ciała** (`sr`) — Newton + kinematyka SR, odosobniona chmura,
  opcjonalna dyssypacja. Nie OTW.
- **Cząstki** (`sm`) — klasyczny gaz + rozpady PDG. To nie jest QFT.
- **Atomy** (`qm`) — Schrödinger · `|ψ|²`. Wodór dokładny, hel wariacyjny
  (~2%), Slater jako lekcja ekranowania.

Liczy na GPU (wgpu/Vulkan: siły O(N²), FFT 2ᵏ, raytracer Schwarzschilda).
Bez karty albo bez f64 wraca na CPU. Nie używa Vercela.

```
cargo build --release          # target/release/BoneCosmo.exe
cargo test --workspace
```

Binarka: `target/release/BoneCosmo.exe`

Panel zapisuje checkpoint i trajektorię, jeśli włączysz nagrywanie. Wznowienie
i odtwarzanie klatek działają z tego samego katalogu (`runs/latest` domyślnie).

```
BoneCosmo lcdm --zestaw struktury --do runs/lss
BoneCosmo sm   --zestaw plazma --do runs/plazma
BoneCosmo qm   --zestaw superpozycja --do runs/atom
BoneCosmo lcdm --wznow --do runs/lss
BoneCosmo sr   --wznow --do runs/frag
BoneCosmo sm   --wznow --do runs/plazma
```

Instalator MSI (wymaga [WiX Toolset](https://wixtoolset.org/) v3):

```
cargo install cargo-wix
cargo wix --nocapture
```

Albo skopiuj `BoneCosmo.exe` — to jeden plik, bez zależności .NET.

Workspace: `core` (fizyka, w tym `gr`), `ui` (mapa, lekcje, okno), `app` (binarka).
