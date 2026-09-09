# Bone Cosmo

Cztery laboratoria na jednym komputerze: N-ciała, kosmologia, cząstki i atomy.

- **Kosmologia** (`lcdm`) — ΛCDM, Planck 2018, Particle-Mesh z izolowanymi
  brzegami (Hockney), warunki Zel'dovicha, całkowanie po `ln a`.
- **N-ciała** (`sr`) — Newton + kinematyka SR, odosobniona chmura,
  opcjonalna dyssypacja. Nie OTW.
- **Cząstki** (`sm`) — klasyczny gaz + rozpady PDG. To nie jest QFT.
- **Atomy** (`qm`) — Schrödinger · `|ψ|²`. Wodór dokładny, hel wariacyjny
  (~2%), Slater jako lekcja ekranowania.

Liczy na tym komputerze (FFT + wgpu do okna). Nie używa Vercela.

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

Workspace: `core` (fizyka), `ui` (okno), `app` (binarka).
