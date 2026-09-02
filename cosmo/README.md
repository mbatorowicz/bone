# Bone Cosmo

Aplikacja desktopowa: dwa modele grawitacji N ciał na jednym komputerze.

- **ΛCDM (Planck 2018)** — próbka materii, Particle-Mesh z izolowanymi brzegami
  (Hockney), warunki Zel'dovicha, całkowanie po `ln a`.
- **SR** — odosobniona chmura, kinematyka szczególnej teorii względności,
  opcjonalne chłodzenie.

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
BoneCosmo lcdm --wznow --do runs/lss
BoneCosmo sr   --wznow --do runs/frag
```

Instalator MSI (wymaga [WiX Toolset](https://wixtoolset.org/) v3):

```
cargo install cargo-wix
cargo wix --nocapture
```

Albo skopiuj `BoneCosmo.exe` — to jeden plik, bez zależności .NET.

Workspace: `core` (fizyka), `ui` (okno), `app` (binarka).
