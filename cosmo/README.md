# Bone Cosmo

Aplikacja desktopowa: **ΛCDM (Planck 2018)**, periodyczny Particle-Mesh, warunki Zel'dovicha. Liczy na tym komputerze (FFT + wgpu do okna). Nie używa Vercela.

```
cargo run --release
```

Binarka: `target/release/BoneCosmo.exe`

Instalator MSI (wymaga [WiX Toolset](https://wixtoolset.org/) v3):

```
cargo install cargo-wix
cargo wix --nocapture
```

Albo skopiuj `BoneCosmo.exe` — to jeden plik, bez zależności .NET.
