//! Laboratorium atomów: Schrödinger, chmura to próbka `|ψ|²`.
//!
//! # Czym to jest
//!
//! Stany związane atomu wodoropodobnego są tu rozwiązaniem
//! **dokładnym** równania Schrödingera: `ψ_{nlm} = R_{nl}(r) Y_{lm}(θ,φ)`,
//! energia `E_n = −μ Z² / (2n²)` hartree. Chmura na ekranie to próbka `|ψ|²`,
//! nie zbiór elektronów — jeden elektron w `1s` jest tysiącami punktów, bo
//! inaczej orbitalu nie widać.
//!
//! Superpozycja stanów o różnych `n` ewoluuje w czasie fazą `e^{−i E t / ħ}`.
//! Gęstość bije; to jedyna dynamika, jaka tu istnieje. Nie ma sił i nie ma
//! trajektorii.
//!
//! Hel jest **wariacją** `ψ = e^{−ζr₁} e^{−ζr₂}` (`ζ = 27/16`): energia
//! całkowita myli się o ~2% wobec dokładnego nietraktacyjnego stanu.
//! Slater zostaje jako lekcja ekranowania — błąd IE wobec NIST jest tematem
//! karty, nie reklamą atomu. Ciężkie atomy (Fe, Cu, Au, U) pokazują
//! konfigurację Aufbau, bez energetyki jonizacji.
//!
//! # Czym to NIE jest
//!
//! - **kwantową teorią pola** ani QED — brak fotonów, samozderzeń, przesunięcia Lamba;
//! - **pełnym atomem wieloelektronowym** — brak korelacji, wymienności Hartree–Focka
//!   i spin-orbity jako dynamiki;
//! - **równaniem Diraca** — struktura subtelna jest wzorem, nie ruchem;
//! - **chemią** — cząsteczki, wiązania i hybryda `sp³` nie są tu liczone, tylko
//!   wspomniane jako to, czego model nie obejmuje;
//! - **poprawką do laboratorium cząstek (`sm`)**. Klasyczny gaz nadal nie wiąże elektronu z
//!   protonem. Ten moduł odpowiada na inne pytanie.
//!
//! Podział: [`units`], [`hydrogen`], [`helium`], [`elements`], [`config`], [`plot`],
//! [`sample`], [`state`], [`engine`], [`diagnostics`], [`presets`].

pub mod config;
pub mod diagnostics;
pub mod elements;
pub mod engine;
pub mod helium;
pub mod hydrogen;
pub mod plot;
pub mod presets;
pub mod sample;
pub mod state;
pub mod units;

pub use config::{Config, Scene, Term};
pub use diagnostics::Snapshot;
pub use engine::Engine;
pub use state::State;
