//! Grawitacja N ciał, gaz cząstek i atomy: cztery modele na wspólnym silniku.
//!
//! # Co tu jest
//!
//! - [`sr`] — odosobniona chmura cząstek. Grawitacja newtonowska, kinematyka
//!   szczególnej teorii względności: pęd `p = γmv` jest zmienną stanu, więc żadna
//!   cząstka nie przekroczy `c`. Do tego dyssypacja zależna od gęstości, dzięki
//!   której chmura potrafi się zapaść i pofragmentować.
//! - [`lcdm`] — próbka wszechświata z parametrami Plancka 2018. Warunki początkowe
//!   z widma mocy i przybliżenia Zel'dovicha, całkowanie po `ln a`.
//! - [`sm`] — Model Standardowy jako klasyczny gaz cząstek. Siedemnaście gatunków
//!   elementarnych plus hadrony, cztery oddziaływania, rozpady i anihilacja.
//!   To nie jest kwantowa teoria pola — granice opisu są wypisane w [`sm`].
//! - [`qm`] — atomy i orbitale. Wodór i jony wodoropodobne są dokładnym
//!   rozwiązaniem Schrödingera; wieloelektronowe atomy — przybliżeniem Slatera
//!   z mierzonym błędem wobec tablic jonizacji. To nie jest QFT.
//!
//! # Co jest wspólne
//!
//! Modele `sr` i `lcdm` różnią się kinematyką i warunkami początkowymi, nie
//! sposobem liczenia grawitacji. Wspólne są więc: [`mesh`] (solver Particle-Mesh
//! z izolowanymi brzegami), [`grid`] (pudło siatki i wagi cloud-in-cell), [`fft`],
//! [`vec3`] i [`rng`]. `sm` bierze z tego kinematykę relatywistyczną i solver
//! dalekozasięgowy (Coulomb to to samo równanie co grawitacja, z innym ładunkiem).
//!
//! Warstwy wyższe: [`io`] (checkpoint i trajektoria), [`session`] (pętla biegu)
//! i [`cli`] (bieg wsadowy) nie zawierają fizyki i nie są przez fizykę używane.
//! Okno leży w osobnym crate `bone-ui`.
//!
//! Stałe katalogowe (CODATA 2022, PDG 2025) są w [`constants`] — jeden wpis
//! na α, ħc i masy e/p/n/W, zamiast osobnych liczb w `sm` i `qm`.

pub mod cli;
pub mod constants;
pub mod fft;
pub mod grid;
pub mod io;
pub mod lcdm;
pub mod mesh;
pub mod qm;
pub mod rng;
pub mod session;
pub mod sm;
pub mod sr;
pub mod vec3;

#[cfg(test)]
pub(crate) mod fixtures;

pub use vec3::{vec3, Vec3};
