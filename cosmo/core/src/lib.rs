//! Cztery laboratoria na wspólnym silniku: N-ciała, kosmologia, cząstki i atomy.
//!
//! # Co tu jest
//!
//! - [`sr`] — laboratorium N-ciał. Grawitacja newtonowska, kinematyka
//!   szczególnej teorii względności: pęd `p = γmv` jest zmienną stanu, więc żadna
//!   cząstka nie przekroczy `c`. Do tego dyssypacja zależna od gęstości, dzięki
//!   której chmura potrafi się zapaść i pofragmentować. To nie jest OTW.
//! - [`lcdm`] — laboratorium kosmologii. ΛCDM z parametrami Plancka 2018,
//!   Particle-Mesh z izolowanymi brzegami. Warunki początkowe z widma mocy
//!   i przybliżenia Zel'dovicha, całkowanie po `ln a`.
//! - [`sm`] — laboratorium cząstek. Klasyczny gaz + losowe rozpady PDG.
//!   To nie jest kwantowa teoria pola — granice opisu są wypisane w [`sm`].
//! - [`qm`] — laboratorium atomów. Wodór i He⁺ są dokładnym rozwiązaniem
//!   Schrödingera; hel — wariacją z błędem ~2%; Slater zostaje jako lekcja
//!   ekranowania z mierzonym błędem IE. Chmura to próbka `|ψ|²`.
//!
//! # Co jest wspólne
//!
//! Laboratoria N-ciał i kosmologii różnią się kinematyką i warunkami
//! początkowymi, nie sposobem liczenia grawitacji. Wspólne są więc: [`mesh`]
//! (solver Particle-Mesh z izolowanymi brzegami), [`grid`] (pudło siatki
//! i wagi cloud-in-cell), [`fft`], [`vec3`] i [`rng`]. Cząstki biorą z tego
//! kinematykę relatywistyczną i solver dalekozasięgowy (Coulomb to to samo
//! równanie co grawitacja, z innym ładunkiem).
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
