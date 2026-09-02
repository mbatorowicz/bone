//! Grawitacja N ciał: dwa modele na wspólnym silniku.
//!
//! # Co tu jest
//!
//! - [`sr`] — odosobniona chmura cząstek. Grawitacja newtonowska, kinematyka
//!   szczególnej teorii względności: pęd `p = γmv` jest zmienną stanu, więc żadna
//!   cząstka nie przekroczy `c`. Do tego dyssypacja zależna od gęstości, dzięki
//!   której chmura potrafi się zapaść i pofragmentować.
//! - [`lcdm`] — próbka wszechświata z parametrami Plancka 2018. Warunki początkowe
//!   z widma mocy i przybliżenia Zel'dovicha, całkowanie po `ln a`.
//!
//! # Co jest wspólne
//!
//! Oba modele różnią się kinematyką i warunkami początkowymi, nie sposobem liczenia
//! grawitacji. Wspólne są więc: [`mesh`] (solver Particle-Mesh z izolowanymi
//! brzegami), [`grid`] (pudło siatki i wagi cloud-in-cell), [`fft`], [`vec3`]
//! i [`rng`]. To nie jest wynik dążenia do współdzielenia kodu — to obserwacja, że
//! obie symulacje rozwiązują to samo równanie Poissona.
//!
//! Warstwy wyższe: [`io`] (checkpoint i trajektoria), [`session`] (pętla biegu)
//! i [`cli`] (bieg wsadowy) nie zawierają fizyki i nie są przez fizykę używane.
//! Okno leży w osobnym crate `bone-ui`.

pub mod cli;
pub mod fft;
pub mod grid;
pub mod io;
pub mod lcdm;
pub mod mesh;
pub mod rng;
pub mod session;
pub mod sr;
pub mod vec3;

#[cfg(test)]
pub(crate) mod fixtures;

pub use vec3::{vec3, Vec3};
