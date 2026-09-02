//! Model standardowy kosmologii: ΛCDM z parametrami Plancka 2018.
//!
//! Bieg zaczyna się przy `z = 49`, kiedy zaburzenia gęstości są jeszcze liniowe,
//! i prowadzi próbkę materii do dziś i dalej. Zimna ciemna materia jest jedyną
//! składową grawitującą — barionów, gazu ani chłodzenia tu nie ma, bo struktura
//! wielkoskalowa ich nie potrzebuje.
//!
//! Podział modułów:
//!
//! - [`units`] — układ jednostek Mpc/h, 10¹⁰ M☉/h, km/s,
//! - [`cosmology`] — tło: `E(a)`, `D(a)`, wiek, czynniki kroku,
//! - [`power`] — widmo mocy `P(k)` znormalizowane przez `σ₈`,
//! - [`ics`] — gaussowskie pole gęstości i przesunięcia Zel'dovicha,
//! - [`engine`] — leapfrog KDK po `ln a` na siatce PM,
//! - [`presets`] — nazwane zestawy nastaw (CLI i panel).
//!
//! Solver grawitacji jest wspólny z trybem SR ([`crate::mesh`]) — te dwa modele
//! różnią się kinematyką i warunkami początkowymi, nie sposobem liczenia sił.

pub mod cosmology;
pub mod engine;
pub mod ics;
pub mod power;
pub mod presets;
pub mod units;

pub use cosmology::Cosmology;
pub use engine::{Engine, RunConfig, Saved};
pub use presets::Preset;
