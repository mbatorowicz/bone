//! Grawitacja z kinematyką szczególnej teorii względności: odosobniona chmura cząstek.
//!
//! Model jest jeden i warto go nazwać wprost, żeby nikt nie oczekiwał po nim czegoś
//! innego: grawitacja jest newtonowska (chwilowa, `1/r²`), a względność siedzi
//! wyłącznie w kinematyce — pęd `p = γmv` jest zmienną stanu, a prędkość wynika z
//! pędu, więc żadna cząstka nie przekroczy `c`. To NIE jest ogólna teoria
//! względności: nie ma metryki, nie ma opóźnienia oddziaływania.
//!
//! Podział na moduły idzie po odpowiedzialnościach, nie po rozmiarach plików:
//!
//! - [`relativity`] — przeliczenia `p ↔ v`, `γ`, energia kinetyczna,
//! - [`state`] — położenia, pędy, masy i pole sił jednego biegu,
//! - [`config`] i [`presets`] — parametry oraz gotowe zestawy nastaw,
//! - [`spawn`] — warunki początkowe (geometria, obrót, równowaga wirialna),
//! - [`backends`] — solvery grawitacji: dokładny `O(N²)` i siatkowy PM,
//! - [`integrator`] — leapfrog KDK z krokiem dobieranym adaptacyjnie,
//! - [`cooling`] — dyssypacja zależna od gęstości,
//! - [`diagnostics`] — energia, wirial, dryf pędu, błąd siły,
//! - [`engine`] — spięcie powyższych w jeden obiekt sterowany krok po kroku.

pub mod backends;
pub mod config;
pub mod cooling;
pub mod diagnostics;
pub mod engine;
pub mod integrator;
pub mod presets;
pub mod relativity;
pub mod spawn;
pub mod state;

pub use config::Config;
pub use engine::Engine;
pub use state::State;
