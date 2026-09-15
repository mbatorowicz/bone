//! Liczby kursu STW → OTW, osobno od laboratorium N-ciał.
//!
//! [`lorentz`] to słownik między układami inercjalnymi: boost, γ, dylatacja,
//! kontrakcja i interwał Minkowskiego. Stanem jest znacznik `(ct, x, y, z)`,
//! nie pęd cząstki. Dlatego ten katalog nie wolno zlewać z [`crate::sr`]:
//! tam `γ` wychodzi z `p`, tu z prędkości układu.
//!
//! [`rk4`] to stepper na wektorze stanu. [`metric`] to Schwarzschild jako
//! `g_μν` i trzy promienie `2M` / `3M` / `6M`. Γ i geodezyjna przychodzą
//! osobno — tu jeszcze nie ma toru, tylko linijka i sposób robienia kroku.

pub mod lorentz;
pub mod metric;
pub mod rk4;

pub use lorentz::{
    boost, boost_x, compose_boost_1d, contract_rod, contracted_length, dilated_time, gamma,
    gamma_from_beta, Event, Superluminal,
};
pub use metric::{horizon_radius, isco_radius, photon_sphere_radius, MetricError, Schwarzschild};
