//! Liczby kursu STW → OTW, osobno od laboratorium N-ciał.
//!
//! [`lorentz`] to słownik między układami inercjalnymi: boost, γ, dylatacja,
//! kontrakcja i interwał Minkowskiego. Stanem jest znacznik `(ct, x, y, z)`,
//! nie pęd cząstki. Dlatego ten katalog nie wolno zlewać z [`crate::sr`]:
//! tam `γ` wychodzi z `p`, tu z prędkości układu.
//!
//! Metryka, RK4 i geodezyjna przychodzą w kolejnych krokach. Teraz jest tylko
//! szczególna teoria względności jako arytmetyka.

pub mod lorentz;

pub use lorentz::{
    boost, boost_x, compose_boost_1d, contract_rod, contracted_length, dilated_time, gamma,
    gamma_from_beta, Event, Superluminal,
};
