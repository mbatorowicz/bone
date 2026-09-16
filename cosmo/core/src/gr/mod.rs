//! Liczby kursu STW → OTW, osobno od laboratorium N-ciał.
//!
//! [`lorentz`] to słownik między układami inercjalnymi: boost, γ, dylatacja,
//! kontrakcja i interwał Minkowskiego. Stanem jest znacznik `(ct, x, y, z)`,
//! nie pęd cząstki. Dlatego ten katalog nie wolno zlewać z [`crate::sr`]:
//! tam `γ` wychodzi z `p`, tu z prędkości układu.
//!
//! [`rk4`] to stepper na wektorze stanu. [`metric`] to Schwarzschild jako
//! `g_μν` i trzy promienie `2M` / `3M` / `6M`. [`kerr`] to Kerr w
//! Boyer-Lindquist: `g_μν` z kratką `g_tφ`, analityczne Γ (osobny typ —
//! [`christoffel::Christoffel`] nie zna `g_tφ`) i geodezyjna na tym samym
//! stanie 8 liczb. [`christoffel`] to analityczne Γ Schwarzschilda, [`geodesic`] składa je z RK4
//! w tor `(t, r, θ, φ)`. [`raytrace`] to obraz: piksel = geodezyjna zerowa
//! wstecz, bez okna. [`tensor`] to algebra 4D: wektor, kowektor, maszyna
//! (1,1), waga (0,2) i η Minkowskiego. [`einstein`] to Riemann, Ricci,
//! `G_μν` i `T_μν` na Schwarzschildu. [`pinn`] to residual ciepła i fali
//! oraz mała sieć bez biblioteki ML.

pub mod christoffel;
pub mod einstein;
pub mod geodesic;
pub mod kerr;
pub mod lorentz;
pub mod metric;
pub mod pinn;
pub mod raytrace;
pub mod rk4;
pub mod tensor;

pub use christoffel::Christoffel;
pub use einstein::{dust, field_residual, vacuum, Curvature};
pub use geodesic::{GeodesicError, GeodesicState};
pub use kerr::{Kerr, KerrChristoffel, KerrError, KerrGeoError};
pub use lorentz::{
    boost, boost_x, compose_boost_1d, contract_rod, contracted_length, dilated_time, gamma,
    gamma_from_beta, Event, Superluminal,
};
pub use metric::{horizon_radius, isco_radius, photon_sphere_radius, MetricError, Schwarzschild};
pub use pinn::{heat_exact, heat_loss, heat_residual, step_heat, wave_exact, wave_residual, Net};
pub use raytrace::{Buffer, Hit, RaytraceError};
pub use tensor::{Covector, Tensor02, Tensor11, Tensor20, Vector};
