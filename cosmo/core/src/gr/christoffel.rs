//! Analityczne symbole Christoffela Schwarzschilda, `G = c = 1`.
//!
//! Γ nie jest zgadywane z różniczki `g_μν`: tu są wzory zamknięte. Dzięki
//! temu geodezyjna w [`super::geodesic`] ma skąd wziąć przyspieszenie
//! `a^μ = −Γ^μ_αβ u^α u^β` bez numerycznej pochodnej metryki.
//!
//! Sygnatura (−,+,+,+) jak w [`super::metric`]. Współrzędne `(t, r, θ, φ)`.
//! Na horyzoncie `r = 2M` te Γ rozjeżdżają się razem z `g_rr` — te same
//! współrzędne, ten sam błąd [`MetricError::AtHorizon`].

use super::metric::{MetricError, Schwarzschild};

/// Niezależne niezerowe `Γ^μ_αβ`. Reszta to zera albo odbicia `α ↔ β`.
///
/// Indeksy dolne są symetryczne. W równaniu geodezyjnej każdy mieszany
/// składnik wchodzi więc dwa razy: `2 Γ^t_tr u^t u^r`, nie `Γ^t_tr u^t u^r`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Christoffel {
    /// `Γ^t_tr = Γ^t_rt = M / (r² f)`.
    pub gamma_t_tr: f64,
    /// `Γ^r_tt = M f / r²`.
    pub gamma_r_tt: f64,
    /// `Γ^r_rr = −M / (r² f)`.
    pub gamma_r_rr: f64,
    /// `Γ^r_θθ = −r f`.
    pub gamma_r_theta_theta: f64,
    /// `Γ^r_φφ = −r f sin²θ`.
    pub gamma_r_phi_phi: f64,
    /// `Γ^θ_rθ = Γ^θ_θr = 1/r`.
    pub gamma_theta_r_theta: f64,
    /// `Γ^θ_φφ = −sinθ cosθ`.
    pub gamma_theta_phi_phi: f64,
    /// `Γ^φ_rφ = Γ^φ_φr = 1/r`.
    pub gamma_phi_r_phi: f64,
    /// `Γ^φ_θφ = Γ^φ_φθ = cot θ`.
    pub gamma_phi_theta_phi: f64,
}

impl Christoffel {
    /// Γ w punkcie `(r, θ)`. Czas nie wchodzi: Schwarzschild jest stacjonarny.
    pub fn at(bh: Schwarzschild, r: f64, theta: f64) -> Result<Self, MetricError> {
        let f = bh.f(r)?;
        if !theta.is_finite() {
            return Err(MetricError::InvalidAngle { theta });
        }
        let sin = theta.sin();
        if sin.abs() < f64::EPSILON {
            return Err(MetricError::InvalidAngle { theta });
        }
        let cos = theta.cos();
        let m = bh.mass();
        let r2 = r * r;
        Ok(Self {
            gamma_t_tr: m / (r2 * f),
            gamma_r_tt: m * f / r2,
            gamma_r_rr: -m / (r2 * f),
            gamma_r_theta_theta: -r * f,
            gamma_r_phi_phi: -r * f * sin * sin,
            gamma_theta_r_theta: 1.0 / r,
            gamma_theta_phi_phi: -sin * cos,
            gamma_phi_r_phi: 1.0 / r,
            gamma_phi_theta_phi: cos / sin,
        })
    }

    /// `a^μ = −Γ^μ_αβ u^α u^β` dla czteroprędkości `u = (u^t, u^r, u^θ, u^φ)`.
    pub fn accel(self, u: [f64; 4]) -> [f64; 4] {
        let [u_t, u_r, u_theta, u_phi] = u;
        let a_t = -2.0 * self.gamma_t_tr * u_t * u_r;
        let a_r = -(self.gamma_r_tt * u_t * u_t
            + self.gamma_r_rr * u_r * u_r
            + self.gamma_r_theta_theta * u_theta * u_theta
            + self.gamma_r_phi_phi * u_phi * u_phi);
        let a_theta = -(2.0 * self.gamma_theta_r_theta * u_r * u_theta
            + self.gamma_theta_phi_phi * u_phi * u_phi);
        let a_phi = -(2.0 * self.gamma_phi_r_phi * u_r * u_phi
            + 2.0 * self.gamma_phi_theta_phi * u_theta * u_phi);
        [a_t, a_r, a_theta, a_phi]
    }
}

impl Schwarzschild {
    /// Analityczne Γ w `(r, θ)`.
    pub fn christoffel(self, r: f64, theta: f64) -> Result<Christoffel, MetricError> {
        Christoffel::at(self, r, theta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

    fn mass_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    #[test]
    fn textbook_values_at_isco_equator() {
        let g = mass_one().christoffel(6.0, FRAC_PI_2).unwrap();
        assert!((g.gamma_t_tr - 1.0 / 24.0).abs() < 1e-15);
        assert!((g.gamma_r_tt - 1.0 / 54.0).abs() < 1e-15);
        assert!((g.gamma_r_rr + 1.0 / 24.0).abs() < 1e-15);
        assert!((g.gamma_r_theta_theta + 4.0).abs() < 1e-15);
        assert!((g.gamma_r_phi_phi + 4.0).abs() < 1e-15);
        assert!((g.gamma_theta_r_theta - 1.0 / 6.0).abs() < 1e-15);
        assert!(g.gamma_theta_phi_phi.abs() < 1e-15);
        assert!((g.gamma_phi_r_phi - 1.0 / 6.0).abs() < 1e-15);
        assert!(g.gamma_phi_theta_phi.abs() < 1e-15);
    }

    #[test]
    fn textbook_values_at_photon_sphere() {
        let g = mass_one().christoffel(3.0, FRAC_PI_2).unwrap();
        assert!((g.gamma_t_tr - 1.0 / 3.0).abs() < 1e-15);
        assert!((g.gamma_r_tt - 1.0 / 27.0).abs() < 1e-15);
        assert!((g.gamma_r_rr + 1.0 / 3.0).abs() < 1e-15);
        assert!((g.gamma_r_theta_theta + 1.0).abs() < 1e-15);
        assert!((g.gamma_r_phi_phi + 1.0).abs() < 1e-15);
    }

    #[test]
    fn four_m_matches_closed_form() {
        let r = 4.0;
        let f = 0.5;
        let g = mass_one().christoffel(r, FRAC_PI_2).unwrap();
        assert!((g.gamma_t_tr - 1.0 / (r * r * f)).abs() < 1e-15);
        assert!((g.gamma_t_tr - 0.125).abs() < 1e-15);
        assert!((g.gamma_r_tt - f / (r * r)).abs() < 1e-15);
    }

    #[test]
    fn zero_mass_is_spherical_minkowski() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let r = 5.0;
        let g = flat.christoffel(r, FRAC_PI_2).unwrap();
        assert_eq!(g.gamma_t_tr, 0.0);
        assert_eq!(g.gamma_r_tt, 0.0);
        assert_eq!(g.gamma_r_rr, 0.0);
        assert!((g.gamma_r_theta_theta + r).abs() < 1e-15);
        assert!((g.gamma_r_phi_phi + r).abs() < 1e-15);
        assert!((g.gamma_theta_r_theta - 1.0 / r).abs() < 1e-15);
        assert!(g.gamma_theta_phi_phi.abs() < 1e-15);
        assert!((g.gamma_phi_r_phi - 1.0 / r).abs() < 1e-15);
        assert!(g.gamma_phi_theta_phi.abs() < 1e-15);
    }

    #[test]
    fn off_equator_phi_piece_is_cot() {
        let g = mass_one().christoffel(6.0, FRAC_PI_4).unwrap();
        let sin = FRAC_PI_4.sin();
        let cos = FRAC_PI_4.cos();
        assert!((g.gamma_theta_phi_phi + sin * cos).abs() < 1e-15);
        assert!((g.gamma_phi_theta_phi - cos / sin).abs() < 1e-15);
        assert!((g.gamma_r_phi_phi - g.gamma_r_theta_theta * sin * sin).abs() < 1e-15);
    }

    #[test]
    fn static_observer_has_zero_accel_in_flat_space() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let a = flat
            .christoffel(3.0, FRAC_PI_2)
            .unwrap()
            .accel([1.0, 0.0, 0.0, 0.0]);
        assert!(a.iter().all(|v| v.abs() < 1e-15), "a={a:?}");
    }

    #[test]
    fn horizon_is_rejected() {
        assert!(matches!(
            mass_one().christoffel(2.0, FRAC_PI_2),
            Err(MetricError::AtHorizon { r, mass }) if r == 2.0 && mass == 1.0
        ));
    }

    #[test]
    fn pole_is_rejected() {
        assert!(matches!(
            mass_one().christoffel(6.0, 0.0),
            Err(MetricError::InvalidAngle { theta }) if theta == 0.0
        ));
        assert!(mass_one().christoffel(6.0, std::f64::consts::PI).is_err());
        assert!(mass_one().christoffel(6.0, f64::NAN).is_err());
    }
}
