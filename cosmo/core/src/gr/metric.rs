//! Metryka Schwarzschilda jako liczby, w jednostkach `G = c = 1`.
//!
//! To linijka i zegar w punkcie, nie tor. Współczynniki `g_μν` wystarczą,
//! żeby lekcja „metryka słowami” miała konkret: daleko jak Minkowski, na
//! horyzoncie współrzędne `(t, r, θ, φ)` się łamią, a `2M` / `3M` / `6M`
//! to trzy różne promienie, nie trzy nazwy tego samego miejsca.
//!
//! Sygnatura (−,+,+,+):
//!
//! ```text
//! ds² = −(1 − 2M/r) dt² + dr² / (1 − 2M/r) + r² dθ² + r² sin²θ dφ²
//! ```
//!
//! [`crate::gr::lorentz`] liczy interwał Minkowskiego ze znakiem `+−−−`
//! i współrzędną `ct`. Tu czas jest `t`, nie `ct`, a znak jest podręcznikowy
//! dla OTW. To nie sprzeczność — to dwa słowniki, zanim Γ je złoży w ruch.

use std::fmt;

/// Masa nie jest liczbą, promień nie jest miejscem, albo stoimy na horyzoncie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MetricError {
    InvalidMass { mass: f64 },
    InvalidRadius { r: f64 },
    InvalidAngle { theta: f64 },
    AtHorizon { r: f64, mass: f64 },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidMass { mass } => {
                write!(f, "masa M={mass} musi być skończona i ≥ 0")
            }
            Self::InvalidRadius { r } => {
                write!(f, "promień r={r} musi być skończony i > 0")
            }
            Self::InvalidAngle { theta } => {
                write!(f, "kąt θ={theta} musi być skończony")
            }
            Self::AtHorizon { r, mass } => write!(
                f,
                "współrzędne Schwarzschilda łamią się na horyzoncie r={r} = 2M (M={mass})"
            ),
        }
    }
}

impl std::error::Error for MetricError {}

/// `r_h = 2M`. Światło stąd już nie wychodzi — w tych współrzędnych.
pub fn horizon_radius(mass: f64) -> f64 {
    2.0 * mass
}

/// Sfera fotonowa: `r = 3M`. Niestabilna orbita zerowa w równiku.
pub fn photon_sphere_radius(mass: f64) -> f64 {
    3.0 * mass
}

/// ISCO masywnej cząstki: `r = 6M`.
pub fn isco_radius(mass: f64) -> f64 {
    6.0 * mass
}

/// Masa w środku, reszta to próżnia. `M = 0` to Minkowski w sferycznych.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schwarzschild {
    mass: f64,
}

impl Schwarzschild {
    pub fn new(mass: f64) -> Result<Self, MetricError> {
        if !mass.is_finite() || mass < 0.0 {
            return Err(MetricError::InvalidMass { mass });
        }
        Ok(Self { mass })
    }

    pub fn mass(self) -> f64 {
        self.mass
    }

    pub fn horizon_radius(self) -> f64 {
        horizon_radius(self.mass)
    }

    pub fn photon_sphere_radius(self) -> f64 {
        photon_sphere_radius(self.mass)
    }

    pub fn isco_radius(self) -> f64 {
        isco_radius(self.mass)
    }

    fn check_r(self, r: f64) -> Result<(), MetricError> {
        if !r.is_finite() || r <= 0.0 {
            Err(MetricError::InvalidRadius { r })
        } else {
            Ok(())
        }
    }

    fn check_theta(theta: f64) -> Result<(), MetricError> {
        if theta.is_finite() {
            Ok(())
        } else {
            Err(MetricError::InvalidAngle { theta })
        }
    }

    fn horizon_if_singular(self, r: f64, f: f64) -> Result<f64, MetricError> {
        if f == 0.0 || !f.is_finite() {
            Err(MetricError::AtHorizon { r, mass: self.mass })
        } else {
            Ok(f)
        }
    }

    /// `f(r) = 1 − 2M/r`. Zero na horyzoncie, jedynka w nieskończoności.
    pub fn f(self, r: f64) -> Result<f64, MetricError> {
        self.check_r(r)?;
        self.horizon_if_singular(r, 1.0 - 2.0 * self.mass / r)
    }

    /// `g_tt = −(1 − 2M/r)`.
    pub fn g_tt(self, r: f64) -> Result<f64, MetricError> {
        Ok(-self.f(r)?)
    }

    /// `g_rr = 1 / (1 − 2M/r)`.
    pub fn g_rr(self, r: f64) -> Result<f64, MetricError> {
        Ok(1.0 / self.f(r)?)
    }

    /// `g_θθ = r²`.
    pub fn g_theta_theta(self, r: f64) -> Result<f64, MetricError> {
        self.check_r(r)?;
        Ok(r * r)
    }

    /// `g_φφ = r² sin²θ`.
    pub fn g_phi_phi(self, r: f64, theta: f64) -> Result<f64, MetricError> {
        self.check_r(r)?;
        Self::check_theta(theta)?;
        let s = theta.sin();
        Ok(r * r * s * s)
    }

    /// Pełna macierz `g_μν` w kolejności `(t, r, θ, φ)`. Poza przekątną zera.
    pub fn components(self, r: f64, theta: f64) -> Result<[[f64; 4]; 4], MetricError> {
        let mut g = [[0.0; 4]; 4];
        g[0][0] = self.g_tt(r)?;
        g[1][1] = self.g_rr(r)?;
        g[2][2] = self.g_theta_theta(r)?;
        g[3][3] = self.g_phi_phi(r, theta)?;
        Ok(g)
    }

    /// `g^{μν}`: odwrotność przekątnej, bo metryka jest diagonalna.
    pub fn inverse(self, r: f64, theta: f64) -> Result<[[f64; 4]; 4], MetricError> {
        let g = self.components(r, theta)?;
        Ok([
            [1.0 / g[0][0], 0.0, 0.0, 0.0],
            [0.0, 1.0 / g[1][1], 0.0, 0.0],
            [0.0, 0.0, 1.0 / g[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0 / g[3][3]],
        ])
    }

    /// `ds² = g_μν dx^μ dx^ν` dla przesunięcia współrzędnych.
    pub fn ds2(self, r: f64, theta: f64, dx: [f64; 4]) -> Result<f64, MetricError> {
        let g = self.components(r, theta)?;
        Ok(g[0][0] * dx[0] * dx[0]
            + g[1][1] * dx[1] * dx[1]
            + g[2][2] * dx[2] * dx[2]
            + g[3][3] * dx[3] * dx[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn mass_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    #[test]
    fn characteristic_radii_are_textbook_multiples() {
        let m = 1.5;
        assert!((horizon_radius(m) - 3.0).abs() < 1e-15);
        assert!((photon_sphere_radius(m) - 4.5).abs() < 1e-15);
        assert!((isco_radius(m) - 9.0).abs() < 1e-15);
        let bh = Schwarzschild::new(m).unwrap();
        assert!((bh.horizon_radius() - 2.0 * m).abs() < 1e-15);
        assert!((bh.photon_sphere_radius() - 3.0 * m).abs() < 1e-15);
        assert!((bh.isco_radius() - 6.0 * m).abs() < 1e-15);
    }

    #[test]
    fn g_tt_and_g_rr_match_schwarzschild_formula() {
        let bh = mass_one();
        let r = 6.0;
        let f = 1.0 - 2.0 / r;
        assert!((bh.f(r).unwrap() - f).abs() < 1e-15);
        assert!((bh.g_tt(r).unwrap() + f).abs() < 1e-15);
        assert!((bh.g_rr(r).unwrap() - 1.0 / f).abs() < 1e-15);
        assert!((bh.g_tt(r).unwrap() + 2.0 / 3.0).abs() < 1e-15);
        assert!((bh.g_rr(r).unwrap() - 1.5).abs() < 1e-15);
    }

    #[test]
    fn g_tt_g_rr_at_four_m() {
        let bh = mass_one();
        let r = 4.0;
        assert!((bh.g_tt(r).unwrap() + 0.5).abs() < 1e-15);
        assert!((bh.g_rr(r).unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn angular_parts_are_sphere() {
        let bh = mass_one();
        let r = 5.0;
        assert!((bh.g_theta_theta(r).unwrap() - 25.0).abs() < 1e-15);
        assert!((bh.g_phi_phi(r, FRAC_PI_2).unwrap() - 25.0).abs() < 1e-15);
        assert!(bh.g_phi_phi(r, 0.0).unwrap().abs() < 1e-15);
        let thirty_deg = 0.5_f64.asin();
        assert!((bh.g_phi_phi(r, thirty_deg).unwrap() - 6.25).abs() < 1e-15);
    }

    fn off_diagonal_is_zero(m: [[f64; 4]; 4]) {
        for (i, row) in m.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if i != j {
                    assert_eq!(val, 0.0);
                }
            }
        }
    }

    #[test]
    fn matrix_is_diagonal() {
        let g = mass_one().components(6.0, FRAC_PI_2).unwrap();
        off_diagonal_is_zero(g);
        assert!((g[0][0] + 2.0 / 3.0).abs() < 1e-15);
        assert!((g[1][1] - 1.5).abs() < 1e-15);
        assert!((g[2][2] - 36.0).abs() < 1e-15);
        assert!((g[3][3] - 36.0).abs() < 1e-15);
    }

    #[test]
    fn inverse_undoes_the_metric() {
        let bh = mass_one();
        let g = bh.components(8.0, 0.7).unwrap();
        let gi = bh.inverse(8.0, 0.7).unwrap();
        off_diagonal_is_zero(gi);
        let diag_g = [g[0][0], g[1][1], g[2][2], g[3][3]];
        let diag_gi = [gi[0][0], gi[1][1], gi[2][2], gi[3][3]];
        for (g_ii, gi_ii) in diag_g.iter().zip(diag_gi) {
            assert!(
                (g_ii * gi_ii - 1.0).abs() < 1e-12,
                "g_ii={g_ii} g^ii={gi_ii}"
            );
        }
    }

    #[test]
    fn zero_mass_is_minkowski_in_sphericals() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let r = 3.0;
        let theta = FRAC_PI_2;
        assert!((flat.g_tt(r).unwrap() + 1.0).abs() < 1e-15);
        assert!((flat.g_rr(r).unwrap() - 1.0).abs() < 1e-15);
        assert!((flat.g_theta_theta(r).unwrap() - 9.0).abs() < 1e-15);
        assert!((flat.g_phi_phi(r, theta).unwrap() - 9.0).abs() < 1e-15);
        assert_eq!(flat.horizon_radius(), 0.0);
    }

    #[test]
    fn far_away_approaches_minkowski() {
        let bh = mass_one();
        let r = 1.0e6;
        assert!((bh.g_tt(r).unwrap() + 1.0).abs() < 3e-6);
        assert!((bh.g_rr(r).unwrap() - 1.0).abs() < 3e-6);
    }

    #[test]
    fn interior_swaps_time_and_radius_character() {
        let bh = mass_one();
        let r = 1.0;
        assert!(bh.g_tt(r).unwrap() > 0.0);
        assert!(bh.g_rr(r).unwrap() < 0.0);
        assert!((bh.g_tt(r).unwrap() - 1.0).abs() < 1e-15);
        assert!((bh.g_rr(r).unwrap() + 1.0).abs() < 1e-15);
    }

    #[test]
    fn horizon_is_rejected() {
        let bh = mass_one();
        assert!(matches!(
            bh.g_rr(2.0),
            Err(MetricError::AtHorizon { r, mass }) if r == 2.0 && mass == 1.0
        ));
        assert!(bh.g_tt(2.0).is_err());
        assert!(bh.components(2.0, FRAC_PI_2).is_err());
    }

    #[test]
    fn rejects_bad_mass_radius_angle() {
        assert!(Schwarzschild::new(-1.0).is_err());
        assert!(Schwarzschild::new(f64::NAN).is_err());
        assert!(Schwarzschild::new(f64::INFINITY).is_err());
        let bh = mass_one();
        assert!(bh.g_tt(0.0).is_err());
        assert!(bh.g_tt(-4.0).is_err());
        assert!(bh.g_phi_phi(4.0, f64::NAN).is_err());
    }

    #[test]
    fn ds2_recovers_g_tt_for_pure_dt() {
        let bh = mass_one();
        let r = 6.0;
        let ds2 = bh.ds2(r, FRAC_PI_2, [1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!((ds2 - bh.g_tt(r).unwrap()).abs() < 1e-15);
    }
}
