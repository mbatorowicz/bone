//! Metryka Kerra w Boyer-Lindquist jako liczby, w jednostkach `G = c = 1`.
//!
//! To linijka i zegar w punkcie, nie tor. Spin `a` miesza czas z azymutem:
//! pojawia się kratka `g_tφ`. Gdy `a = 0`, kratka gaśnie i zostaje
//! Schwarzschild. `|a| > M` ta mata nie umie — nadmierny spin to błąd,
//! nie ekstremalna dziura (`|a| = M` jest kreską).
//!
//! Sygnatura (−,+,+,+):
//!
//! ```text
//! Δ = r² − 2 M r + a²
//! Σ = r² + a² cos²θ
//! g_tt  = −(1 − 2 M r / Σ)
//! g_tφ  = −2 M a r sin²θ / Σ
//! g_rr  = Σ / Δ
//! g_θθ  = Σ
//! g_φφ  = (r² + a² + 2 M r a² sin²θ / Σ) sin²θ
//! r+    = M + √(M² − a²)
//! ```
//!
//! Γ, geodezyjna i raytracer nie wchodzą. Horyzont wewnętrzny (`r−`) też nie.

use std::fmt;

/// Masa nie jest liczbą, spin rozsadza matę, promień nie jest miejscem,
/// albo stoimy tam, gdzie `Δ = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KerrError {
    InvalidMass { mass: f64 },
    InvalidSpin { mass: f64, spin: f64 },
    InvalidRadius { r: f64 },
    InvalidAngle { theta: f64 },
    AtHorizon { r: f64, mass: f64, spin: f64 },
}

impl fmt::Display for KerrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidMass { mass } => {
                write!(f, "masa M={mass} musi być skończona i ≥ 0")
            }
            Self::InvalidSpin { mass, spin } => {
                write!(f, "spin a={spin} musi być skończony i |a| ≤ M (M={mass})")
            }
            Self::InvalidRadius { r } => {
                write!(f, "promień r={r} musi być skończony i > 0")
            }
            Self::InvalidAngle { theta } => {
                write!(f, "kąt θ={theta} musi być skończony")
            }
            Self::AtHorizon { r, mass, spin } => write!(
                f,
                "współrzędne Boyer-Lindquista łamią się przy Δ=0 (r={r}, M={mass}, a={spin})"
            ),
        }
    }
}

impl std::error::Error for KerrError {}

/// Masa i spin. `spin = 0` to Schwarzschild w tych samych współrzędnych.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kerr {
    mass: f64,
    spin: f64,
}

impl Kerr {
    pub fn new(mass: f64, spin: f64) -> Result<Self, KerrError> {
        if !mass.is_finite() || mass < 0.0 {
            return Err(KerrError::InvalidMass { mass });
        }
        if !spin.is_finite() || spin.abs() > mass {
            return Err(KerrError::InvalidSpin { mass, spin });
        }
        Ok(Self { mass, spin })
    }

    pub fn mass(self) -> f64 {
        self.mass
    }

    pub fn spin(self) -> f64 {
        self.spin
    }

    /// `r+ = M + √(M² − a²)`. Przy `a = 0` to `2M`.
    pub fn horizon_radius(self) -> f64 {
        self.mass + disc(self.mass, self.spin)
    }

    /// Zewnętrzna ergosphera: `M + √(M² − a² cos²θ)`.
    ///
    /// Na równiku zawsze `2M`. Na biegunach klei się do `r+`. Przy `a = 0`
    /// to znowu `2M`.
    pub fn ergo(self, theta: f64) -> Result<f64, KerrError> {
        check_theta(theta)?;
        let a_cos = self.spin * theta.cos();
        Ok(self.mass + disc(self.mass, a_cos))
    }

    /// Sfera fotonowa współobrotu (bliżej studni). Przy `a = 0` to `3M`.
    pub fn photon_plus(self) -> f64 {
        self.photon_orbit(true)
    }

    /// Sfera fotonowa przeciwobrotu (dalej od studni). Przy `a = 0` to `3M`.
    pub fn photon_minus(self) -> f64 {
        self.photon_orbit(false)
    }

    /// ISCO współobrotu. Przy `a = 0` to `6M`; przy spinie ciaśniejsze.
    pub fn isco_plus(self) -> f64 {
        self.isco_orbit(true)
    }

    /// ISCO przeciwobrotu. Przy `a = 0` to `6M`; przy spinie szersze.
    pub fn isco_minus(self) -> f64 {
        self.isco_orbit(false)
    }

    /// `Δ = r² − 2 M r + a²`. Zero na horyzoncie.
    pub fn delta(self, r: f64) -> Result<f64, KerrError> {
        check_r(r)?;
        Ok(r * r - 2.0 * self.mass * r + self.spin * self.spin)
    }

    /// `Σ = r² + a² cos²θ`.
    pub fn sigma(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        check_r(r)?;
        check_theta(theta)?;
        let a_cos = self.spin * theta.cos();
        Ok(r * r + a_cos * a_cos)
    }

    /// `g_tt = −(1 − 2 M r / Σ)`. Zero na ergosferze, nie na `r+`.
    pub fn g_tt(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        let sigma = self.sigma(r, theta)?;
        Ok(-(1.0 - 2.0 * self.mass * r / sigma))
    }

    /// `g_tφ = −2 M a r sin²θ / Σ`. Gaśnie przy `a = 0`.
    pub fn g_t_phi(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        let sigma = self.sigma(r, theta)?;
        let sin = theta.sin();
        Ok(-2.0 * self.mass * self.spin * r * sin * sin / sigma)
    }

    /// `g_rr = Σ / Δ`. Łamie się przy `Δ = 0`.
    pub fn g_rr(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        let sigma = self.sigma(r, theta)?;
        let delta = self.delta_at_horizon(r)?;
        Ok(sigma / delta)
    }

    /// `g_θθ = Σ`.
    pub fn g_theta_theta(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        self.sigma(r, theta)
    }

    /// `g_φφ = (r² + a² + 2 M r a² sin²θ / Σ) sin²θ`.
    pub fn g_phi_phi(self, r: f64, theta: f64) -> Result<f64, KerrError> {
        let sigma = self.sigma(r, theta)?;
        let sin = theta.sin();
        let a2 = self.spin * self.spin;
        Ok((r * r + a2 + 2.0 * self.mass * r * a2 * sin * sin / sigma) * sin * sin)
    }

    /// Pełna macierz `g_μν` w kolejności `(t, r, θ, φ)`. Poza `g_tφ` zera.
    pub fn components(self, r: f64, theta: f64) -> Result<[[f64; 4]; 4], KerrError> {
        let mut g = [[0.0; 4]; 4];
        g[0][0] = self.g_tt(r, theta)?;
        g[1][1] = self.g_rr(r, theta)?;
        g[2][2] = self.g_theta_theta(r, theta)?;
        g[3][3] = self.g_phi_phi(r, theta)?;
        let mix = self.g_t_phi(r, theta)?;
        g[0][3] = mix;
        g[3][0] = mix;
        Ok(g)
    }

    /// `g^{μν}`: przekątna `r, θ` plus odwrotność bloku `t–φ`.
    pub fn inverse(self, r: f64, theta: f64) -> Result<[[f64; 4]; 4], KerrError> {
        let g = self.components(r, theta)?;
        let det = g[0][0] * g[3][3] - g[0][3] * g[3][0];
        if det == 0.0 || !det.is_finite() {
            return Err(KerrError::AtHorizon {
                r,
                mass: self.mass,
                spin: self.spin,
            });
        }
        let inv_rr = 1.0 / g[1][1];
        let inv_th = 1.0 / g[2][2];
        if !inv_rr.is_finite() || !inv_th.is_finite() {
            return Err(KerrError::AtHorizon {
                r,
                mass: self.mass,
                spin: self.spin,
            });
        }
        Ok([
            [g[3][3] / det, 0.0, 0.0, -g[0][3] / det],
            [0.0, inv_rr, 0.0, 0.0],
            [0.0, 0.0, inv_th, 0.0],
            [-g[3][0] / det, 0.0, 0.0, g[0][0] / det],
        ])
    }

    /// `ds² = g_μν dx^μ dx^ν`, w tym mieszany `2 g_tφ dt dφ`.
    pub fn ds2(self, r: f64, theta: f64, dx: [f64; 4]) -> Result<f64, KerrError> {
        let g = self.components(r, theta)?;
        Ok(g[0][0] * dx[0] * dx[0]
            + g[1][1] * dx[1] * dx[1]
            + g[2][2] * dx[2] * dx[2]
            + g[3][3] * dx[3] * dx[3]
            + 2.0 * g[0][3] * dx[0] * dx[3])
    }

    fn delta_at_horizon(self, r: f64) -> Result<f64, KerrError> {
        let delta = self.delta(r)?;
        if delta == 0.0 || !delta.is_finite() {
            Err(KerrError::AtHorizon {
                r,
                mass: self.mass,
                spin: self.spin,
            })
        } else {
            Ok(delta)
        }
    }

    /// `|a|/M` w `[0, 1]`. Zero masy to Minkowski, nie `0/0`.
    fn chi(self) -> f64 {
        if self.mass == 0.0 {
            0.0
        } else {
            (self.spin.abs() / self.mass).clamp(0.0, 1.0)
        }
    }

    /// `r_γ± = 2M [1 + cos((2/3) arccos(∓ a/M))]`. Plus = współobrót.
    fn photon_orbit(self, prograde: bool) -> f64 {
        let chi = self.chi();
        let arg = if prograde { -chi } else { chi };
        2.0 * self.mass * (1.0 + ((2.0 / 3.0) * arg.acos()).cos())
    }

    /// Bardeen–Press–Teukolsky. Plus = współobrót = minus pod pierwiastkiem.
    fn isco_orbit(self, prograde: bool) -> f64 {
        let chi = self.chi();
        let one_minus = (1.0 - chi * chi).max(0.0).cbrt();
        let z1 = 1.0 + one_minus * ((1.0 + chi).cbrt() + (1.0 - chi).cbrt());
        let z2 = (3.0 * chi * chi + z1 * z1).sqrt();
        let inner = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).max(0.0).sqrt();
        let r_over_m = if prograde {
            3.0 + z2 - inner
        } else {
            3.0 + z2 + inner
        };
        r_over_m * self.mass
    }
}

fn disc(mass: f64, spin_like: f64) -> f64 {
    (mass * mass - spin_like * spin_like).max(0.0).sqrt()
}

fn check_r(r: f64) -> Result<(), KerrError> {
    if !r.is_finite() || r <= 0.0 {
        Err(KerrError::InvalidRadius { r })
    } else {
        Ok(())
    }
}

fn check_theta(theta: f64) -> Result<(), KerrError> {
    if theta.is_finite() {
        Ok(())
    } else {
        Err(KerrError::InvalidAngle { theta })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gr::metric::{horizon_radius, isco_radius, photon_sphere_radius, Schwarzschild};
    use std::f64::consts::{FRAC_PI_2, PI};

    fn mass_one() -> Kerr {
        Kerr::new(1.0, 0.0).unwrap()
    }

    fn spinning() -> Kerr {
        Kerr::new(1.0, 0.5).unwrap()
    }

    fn schw_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    fn close(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() < eps, "{a} ≉ {b} (eps={eps})");
    }

    #[test]
    fn zero_spin_matches_schwarzschild_radii() {
        let m = 1.5;
        let bh = Kerr::new(m, 0.0).unwrap();
        close(bh.horizon_radius(), horizon_radius(m), 1e-15);
        close(bh.horizon_radius(), 2.0 * m, 1e-15);
        close(bh.photon_plus(), photon_sphere_radius(m), 1e-15);
        close(bh.photon_minus(), photon_sphere_radius(m), 1e-15);
        close(bh.isco_plus(), isco_radius(m), 1e-12);
        close(bh.isco_minus(), isco_radius(m), 1e-12);
        close(bh.ergo(FRAC_PI_2).unwrap(), 2.0 * m, 1e-15);
        close(bh.ergo(0.0).unwrap(), 2.0 * m, 1e-15);
    }

    #[test]
    fn zero_spin_metric_matches_schwarzschild() {
        let kerr = mass_one();
        let schw = schw_one();
        let r = 6.0;
        let theta = FRAC_PI_2;
        close(kerr.g_tt(r, theta).unwrap(), schw.g_tt(r).unwrap(), 1e-15);
        close(kerr.g_rr(r, theta).unwrap(), schw.g_rr(r).unwrap(), 1e-15);
        close(
            kerr.g_theta_theta(r, theta).unwrap(),
            schw.g_theta_theta(r).unwrap(),
            1e-15,
        );
        close(
            kerr.g_phi_phi(r, theta).unwrap(),
            schw.g_phi_phi(r, theta).unwrap(),
            1e-15,
        );
        assert_eq!(kerr.g_t_phi(r, theta).unwrap(), 0.0);
        assert_eq!(kerr.g_t_phi(r, 0.3).unwrap(), 0.0);
        let g = kerr.components(r, theta).unwrap();
        let gs = schw.components(r, theta).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                close(g[i][j], gs[i][j], 1e-15);
            }
        }
    }

    #[test]
    fn delta_sigma_and_g_t_phi_match_textbook() {
        let bh = spinning();
        let r = 4.0;
        close(bh.delta(r).unwrap(), 8.25, 1e-15);
        close(bh.sigma(r, FRAC_PI_2).unwrap(), 16.0, 1e-15);
        close(bh.g_t_phi(r, FRAC_PI_2).unwrap(), -0.25, 1e-15);
        close(bh.g_tt(r, FRAC_PI_2).unwrap(), -0.5, 1e-15);
        close(bh.g_phi_phi(r, FRAC_PI_2).unwrap(), 16.375, 1e-15);
        assert!(bh.g_t_phi(r, 0.0).unwrap().abs() < 1e-15);
        close(bh.sigma(r, 0.0).unwrap(), 16.25, 1e-15);
    }

    #[test]
    fn t_phi_block_determinant_is_minus_delta_sin2() {
        let bh = spinning();
        let r = 5.0;
        let theta = 0.7;
        let g = bh.components(r, theta).unwrap();
        let det = g[0][0] * g[3][3] - g[0][3] * g[3][0];
        let expected = -bh.delta(r).unwrap() * theta.sin().powi(2);
        close(det, expected, 1e-12);
        assert_eq!(g[0][3], g[3][0]);
        assert_eq!(g[0][1], 0.0);
        assert_eq!(g[0][2], 0.0);
        assert_eq!(g[1][2], 0.0);
        assert_eq!(g[1][3], 0.0);
        assert_eq!(g[2][3], 0.0);
    }

    #[test]
    fn ergo_equator_is_two_m_poles_meet_horizon() {
        let bh = spinning();
        close(bh.ergo(FRAC_PI_2).unwrap(), 2.0, 1e-15);
        close(bh.ergo(0.0).unwrap(), bh.horizon_radius(), 1e-15);
        close(bh.ergo(PI).unwrap(), bh.horizon_radius(), 1e-15);
        assert!(bh.horizon_radius() < 2.0);
        assert!(bh.horizon_radius() > 0.0);
    }

    #[test]
    fn spin_splits_photon_and_isco() {
        let bh = spinning();
        let six = isco_radius(1.0);
        let three = photon_sphere_radius(1.0);
        assert!(bh.isco_plus() < six);
        assert!(bh.isco_minus() > six);
        assert!(bh.photon_plus() < three);
        assert!(bh.photon_minus() > three);
        assert!(bh.photon_plus() > bh.horizon_radius());
        assert!(bh.isco_plus() > bh.photon_plus());
        assert!(bh.isco_minus() > bh.photon_minus());
    }

    #[test]
    fn extremal_limits() {
        let bh = Kerr::new(1.0, 1.0).unwrap();
        close(bh.horizon_radius(), 1.0, 1e-15);
        close(bh.photon_plus(), 1.0, 1e-12);
        close(bh.photon_minus(), 4.0, 1e-12);
        close(bh.isco_plus(), 1.0, 1e-12);
        close(bh.isco_minus(), 9.0, 1e-12);
        close(bh.ergo(FRAC_PI_2).unwrap(), 2.0, 1e-15);
        close(bh.ergo(0.0).unwrap(), 1.0, 1e-15);
    }

    #[test]
    fn negative_spin_flips_drag_not_radii() {
        let plus = Kerr::new(1.0, 0.5).unwrap();
        let minus = Kerr::new(1.0, -0.5).unwrap();
        close(plus.horizon_radius(), minus.horizon_radius(), 1e-15);
        close(plus.photon_plus(), minus.photon_plus(), 1e-15);
        close(plus.isco_minus(), minus.isco_minus(), 1e-15);
        close(
            plus.g_t_phi(4.0, FRAC_PI_2).unwrap(),
            -minus.g_t_phi(4.0, FRAC_PI_2).unwrap(),
            1e-15,
        );
    }

    #[test]
    fn inverse_undoes_the_metric() {
        let bh = spinning();
        let g = bh.components(8.0, 0.7).unwrap();
        let gi = bh.inverse(8.0, 0.7).unwrap();
        close(g[0][0] * gi[0][0] + g[0][3] * gi[3][0], 1.0, 1e-12);
        close(g[0][0] * gi[0][3] + g[0][3] * gi[3][3], 0.0, 1e-12);
        close(g[3][0] * gi[0][0] + g[3][3] * gi[3][0], 0.0, 1e-12);
        close(g[3][0] * gi[0][3] + g[3][3] * gi[3][3], 1.0, 1e-12);
        close(g[1][1] * gi[1][1], 1.0, 1e-12);
        close(g[2][2] * gi[2][2], 1.0, 1e-12);
        assert_eq!(gi[0][1], 0.0);
        assert_eq!(gi[0][2], 0.0);
        assert_eq!(gi[1][3], 0.0);
    }

    #[test]
    fn zero_mass_is_minkowski_in_sphericals() {
        let flat = Kerr::new(0.0, 0.0).unwrap();
        let r = 3.0;
        let theta = FRAC_PI_2;
        close(flat.g_tt(r, theta).unwrap(), -1.0, 1e-15);
        close(flat.g_rr(r, theta).unwrap(), 1.0, 1e-15);
        close(flat.g_theta_theta(r, theta).unwrap(), 9.0, 1e-15);
        close(flat.g_phi_phi(r, theta).unwrap(), 9.0, 1e-15);
        assert_eq!(flat.g_t_phi(r, theta).unwrap(), 0.0);
        assert_eq!(flat.horizon_radius(), 0.0);
        assert_eq!(flat.photon_plus(), 0.0);
        assert_eq!(flat.isco_plus(), 0.0);
    }

    #[test]
    fn ds2_sees_frame_dragging() {
        let bh = spinning();
        let r = 4.0;
        let theta = FRAC_PI_2;
        let mix = bh.ds2(r, theta, [1.0, 0.0, 0.0, 1.0]).unwrap();
        let expected = bh.g_tt(r, theta).unwrap()
            + bh.g_phi_phi(r, theta).unwrap()
            + 2.0 * bh.g_t_phi(r, theta).unwrap();
        close(mix, expected, 1e-15);
        let dt = bh.ds2(r, theta, [1.0, 0.0, 0.0, 0.0]).unwrap();
        close(dt, bh.g_tt(r, theta).unwrap(), 1e-15);
    }

    #[test]
    fn horizon_rejects_g_rr_not_g_tt() {
        let bh = mass_one();
        assert!(matches!(
            bh.g_rr(2.0, FRAC_PI_2),
            Err(KerrError::AtHorizon { r, mass, spin })
                if r == 2.0 && mass == 1.0 && spin == 0.0
        ));
        assert!(bh.g_tt(2.0, FRAC_PI_2).is_ok());
        assert!(bh.g_t_phi(2.0, FRAC_PI_2).is_ok());
        assert!(bh.components(2.0, FRAC_PI_2).is_err());
        assert_eq!(bh.delta(2.0).unwrap(), 0.0);
    }

    #[test]
    fn rejects_super_extremal_and_bad_args() {
        assert!(Kerr::new(-1.0, 0.0).is_err());
        assert!(Kerr::new(f64::NAN, 0.0).is_err());
        assert!(Kerr::new(1.0, 1.1).is_err());
        assert!(Kerr::new(1.0, -1.1).is_err());
        assert!(Kerr::new(0.0, 1e-9).is_err());
        assert!(Kerr::new(1.0, f64::NAN).is_err());
        assert!(Kerr::new(1.0, f64::INFINITY).is_err());
        assert!(Kerr::new(1.0, 1.0).is_ok());
        let bh = spinning();
        assert!(bh.g_tt(0.0, FRAC_PI_2).is_err());
        assert!(bh.g_tt(-4.0, FRAC_PI_2).is_err());
        assert!(bh.g_t_phi(4.0, f64::NAN).is_err());
        assert!(bh.ergo(f64::NAN).is_err());
    }
}
