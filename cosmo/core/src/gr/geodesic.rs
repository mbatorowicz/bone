//! Geodezyjna Schwarzschilda: stan `(t, r, θ, φ)` plus `dx^μ/dλ`.
//!
//! Równanie `d²x^μ/dλ² + Γ^μ_αβ u^α u^β = 0` jest układem pierwszego rzędu
//! na ośmiu liczbach. Krok robi [`crate::gr::rk4`], Γ bierze
//! [`crate::gr::christoffel`]. Tor jest w równiku albo poza nim — laboratorium
//! zrzucania przyjdzie później; tu liczymy, czy E, L i koło w `6M` trzymają
//! się wzoru.
//!
//! Zachowane: `E = (1 − 2M/r) dt/dλ` (energia Killinga) i
//! `L = r² sin²θ dφ/dλ` (moment). `κ = g_μν u^μ u^ν` też stoi: −1 czasowa,
//! 0 zerowa.

use std::fmt;

use super::christoffel::Christoffel;
use super::metric::{MetricError, Schwarzschild};
use super::rk4::{self, Scratch};

/// `y = (t, r, θ, φ, u^t, u^r, u^θ, u^φ)`.
pub const STATE_LEN: usize = 8;

/// Masa, promień albo długość stanu nie dają geodezyjnej.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeodesicError {
    Metric(MetricError),
    BadStateLen { len: usize },
    NoCircularOrbit { r: f64, mass: f64 },
}

impl fmt::Display for GeodesicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Metric(ref e) => write!(f, "{e}"),
            Self::BadStateLen { len } => write!(
                f,
                "stan geodezyjnej ma {len} składowych, potrzeba {STATE_LEN}"
            ),
            Self::NoCircularOrbit { r, mass } => write!(
                f,
                "brak orbity kołowej przy r={r}, M={mass} (czasowa wymaga r > 3M, zerowa tylko 3M)"
            ),
        }
    }
}

impl std::error::Error for GeodesicError {}

impl From<MetricError> for GeodesicError {
    fn from(value: MetricError) -> Self {
        Self::Metric(value)
    }
}

/// Współrzędne i czteroprędkość afiniczna.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicState {
    pub t: f64,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub u_t: f64,
    pub u_r: f64,
    pub u_theta: f64,
    pub u_phi: f64,
}

impl GeodesicState {
    pub fn to_array(self) -> [f64; STATE_LEN] {
        [
            self.t,
            self.r,
            self.theta,
            self.phi,
            self.u_t,
            self.u_r,
            self.u_theta,
            self.u_phi,
        ]
    }

    pub fn from_slice(y: &[f64]) -> Result<Self, GeodesicError> {
        let y: [f64; STATE_LEN] = y
            .try_into()
            .map_err(|_| GeodesicError::BadStateLen { len: y.len() })?;
        Ok(Self {
            t: y[0],
            r: y[1],
            theta: y[2],
            phi: y[3],
            u_t: y[4],
            u_r: y[5],
            u_theta: y[6],
            u_phi: y[7],
        })
    }

    pub fn four_velocity(self) -> [f64; 4] {
        [self.u_t, self.u_r, self.u_theta, self.u_phi]
    }

    /// `E = −u_t = (1 − 2M/r) u^t`.
    pub fn energy(self, bh: Schwarzschild) -> Result<f64, MetricError> {
        Ok(bh.f(self.r)? * self.u_t)
    }

    /// `L = u_φ = r² sin²θ u^φ`.
    pub fn angular_momentum(self, bh: Schwarzschild) -> Result<f64, MetricError> {
        Ok(bh.g_phi_phi(self.r, self.theta)? * self.u_phi)
    }

    /// `κ = g_μν u^μ u^ν`.
    pub fn u_sq(self, bh: Schwarzschild) -> Result<f64, MetricError> {
        bh.ds2(self.r, self.theta, self.four_velocity())
    }

    /// Kołowa orbita czasowa w równiku. Istnieje dla `r > 3M`.
    ///
    /// `u^t = √(r / (r − 3M))`, `u^φ = √(M / (r² (r − 3M)))`, reszta zero.
    /// Przy `r = 6M` to ISCO.
    pub fn circular_timelike(bh: Schwarzschild, r: f64) -> Result<Self, GeodesicError> {
        if !r.is_finite() || r <= 0.0 {
            return Err(MetricError::InvalidRadius { r }.into());
        }
        let mass = bh.mass();
        let denom = r - 3.0 * mass;
        if denom <= 0.0 {
            return Err(GeodesicError::NoCircularOrbit { r, mass });
        }
        bh.f(r)?;
        Ok(Self {
            t: 0.0,
            r,
            theta: std::f64::consts::FRAC_PI_2,
            phi: 0.0,
            u_t: (r / denom).sqrt(),
            u_r: 0.0,
            u_theta: 0.0,
            u_phi: (mass / (r * r * denom)).sqrt(),
        })
    }

    /// Kołowa orbita zerowa na sferze fotonowej `r = 3M`. Affine tak, by `E = 1`.
    pub fn circular_photon(bh: Schwarzschild) -> Result<Self, GeodesicError> {
        let mass = bh.mass();
        if mass == 0.0 {
            return Err(GeodesicError::NoCircularOrbit { r: 0.0, mass });
        }
        let r = 3.0 * mass;
        let f = bh.f(r)?;
        let energy = 1.0;
        let ell = 3.0 * 3.0_f64.sqrt() * mass;
        Ok(Self {
            t: 0.0,
            r,
            theta: std::f64::consts::FRAC_PI_2,
            phi: 0.0,
            u_t: energy / f,
            u_r: 0.0,
            u_theta: 0.0,
            u_phi: ell / (r * r),
        })
    }
}

/// `y' = (u, −Γ u u)`. `λ` nie wchodzi: metryka jest stacjonarna.
pub fn rhs(metric: Schwarzschild, y: &[f64], dy: &mut [f64]) -> Result<(), GeodesicError> {
    if y.len() != STATE_LEN {
        return Err(GeodesicError::BadStateLen { len: y.len() });
    }
    if dy.len() != STATE_LEN {
        return Err(GeodesicError::BadStateLen { len: dy.len() });
    }
    let state = GeodesicState::from_slice(y)?;
    let gamma = Christoffel::at(metric, state.r, state.theta)?;
    let a = gamma.accel(state.four_velocity());
    dy[0] = state.u_t;
    dy[1] = state.u_r;
    dy[2] = state.u_theta;
    dy[3] = state.u_phi;
    dy[4] = a[0];
    dy[5] = a[1];
    dy[6] = a[2];
    dy[7] = a[3];
    Ok(())
}

/// Jeden krok RK4 po parametrze afinicznym. Przy błędzie stan wraca do wejścia.
pub fn step(
    metric: Schwarzschild,
    lambda: f64,
    y: &mut [f64],
    h: f64,
    scratch: &mut Scratch,
) -> Result<(), GeodesicError> {
    if y.len() != STATE_LEN {
        return Err(GeodesicError::BadStateLen { len: y.len() });
    }
    let saved = [
        y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
    ];
    let mut err = None;
    rk4::step(lambda, y, h, scratch, |_lam, y, dy| {
        if err.is_some() {
            dy.fill(0.0);
            return;
        }
        if let Err(e) = rhs(metric, y, dy) {
            err = Some(e);
            dy.fill(0.0);
        }
    });
    if let Some(e) = err {
        y.copy_from_slice(&saved);
        return Err(e);
    }
    Ok(())
}

/// `n` kroków od `y0`. Zwraca stan po `n · h`.
pub fn integrate(
    metric: Schwarzschild,
    lambda0: f64,
    y0: &[f64],
    h: f64,
    n: usize,
) -> Result<Vec<f64>, GeodesicError> {
    let mut y = y0.to_vec();
    let mut scratch = Scratch::with_len(y.len());
    let mut lambda = lambda0;
    for _ in 0..n {
        step(metric, lambda, &mut y, h, &mut scratch)?;
        lambda += h;
    }
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, TAU};

    fn mass_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    fn cartesian(s: GeodesicState) -> [f64; 3] {
        let st = s.theta.sin();
        [
            s.r * st * s.phi.cos(),
            s.r * st * s.phi.sin(),
            s.r * s.theta.cos(),
        ]
    }

    fn equatorial(r: f64, u_t: f64, u_r: f64, u_phi: f64) -> GeodesicState {
        GeodesicState {
            t: 0.0,
            r,
            theta: FRAC_PI_2,
            phi: 0.0,
            u_t,
            u_r,
            u_theta: 0.0,
            u_phi,
        }
    }

    fn equatorial_timelike_turning(bh: Schwarzschild, r: f64, ell: f64) -> GeodesicState {
        let f = bh.f(r).unwrap();
        let u_phi = ell / (r * r);
        let u_t = ((1.0 + r * r * u_phi * u_phi) / f).sqrt();
        equatorial(r, u_t, 0.0, u_phi)
    }

    #[test]
    fn circular_timelike_matches_textbook_e_and_l() {
        let bh = mass_one();
        let s = GeodesicState::circular_timelike(bh, 6.0).unwrap();
        let energy = s.energy(bh).unwrap();
        let ell = s.angular_momentum(bh).unwrap();
        let want_e = 4.0 / 18.0_f64.sqrt();
        let want_l = 12.0_f64.sqrt();
        assert!(
            (energy - want_e).abs() < 1e-15,
            "E = {energy}, wzór {want_e}"
        );
        assert!((ell - want_l).abs() < 1e-15, "L = {ell}, wzór {want_l}");
        assert!((s.u_sq(bh).unwrap() + 1.0).abs() < 1e-14);
        let omega = s.u_phi / s.u_t;
        assert!(
            (omega - (1.0 / 216.0_f64).sqrt()).abs() < 1e-15,
            "Ω = {omega}"
        );
    }

    #[test]
    fn circular_photon_has_critical_impact_parameter() {
        let bh = mass_one();
        let s = GeodesicState::circular_photon(bh).unwrap();
        assert!((s.r - 3.0).abs() < 1e-15);
        let energy = s.energy(bh).unwrap();
        let ell = s.angular_momentum(bh).unwrap();
        assert!((energy - 1.0).abs() < 1e-15);
        assert!(
            (ell - 3.0 * 3.0_f64.sqrt()).abs() < 1e-15,
            "L/E = {}, wzór 3√3",
            ell / energy
        );
        assert!(s.u_sq(bh).unwrap().abs() < 1e-14);
    }

    #[test]
    fn circular_four_velocity_has_zero_accel() {
        let bh = mass_one();
        for s in [
            GeodesicState::circular_timelike(bh, 6.0).unwrap(),
            GeodesicState::circular_timelike(bh, 10.0).unwrap(),
            GeodesicState::circular_photon(bh).unwrap(),
        ] {
            let a = bh.christoffel(s.r, s.theta).unwrap().accel(s.four_velocity());
            assert!(
                a.iter().all(|v| v.abs() < 1e-14),
                "a={a:?} przy r={}",
                s.r
            );
        }
    }

    #[test]
    fn circular_requires_outside_photon_sphere() {
        let bh = mass_one();
        assert!(matches!(
            GeodesicState::circular_timelike(bh, 3.0),
            Err(GeodesicError::NoCircularOrbit { r, mass }) if r == 3.0 && mass == 1.0
        ));
        assert!(GeodesicState::circular_timelike(bh, 2.5).is_err());
        assert!(GeodesicState::circular_photon(Schwarzschild::new(0.0).unwrap()).is_err());
    }

    #[test]
    fn flat_radial_is_uniform_straight_line() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let u_r: f64 = 0.25;
        let u_t = (1.0 + u_r * u_r).sqrt();
        let start = equatorial(2.0, u_t, u_r, 0.0);
        assert!((start.u_sq(flat).unwrap() + 1.0).abs() < 1e-14);
        let h = 0.05;
        let n = 80;
        let y = integrate(flat, 0.0, &start.to_array(), h, n).unwrap();
        let end = GeodesicState::from_slice(&y).unwrap();
        let lam = h * n as f64;
        assert!(
            (end.r - (2.0 + u_r * lam)).abs() < 1e-10,
            "r = {}, prosta {}",
            end.r,
            2.0 + u_r * lam
        );
        assert!(end.phi.abs() < 1e-14);
        assert!((end.theta - FRAC_PI_2).abs() < 1e-14);
        let xyz = cartesian(end);
        assert!((xyz[0] - end.r).abs() < 1e-12);
        assert!(xyz[1].abs() < 1e-12 && xyz[2].abs() < 1e-12);
    }

    #[test]
    fn flat_offset_photon_stays_cartesian_straight() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let b = 4.0;
        let start = equatorial(b, 1.0, 0.0, 1.0 / b);
        assert!(start.u_sq(flat).unwrap().abs() < 1e-14);
        let lam = 5.0;
        let n = 500;
        let h = lam / n as f64;
        let y = integrate(flat, 0.0, &start.to_array(), h, n).unwrap();
        let end = GeodesicState::from_slice(&y).unwrap();
        let xyz = cartesian(end);
        const TOL: f64 = 1e-8;
        assert!(
            (xyz[0] - b).abs() < TOL,
            "x = {}, prosta x = {b}, |Δ| = {}",
            xyz[0],
            (xyz[0] - b).abs()
        );
        assert!(
            (xyz[1] - lam).abs() < TOL,
            "y = {}, prosta y = {lam}, |Δ| = {}",
            xyz[1],
            (xyz[1] - lam).abs()
        );
        assert!(xyz[2].abs() < TOL, "z = {}", xyz[2]);
        assert!((end.t - lam).abs() < TOL);
    }

    #[test]
    fn energy_and_angular_momentum_drift_is_measured() {
        let bh = mass_one();
        let start = equatorial_timelike_turning(bh, 12.0, 3.5);
        let e0 = start.energy(bh).unwrap();
        let l0 = start.angular_momentum(bh).unwrap();
        let k0 = start.u_sq(bh).unwrap();
        let mut y = start.to_array();
        let mut scratch = Scratch::with_len(STATE_LEN);
        let h = 0.05;
        let mut max_de = 0.0_f64;
        let mut max_dl = 0.0_f64;
        let mut max_dk = 0.0_f64;
        let mut r_min = start.r;
        let mut r_max = start.r;
        for _ in 0..1600 {
            step(bh, 0.0, &mut y, h, &mut scratch).unwrap();
            let s = GeodesicState::from_slice(&y).unwrap();
            max_de = max_de.max((s.energy(bh).unwrap() - e0).abs());
            max_dl = max_dl.max((s.angular_momentum(bh).unwrap() - l0).abs());
            max_dk = max_dk.max((s.u_sq(bh).unwrap() - k0).abs());
            r_min = r_min.min(s.r);
            r_max = r_max.max(s.r);
        }
        const TOL_E: f64 = 1e-8;
        const TOL_L: f64 = 1e-8;
        const TOL_K: f64 = 1e-8;
        assert!(
            max_de < TOL_E,
            "max |ΔE| = {max_de}, tolerancja {TOL_E} (E₀ = {e0})"
        );
        assert!(
            max_dl < TOL_L,
            "max |ΔL| = {max_dl}, tolerancja {TOL_L} (L₀ = {l0})"
        );
        assert!(
            max_dk < TOL_K,
            "max |Δκ| = {max_dk}, tolerancja {TOL_K} (κ₀ = {k0})"
        );
        assert!(
            r_max > r_min + 0.5,
            "orbita miała oddychać w r, dostała [{r_min}, {r_max}]"
        );
        assert!(r_min > 2.0, "r weszło pod horyzont: {r_min}");
    }

    #[test]
    fn circular_isco_stays_at_six_m() {
        let bh = mass_one();
        let start = GeodesicState::circular_timelike(bh, 6.0).unwrap();
        let e0 = start.energy(bh).unwrap();
        let l0 = start.angular_momentum(bh).unwrap();
        let period = TAU / start.u_phi;
        let n = 1600;
        let h = period / n as f64;
        let y = integrate(bh, 0.0, &start.to_array(), h, n).unwrap();
        let end = GeodesicState::from_slice(&y).unwrap();
        const TOL_R: f64 = 1e-7;
        const TOL_E: f64 = 1e-9;
        const TOL_PHI: f64 = 1e-6;
        let dr = (end.r - 6.0).abs();
        let de = (end.energy(bh).unwrap() - e0).abs();
        let dl = (end.angular_momentum(bh).unwrap() - l0).abs();
        let dphi = (end.phi - TAU).abs();
        assert!(dr < TOL_R, "|Δr| = {dr} po jednym okrążeniu, tolerancja {TOL_R}");
        assert!(de < TOL_E, "|ΔE| = {de}, tolerancja {TOL_E}");
        assert!(dl < TOL_E, "|ΔL| = {dl}, tolerancja {TOL_E}");
        assert!(dphi < TOL_PHI, "φ = {}, chcemy 2π, |Δ| = {dphi}", end.phi);
        assert!(end.u_r.abs() < 1e-7, "u^r = {}", end.u_r);
        assert!((end.theta - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn photon_sphere_stays_at_three_m() {
        let bh = mass_one();
        let start = GeodesicState::circular_photon(bh).unwrap();
        let e0 = start.energy(bh).unwrap();
        let l0 = start.angular_momentum(bh).unwrap();
        let period = TAU / start.u_phi;
        let n = 2000;
        let h = period / n as f64;
        let y = integrate(bh, 0.0, &start.to_array(), h, n).unwrap();
        let end = GeodesicState::from_slice(&y).unwrap();
        const TOL_R: f64 = 1e-6;
        const TOL_E: f64 = 1e-9;
        const TOL_PHI: f64 = 1e-5;
        let dr = (end.r - 3.0).abs();
        let de = (end.energy(bh).unwrap() - e0).abs();
        let dl = (end.angular_momentum(bh).unwrap() - l0).abs();
        let dphi = (end.phi - TAU).abs();
        assert!(
            dr < TOL_R,
            "|Δr| = {dr} na sferze fotonowej po okrążeniu, tolerancja {TOL_R}"
        );
        assert!(de < TOL_E, "|ΔE| = {de}, tolerancja {TOL_E}");
        assert!(dl < TOL_E, "|ΔL| = {dl}, tolerancja {TOL_E}");
        assert!(dphi < TOL_PHI, "φ = {}, chcemy 2π, |Δ| = {dphi}", end.phi);
        assert!(end.u_sq(bh).unwrap().abs() < 1e-9);
    }

    #[test]
    fn rejects_wrong_state_len() {
        let bh = mass_one();
        let mut dy = [0.0; STATE_LEN];
        assert!(matches!(
            rhs(bh, &[1.0, 2.0], &mut dy),
            Err(GeodesicError::BadStateLen { len: 2 })
        ));
        let mut y = vec![6.0; 3];
        let mut scratch = Scratch::new();
        assert!(matches!(
            step(bh, 0.0, &mut y, 0.01, &mut scratch),
            Err(GeodesicError::BadStateLen { len: 3 })
        ));
    }

    #[test]
    fn step_failure_restores_state() {
        let bh = mass_one();
        let mut y = equatorial(2.0, 1.0, 0.0, 0.0).to_array();
        let saved = y;
        let mut scratch = Scratch::with_len(STATE_LEN);
        assert!(step(bh, 0.0, &mut y, 0.01, &mut scratch).is_err());
        assert_eq!(y, saved);
    }
}
