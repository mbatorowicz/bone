//! Metryka Kerra w Boyer-Lindquist, analityczne Γ i geodezyjna, `G = c = 1`.
//!
//! To linijka i zegar w punkcie, plus przyspieszenie `a^μ = −Γ^μ_αβ u^α u^β`.
//! Spin `a` miesza czas z azymutem: pojawia się kratka `g_tφ`, więc Γ nie
//! mieści się w [`super::christoffel::Christoffel`] — tam `accel` nie zna
//! mieszanych `Γ^t_rφ` / `Γ^r_tφ`. Gdy `a = 0`, kratka gaśnie, Γ zbiega do
//! Schwarzschilda, a koło `6M` wraca. `|a| > M` ta mata nie umie.
//!
//! Stan geodezyjnej to te same 8 liczb co w [`super::geodesic`]:
//! `(t, r, θ, φ, u^t, u^r, u^θ, u^φ)`. Krok robi [`super::rk4`].
//! Zachowane: `E = −u_t` i `L_z = u_φ`. `r−` nie wchodzi; obraz pikseli
//! jest w [`super::raytrace`] (`Config.spin`).
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

use std::fmt;

use super::geodesic::{GeodesicState, STATE_LEN};
use super::rk4::{self, Scratch};

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

/// Masa, spin, promień albo długość stanu nie dają geodezyjnej Kerra.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KerrGeoError {
    Metric(KerrError),
    BadStateLen { len: usize },
    NoCircularOrbit { r: f64, mass: f64, spin: f64 },
}

impl fmt::Display for KerrGeoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Metric(ref e) => write!(f, "{e}"),
            Self::BadStateLen { len } => write!(
                f,
                "stan geodezyjnej ma {len} składowych, potrzeba {STATE_LEN}"
            ),
            Self::NoCircularOrbit { r, mass, spin } => write!(
                f,
                "brak orbity kołowej przy r={r}, M={mass}, a={spin} (czasowa wymaga r poza sferą fotonową)"
            ),
        }
    }
}

impl std::error::Error for KerrGeoError {}

impl From<KerrError> for KerrGeoError {
    fn from(value: KerrError) -> Self {
        Self::Metric(value)
    }
}

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

    /// Analityczne Γ w `(r, θ)`. Czas i `φ` nie wchodzą: Kerr jest stacjonarny.
    pub fn christoffel(self, r: f64, theta: f64) -> Result<KerrChristoffel, KerrError> {
        KerrChristoffel::at(self, r, theta)
    }

    /// `E = −u_t = −g_tt u^t − g_tφ u^φ`. Przy `a = 0` to `(1 − 2M/r) u^t`.
    pub fn energy(self, s: GeodesicState) -> Result<f64, KerrError> {
        Ok(-self.g_tt(s.r, s.theta)? * s.u_t - self.g_t_phi(s.r, s.theta)? * s.u_phi)
    }

    /// `L_z = u_φ = g_tφ u^t + g_φφ u^φ`. Przy `a = 0` to `r² sin²θ u^φ`.
    pub fn angular_momentum(self, s: GeodesicState) -> Result<f64, KerrError> {
        Ok(self.g_t_phi(s.r, s.theta)? * s.u_t + self.g_phi_phi(s.r, s.theta)? * s.u_phi)
    }

    /// `κ = g_μν u^μ u^ν`.
    pub fn u_sq(self, s: GeodesicState) -> Result<f64, KerrError> {
        self.ds2(s.r, s.theta, s.four_velocity())
    }

    /// Kołowa orbita czasowa w równiku. `prograde` = współobrót (`Ω > 0` przy `a ≥ 0`).
    ///
    /// `Ω = ±√M / (r^{3/2} ± a √M)`, potem `u^t` z `g_μν u^μ u^ν = −1`.
    /// Przy `a = 0` to koło Schwarzschilda (`r > 3M`, ISCO przy `6M`).
    pub fn circular_timelike(self, r: f64, prograde: bool) -> Result<GeodesicState, KerrGeoError> {
        check_r(r)?;
        let mass = self.mass;
        let spin = self.spin;
        let sqrt_m = mass.sqrt();
        let r32 = r.powf(1.5);
        let denom = if prograde {
            r32 + spin * sqrt_m
        } else {
            r32 - spin * sqrt_m
        };
        if !denom.is_finite() || denom == 0.0 {
            return Err(KerrGeoError::NoCircularOrbit { r, mass, spin });
        }
        let omega = if prograde {
            sqrt_m / denom
        } else {
            -sqrt_m / denom
        };
        if !omega.is_finite() {
            return Err(KerrGeoError::NoCircularOrbit { r, mass, spin });
        }
        let theta = std::f64::consts::FRAC_PI_2;
        let gtt = self.g_tt(r, theta)?;
        let gtphi = self.g_t_phi(r, theta)?;
        let gphiphi = self.g_phi_phi(r, theta)?;
        let norm = gtt + 2.0 * gtphi * omega + gphiphi * omega * omega;
        if !(norm < 0.0) || !norm.is_finite() {
            return Err(KerrGeoError::NoCircularOrbit { r, mass, spin });
        }
        let u_t = 1.0 / (-norm).sqrt();
        if !u_t.is_finite() {
            return Err(KerrGeoError::NoCircularOrbit { r, mass, spin });
        }
        Ok(GeodesicState {
            t: 0.0,
            r,
            theta,
            phi: 0.0,
            u_t,
            u_r: 0.0,
            u_theta: 0.0,
            u_phi: omega * u_t,
        })
    }
}

/// Niezależne niezerowe `Γ^μ_αβ` Kerra. Reszta to zera albo odbicia `α ↔ β`.
///
/// `Christoffel` Schwarzschilda nie zna `g_tφ`, więc nie ma `Γ^t_rφ`
/// ani `Γ^r_tφ`. Tutaj mieszane wchodzą do `accel` z czynnikiem 2,
/// tak samo jak `2 Γ^t_tr u^t u^r` na starej macie.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KerrChristoffel {
    /// `Γ^t_tr = Γ^t_rt`.
    pub gamma_t_tr: f64,
    /// `Γ^t_tθ = Γ^t_θt`.
    pub gamma_t_t_theta: f64,
    /// `Γ^t_rφ = Γ^t_φr` — wleczenie, gaśnie przy `a = 0`.
    pub gamma_t_r_phi: f64,
    /// `Γ^t_θφ = Γ^t_φθ`.
    pub gamma_t_theta_phi: f64,
    /// `Γ^r_tt`.
    pub gamma_r_tt: f64,
    /// `Γ^r_tφ = Γ^r_φt` — wleczenie, gaśnie przy `a = 0`.
    pub gamma_r_t_phi: f64,
    /// `Γ^r_rr`.
    pub gamma_r_rr: f64,
    /// `Γ^r_rθ = Γ^r_θr`.
    pub gamma_r_r_theta: f64,
    /// `Γ^r_θθ`.
    pub gamma_r_theta_theta: f64,
    /// `Γ^r_φφ`.
    pub gamma_r_phi_phi: f64,
    /// `Γ^θ_tt`.
    pub gamma_theta_tt: f64,
    /// `Γ^θ_tφ = Γ^θ_φt`.
    pub gamma_theta_t_phi: f64,
    /// `Γ^θ_rr`.
    pub gamma_theta_rr: f64,
    /// `Γ^θ_rθ = Γ^θ_θr`.
    pub gamma_theta_r_theta: f64,
    /// `Γ^θ_θθ`.
    pub gamma_theta_theta_theta: f64,
    /// `Γ^θ_φφ`.
    pub gamma_theta_phi_phi: f64,
    /// `Γ^φ_tr = Γ^φ_rt` — wleczenie, gaśnie przy `a = 0`.
    pub gamma_phi_tr: f64,
    /// `Γ^φ_tθ = Γ^φ_θt`.
    pub gamma_phi_t_theta: f64,
    /// `Γ^φ_rφ = Γ^φ_φr`.
    pub gamma_phi_r_phi: f64,
    /// `Γ^φ_θφ = Γ^φ_φθ`.
    pub gamma_phi_theta_phi: f64,
}

impl KerrChristoffel {
    /// Γ w punkcie `(r, θ)` ze wzoru `½ g^{σρ}(∂_μ g_ρν + ∂_ν g_ρμ − ∂_ρ g_μν)`.
    ///
    /// Pochodne `g_μν` są zamknięte (tylko `r` i `θ`). Na horyzoncie `Δ = 0`
    /// te Γ rozjeżdżają się razem z `g_rr`.
    pub fn at(bh: Kerr, r: f64, theta: f64) -> Result<Self, KerrError> {
        check_theta(theta)?;
        let sin = theta.sin();
        if sin.abs() < f64::EPSILON {
            return Err(KerrError::InvalidAngle { theta });
        }
        let gi = bh.inverse(r, theta)?;
        let (dg_r, dg_th) = metric_partials(bh, r, theta)?;
        let g = |sigma: usize, mu: usize, nu: usize| gamma_comp(&gi, &dg_r, &dg_th, sigma, mu, nu);
        Ok(Self {
            gamma_t_tr: g(0, 0, 1),
            gamma_t_t_theta: g(0, 0, 2),
            gamma_t_r_phi: g(0, 1, 3),
            gamma_t_theta_phi: g(0, 2, 3),
            gamma_r_tt: g(1, 0, 0),
            gamma_r_t_phi: g(1, 0, 3),
            gamma_r_rr: g(1, 1, 1),
            gamma_r_r_theta: g(1, 1, 2),
            gamma_r_theta_theta: g(1, 2, 2),
            gamma_r_phi_phi: g(1, 3, 3),
            gamma_theta_tt: g(2, 0, 0),
            gamma_theta_t_phi: g(2, 0, 3),
            gamma_theta_rr: g(2, 1, 1),
            gamma_theta_r_theta: g(2, 1, 2),
            gamma_theta_theta_theta: g(2, 2, 2),
            gamma_theta_phi_phi: g(2, 3, 3),
            gamma_phi_tr: g(3, 0, 1),
            gamma_phi_t_theta: g(3, 0, 2),
            gamma_phi_r_phi: g(3, 1, 3),
            gamma_phi_theta_phi: g(3, 2, 3),
        })
    }

    /// `a^μ = −Γ^μ_αβ u^α u^β` dla `u = (u^t, u^r, u^θ, u^φ)`.
    pub fn accel(self, u: [f64; 4]) -> [f64; 4] {
        let [u_t, u_r, u_theta, u_phi] = u;
        let a_t = -(2.0 * self.gamma_t_tr * u_t * u_r
            + 2.0 * self.gamma_t_t_theta * u_t * u_theta
            + 2.0 * self.gamma_t_r_phi * u_r * u_phi
            + 2.0 * self.gamma_t_theta_phi * u_theta * u_phi);
        let a_r = -(self.gamma_r_tt * u_t * u_t
            + 2.0 * self.gamma_r_t_phi * u_t * u_phi
            + self.gamma_r_rr * u_r * u_r
            + 2.0 * self.gamma_r_r_theta * u_r * u_theta
            + self.gamma_r_theta_theta * u_theta * u_theta
            + self.gamma_r_phi_phi * u_phi * u_phi);
        let a_theta = -(self.gamma_theta_tt * u_t * u_t
            + 2.0 * self.gamma_theta_t_phi * u_t * u_phi
            + self.gamma_theta_rr * u_r * u_r
            + 2.0 * self.gamma_theta_r_theta * u_r * u_theta
            + self.gamma_theta_theta_theta * u_theta * u_theta
            + self.gamma_theta_phi_phi * u_phi * u_phi);
        let a_phi = -(2.0 * self.gamma_phi_tr * u_t * u_r
            + 2.0 * self.gamma_phi_t_theta * u_t * u_theta
            + 2.0 * self.gamma_phi_r_phi * u_r * u_phi
            + 2.0 * self.gamma_phi_theta_phi * u_theta * u_phi);
        [a_t, a_r, a_theta, a_phi]
    }
}

/// `y' = (u, −Γ u u)`. `λ` nie wchodzi: Kerr jest stacjonarny.
pub fn rhs(metric: Kerr, y: &[f64], dy: &mut [f64]) -> Result<(), KerrGeoError> {
    if y.len() != STATE_LEN {
        return Err(KerrGeoError::BadStateLen { len: y.len() });
    }
    if dy.len() != STATE_LEN {
        return Err(KerrGeoError::BadStateLen { len: dy.len() });
    }
    let state =
        GeodesicState::from_slice(y).map_err(|_| KerrGeoError::BadStateLen { len: y.len() })?;
    let gamma = KerrChristoffel::at(metric, state.r, state.theta)?;
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
    metric: Kerr,
    lambda: f64,
    y: &mut [f64],
    h: f64,
    scratch: &mut Scratch,
) -> Result<(), KerrGeoError> {
    if y.len() != STATE_LEN {
        return Err(KerrGeoError::BadStateLen { len: y.len() });
    }
    let saved = [y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]];
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
    metric: Kerr,
    lambda0: f64,
    y0: &[f64],
    h: f64,
    n: usize,
) -> Result<Vec<f64>, KerrGeoError> {
    let mut y = y0.to_vec();
    let mut scratch = Scratch::with_len(y.len());
    let mut lambda = lambda0;
    for _ in 0..n {
        step(metric, lambda, &mut y, h, &mut scratch)?;
        lambda += h;
    }
    Ok(y)
}

fn disc(mass: f64, spin_like: f64) -> f64 {
    (mass * mass - spin_like * spin_like).max(0.0).sqrt()
}

/// Analityczne `∂_r g_μν` i `∂_θ g_μν`. `∂_t` i `∂_φ` są zerem.
fn metric_partials(
    bh: Kerr,
    r: f64,
    theta: f64,
) -> Result<([[f64; 4]; 4], [[f64; 4]; 4]), KerrError> {
    let m = bh.mass;
    let a = bh.spin;
    let s = theta.sin();
    let c = theta.cos();
    let s2 = s * s;
    let a2 = a * a;
    let sigma = bh.sigma(r, theta)?;
    let sigma2 = sigma * sigma;
    let delta = bh.delta_at_horizon(r)?;
    let dsigma_r = 2.0 * r;
    let dsigma_th = -2.0 * a2 * c * s;
    let ddelta_r = 2.0 * (r - m);

    let mut dg_r = [[0.0; 4]; 4];
    let mut dg_th = [[0.0; 4]; 4];

    dg_r[0][0] = 2.0 * m * (sigma - 2.0 * r * r) / sigma2;
    dg_th[0][0] = -2.0 * m * r * dsigma_th / sigma2;

    let mix_r = -2.0 * m * a * s2 * (sigma - 2.0 * r * r) / sigma2;
    let mix_th = -2.0 * m * a * r * (2.0 * s * c * sigma - s2 * dsigma_th) / sigma2;
    dg_r[0][3] = mix_r;
    dg_r[3][0] = mix_r;
    dg_th[0][3] = mix_th;
    dg_th[3][0] = mix_th;

    dg_r[1][1] = (dsigma_r * delta - sigma * ddelta_r) / (delta * delta);
    dg_th[1][1] = dsigma_th / delta;

    dg_r[2][2] = dsigma_r;
    dg_th[2][2] = dsigma_th;

    let b = r * r + a2 + 2.0 * m * r * a2 * s2 / sigma;
    let db_r = 2.0 * r + 2.0 * m * a2 * s2 * (sigma - 2.0 * r * r) / sigma2;
    let db_th = 2.0 * m * r * a2 * (2.0 * s * c * sigma - s2 * dsigma_th) / sigma2;
    dg_r[3][3] = db_r * s2;
    dg_th[3][3] = db_th * s2 + b * 2.0 * s * c;

    Ok((dg_r, dg_th))
}

fn gamma_comp(
    gi: &[[f64; 4]; 4],
    dg_r: &[[f64; 4]; 4],
    dg_th: &[[f64; 4]; 4],
    sigma: usize,
    mu: usize,
    nu: usize,
) -> f64 {
    let d = |coord: usize, i: usize, j: usize| -> f64 {
        match coord {
            1 => dg_r[i][j],
            2 => dg_th[i][j],
            _ => 0.0,
        }
    };
    let mut acc = 0.0;
    for rho in 0..4 {
        acc += gi[sigma][rho] * (d(mu, rho, nu) + d(nu, rho, mu) - d(rho, mu, nu));
    }
    0.5 * acc
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
    use crate::gr::christoffel::Christoffel;
    use crate::gr::geodesic;
    use crate::gr::metric::{horizon_radius, isco_radius, photon_sphere_radius, Schwarzschild};
    use crate::gr::rk4::Scratch;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

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

    fn equatorial_timelike_turning(bh: Kerr, r: f64, ell: f64) -> GeodesicState {
        let theta = FRAC_PI_2;
        let gtt = bh.g_tt(r, theta).unwrap();
        let gtphi = bh.g_t_phi(r, theta).unwrap();
        let gphiphi = bh.g_phi_phi(r, theta).unwrap();
        let det = gtt * gphiphi - gtphi * gtphi;
        let u_t = (-(ell * ell + gphiphi) / det).sqrt();
        let u_phi = (ell - gtphi * u_t) / gphiphi;
        equatorial(r, u_t, 0.0, u_phi)
    }

    fn bardeen_energy_l(bh: Kerr, r: f64, prograde: bool) -> (f64, f64) {
        let m = bh.mass();
        let a = bh.spin();
        let sqrt_mr = (m * r).sqrt();
        let sign = if prograde { 1.0 } else { -1.0 };
        let denom = r * (r * r - 3.0 * m * r + sign * 2.0 * a * sqrt_mr).sqrt();
        let e = (r * r - 2.0 * m * r + sign * a * sqrt_mr) / denom;
        let ell = sign * sqrt_mr * (r * r - sign * 2.0 * a * sqrt_mr + a * a) / denom;
        (e, ell)
    }

    #[test]
    fn zero_spin_gamma_matches_schwarzschild() {
        let kerr = mass_one();
        let schw = schw_one();
        let r = 6.0;
        let theta = FRAC_PI_2;
        let gk = kerr.christoffel(r, theta).unwrap();
        let gs = Christoffel::at(schw, r, theta).unwrap();
        close(gk.gamma_t_tr, gs.gamma_t_tr, 1e-15);
        close(gk.gamma_r_tt, gs.gamma_r_tt, 1e-15);
        close(gk.gamma_r_rr, gs.gamma_r_rr, 1e-15);
        close(gk.gamma_r_theta_theta, gs.gamma_r_theta_theta, 1e-15);
        close(gk.gamma_r_phi_phi, gs.gamma_r_phi_phi, 1e-15);
        close(gk.gamma_theta_r_theta, gs.gamma_theta_r_theta, 1e-15);
        close(gk.gamma_theta_phi_phi, gs.gamma_theta_phi_phi, 1e-15);
        close(gk.gamma_phi_r_phi, gs.gamma_phi_r_phi, 1e-15);
        close(gk.gamma_phi_theta_phi, gs.gamma_phi_theta_phi, 1e-15);
        assert!(gk.gamma_t_t_theta.abs() < 1e-15);
        assert!(gk.gamma_t_r_phi.abs() < 1e-15);
        assert!(gk.gamma_t_theta_phi.abs() < 1e-15);
        assert!(gk.gamma_r_t_phi.abs() < 1e-15);
        assert!(gk.gamma_r_r_theta.abs() < 1e-15);
        assert!(gk.gamma_theta_tt.abs() < 1e-15);
        assert!(gk.gamma_theta_t_phi.abs() < 1e-15);
        assert!(gk.gamma_theta_rr.abs() < 1e-15);
        assert!(gk.gamma_theta_theta_theta.abs() < 1e-15);
        assert!(gk.gamma_phi_tr.abs() < 1e-15);
        assert!(gk.gamma_phi_t_theta.abs() < 1e-15);
    }

    #[test]
    fn metric_partials_match_central_difference() {
        let bh = spinning();
        let r = 5.0;
        let theta = 0.7;
        let h_r = 1e-6;
        let h_th = 1e-7;
        let (dg_r, dg_th) = metric_partials(bh, r, theta).unwrap();
        let g = |rr: f64, th: f64| bh.components(rr, th).unwrap();
        let gp = g(r + h_r, theta);
        let gm = g(r - h_r, theta);
        let tp = g(r, theta + h_th);
        let tm = g(r, theta - h_th);
        for i in 0..4 {
            for j in 0..4 {
                let num_r = (gp[i][j] - gm[i][j]) / (2.0 * h_r);
                let num_th = (tp[i][j] - tm[i][j]) / (2.0 * h_th);
                close(dg_r[i][j], num_r, 1e-7);
                close(dg_th[i][j], num_th, 1e-7);
            }
        }
    }

    #[test]
    fn spin_turns_on_frame_dragging_gamma() {
        let gk = spinning().christoffel(4.0, FRAC_PI_2).unwrap();
        assert!(gk.gamma_t_r_phi.abs() > 1e-6);
        assert!(gk.gamma_r_t_phi.abs() > 1e-6);
        assert!(gk.gamma_phi_tr.abs() > 1e-6);
        assert!(gk.gamma_t_t_theta.abs() < 1e-14);
        assert!(gk.gamma_r_r_theta.abs() < 1e-14);
        assert!(gk.gamma_theta_tt.abs() < 1e-14);
    }

    #[test]
    fn zero_spin_circular_matches_geodesic_state() {
        let kerr = mass_one();
        let schw = schw_one();
        let sk = kerr.circular_timelike(6.0, true).unwrap();
        let ss = GeodesicState::circular_timelike(schw, 6.0).unwrap();
        close(sk.u_t, ss.u_t, 1e-14);
        close(sk.u_phi, ss.u_phi, 1e-14);
        close(kerr.energy(sk).unwrap(), ss.energy(schw).unwrap(), 1e-14);
        close(
            kerr.angular_momentum(sk).unwrap(),
            ss.angular_momentum(schw).unwrap(),
            1e-14,
        );
        close(kerr.u_sq(sk).unwrap(), -1.0, 1e-14);
        let want_e = 4.0 / 18.0_f64.sqrt();
        let want_l = 12.0_f64.sqrt();
        close(kerr.energy(sk).unwrap(), want_e, 1e-14);
        close(kerr.angular_momentum(sk).unwrap(), want_l, 1e-14);
    }

    #[test]
    fn circular_four_velocity_has_zero_accel() {
        let kerr0 = mass_one();
        let kerr_a = spinning();
        for (bh, r, pro) in [
            (kerr0, 6.0, true),
            (kerr0, 10.0, true),
            (kerr_a, kerr_a.isco_plus(), true),
            (kerr_a, kerr_a.isco_minus(), false),
            (kerr_a, 12.0, true),
        ] {
            let s = bh.circular_timelike(r, pro).unwrap();
            let a = bh
                .christoffel(s.r, s.theta)
                .unwrap()
                .accel(s.four_velocity());
            assert!(
                a.iter().all(|v| v.abs() < 1e-12),
                "a={a:?} przy r={}, a_spin={}",
                s.r,
                bh.spin()
            );
        }
    }

    #[test]
    fn circular_requires_outside_photon_orbit() {
        let kerr = mass_one();
        assert!(matches!(
            kerr.circular_timelike(3.0, true),
            Err(KerrGeoError::NoCircularOrbit { r, mass, spin })
                if r == 3.0 && mass == 1.0 && spin == 0.0
        ));
        assert!(kerr.circular_timelike(2.5, true).is_err());
        let spinning_bh = spinning();
        assert!(spinning_bh
            .circular_timelike(spinning_bh.photon_plus(), true)
            .is_err());
        assert!(Kerr::new(0.0, 0.0)
            .unwrap()
            .circular_timelike(5.0, true)
            .is_ok());
    }

    #[test]
    fn spinning_isco_energy_matches_bardeen() {
        let bh = spinning();
        let r = bh.isco_plus();
        assert!(r < 6.0);
        let s = bh.circular_timelike(r, true).unwrap();
        let (want_e, want_l) = bardeen_energy_l(bh, r, true);
        close(bh.energy(s).unwrap(), want_e, 1e-12);
        close(bh.angular_momentum(s).unwrap(), want_l, 1e-12);
        close(bh.u_sq(s).unwrap(), -1.0, 1e-12);
        let retro = bh.circular_timelike(bh.isco_minus(), false).unwrap();
        assert!(bh.isco_minus() > 6.0);
        let (e_m, l_m) = bardeen_energy_l(bh, bh.isco_minus(), false);
        close(bh.energy(retro).unwrap(), e_m, 1e-12);
        close(bh.angular_momentum(retro).unwrap(), l_m, 1e-12);
    }

    #[test]
    fn zero_spin_isco_stays_at_six_m_like_geodesic() {
        let kerr = mass_one();
        let schw = schw_one();
        let start_k = kerr.circular_timelike(6.0, true).unwrap();
        let start_s = GeodesicState::circular_timelike(schw, 6.0).unwrap();
        let e0 = kerr.energy(start_k).unwrap();
        let l0 = kerr.angular_momentum(start_k).unwrap();
        let period = TAU / start_k.u_phi;
        let n = 1600;
        let h = period / n as f64;
        let yk = integrate(kerr, 0.0, &start_k.to_array(), h, n).unwrap();
        let ys = geodesic::integrate(schw, 0.0, &start_s.to_array(), h, n).unwrap();
        let end_k = GeodesicState::from_slice(&yk).unwrap();
        let end_s = GeodesicState::from_slice(&ys).unwrap();
        const TOL_R: f64 = 1e-7;
        const TOL_E: f64 = 1e-9;
        const TOL_PHI: f64 = 1e-6;
        let dr = (end_k.r - 6.0).abs();
        let de = (kerr.energy(end_k).unwrap() - e0).abs();
        let dl = (kerr.angular_momentum(end_k).unwrap() - l0).abs();
        let dphi = (end_k.phi - TAU).abs();
        assert!(dr < TOL_R, "|Δr| = {dr} po okrążeniu, tolerancja {TOL_R}");
        assert!(de < TOL_E, "|ΔE| = {de}, tolerancja {TOL_E}");
        assert!(dl < TOL_E, "|ΔL| = {dl}, tolerancja {TOL_E}");
        assert!(dphi < TOL_PHI, "φ = {}, chcemy 2π, |Δ| = {dphi}", end_k.phi);
        close(end_k.r, end_s.r, 1e-9);
        close(end_k.phi, end_s.phi, 1e-9);
        close(end_k.u_t, end_s.u_t, 1e-9);
        close(end_k.u_phi, end_s.u_phi, 1e-9);
        assert!((end_k.theta - FRAC_PI_2).abs() < 1e-12);
        assert!(end_k.u_r.abs() < 1e-7);
    }

    #[test]
    fn spinning_isco_plus_stays_inside_six_m() {
        let bh = spinning();
        let r0 = bh.isco_plus();
        assert!(r0 < 6.0, "ISCO+ = {r0} miało być < 6M");
        let start = bh.circular_timelike(r0, true).unwrap();
        let e0 = bh.energy(start).unwrap();
        let l0 = bh.angular_momentum(start).unwrap();
        let period = TAU / start.u_phi.abs();
        let n = 2000;
        let h = period / n as f64;
        let y = integrate(bh, 0.0, &start.to_array(), h, n).unwrap();
        let end = GeodesicState::from_slice(&y).unwrap();
        const TOL_R: f64 = 2e-6;
        const TOL_E: f64 = 1e-8;
        let dr = (end.r - r0).abs();
        let de = (bh.energy(end).unwrap() - e0).abs();
        let dl = (bh.angular_momentum(end).unwrap() - l0).abs();
        let dphi = (end.phi - TAU).abs();
        assert!(
            dr < TOL_R,
            "|Δr| = {dr} na ISCO+ = {r0}, tolerancja {TOL_R}"
        );
        assert!(de < TOL_E, "|ΔE| = {de}, tolerancja {TOL_E}");
        assert!(dl < TOL_E, "|ΔL| = {dl}, tolerancja {TOL_E}");
        assert!(dphi < 2e-5, "φ = {}, chcemy 2π, |Δ| = {dphi}", end.phi);
        assert!(end.r < 6.0);
        assert!((end.theta - FRAC_PI_2).abs() < 1e-12);
        close(bh.u_sq(end).unwrap(), -1.0, 1e-8);
    }

    #[test]
    fn energy_and_lz_drift_is_measured_with_spin() {
        let bh = spinning();
        let start = equatorial_timelike_turning(bh, 12.0, 3.5);
        assert!((bh.u_sq(start).unwrap() + 1.0).abs() < 1e-12);
        let e0 = bh.energy(start).unwrap();
        let l0 = bh.angular_momentum(start).unwrap();
        let k0 = bh.u_sq(start).unwrap();
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
            max_de = max_de.max((bh.energy(s).unwrap() - e0).abs());
            max_dl = max_dl.max((bh.angular_momentum(s).unwrap() - l0).abs());
            max_dk = max_dk.max((bh.u_sq(s).unwrap() - k0).abs());
            r_min = r_min.min(s.r);
            r_max = r_max.max(s.r);
        }
        const TOL: f64 = 1e-7;
        assert!(
            max_de < TOL,
            "max |ΔE| = {max_de}, tolerancja {TOL} (E₀ = {e0})"
        );
        assert!(
            max_dl < TOL,
            "max |ΔL_z| = {max_dl}, tolerancja {TOL} (L₀ = {l0})"
        );
        assert!(
            max_dk < TOL,
            "max |Δκ| = {max_dk}, tolerancja {TOL} (κ₀ = {k0})"
        );
        assert!(
            r_max > r_min + 0.4,
            "orbita miała oddychać w r, dostała [{r_min}, {r_max}]"
        );
        assert!(r_min > bh.horizon_radius(), "r weszło pod r+: {r_min}");
        assert!((GeodesicState::from_slice(&y).unwrap().theta - FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn rejects_wrong_state_len() {
        let bh = mass_one();
        let mut dy = [0.0; STATE_LEN];
        assert!(matches!(
            rhs(bh, &[1.0, 2.0], &mut dy),
            Err(KerrGeoError::BadStateLen { len: 2 })
        ));
        let mut y = vec![6.0; 3];
        let mut scratch = Scratch::new();
        assert!(matches!(
            step(bh, 0.0, &mut y, 0.01, &mut scratch),
            Err(KerrGeoError::BadStateLen { len: 3 })
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

    #[test]
    fn gamma_rejects_horizon_and_pole() {
        assert!(matches!(
            mass_one().christoffel(2.0, FRAC_PI_2),
            Err(KerrError::AtHorizon { r, mass, spin })
                if r == 2.0 && mass == 1.0 && spin == 0.0
        ));
        assert!(matches!(
            mass_one().christoffel(6.0, 0.0),
            Err(KerrError::InvalidAngle { theta }) if theta == 0.0
        ));
        assert!(spinning()
            .christoffel(spinning().horizon_radius(), FRAC_PI_2)
            .is_err());
    }
}
