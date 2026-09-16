//! Siatka 1D: ciepło FTCS i fala leapfrog. Bez CUDA, bez `g_μν`, bez 2D.
//!
//! Residual jest ten sam co w [`super::pinn`]: ciepło `u_t − k u_xx`, fala
//! `u_tt − c² u_xx`. Pamięć jest inna. PINN trzyma `u(x, t)` w suwakach.
//! Tu `u` siedzi na węzłach odcinka `[0, 1]`. Analityczne sinusy zostają
//! sędzią. CFL jest sitkiem, nie ozdobą: za gruby `Δt` i liczby wybuchają.
//! [`crate::grid`] to CIC N-ciał — inny zawód, ten plik go nie rusza.

use std::fmt;

use super::pinn::{heat_exact, heat_residual, wave_exact, wave_residual};

/// Ten sam `k` co obraz PINN. Residual analitycznego ciepła znika.
pub const K_HEAT: f64 = 0.3;
/// Ten sam `c` co obraz PINN. Residual analitycznej fali znika.
pub const C_WAVE: f64 = 1.0;
/// FTCS: `r = k Δt / Δx² ≤ 1/2`.
pub const CFL_HEAT: f64 = 0.5;
/// Leapfrog: `λ = c Δt / Δx ≤ 1`.
pub const CFL_WAVE: f64 = 1.0;
/// Domyślne `r` pod sitkiem. Widać stygnięcie, nie wybuch.
pub const R_SAFE: f64 = 0.4;
/// Domyślne `λ` pod sitkiem. Struna drga, nie pęka.
pub const LAMBDA_SAFE: f64 = 0.8;

/// Siatka za rzadka, krok psuje CFL, albo wektor ma złą długość.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FdError {
    TooCoarse { cells: usize },
    BadLen { expected: usize, got: usize },
    CflHeat { r: f64 },
    CflWave { lambda: f64 },
    BadParam,
}

impl fmt::Display for FdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::TooCoarse { cells } => {
                write!(f, "siatka 1D potrzebuje ≥ 2 komórek, jest {cells}")
            }
            Self::BadLen { expected, got } => {
                write!(f, "wektor ma {got} liczb, siatka chce {expected} węzłów")
            }
            Self::CflHeat { r } => {
                write!(f, "CFL ciepła pęka: r={r} > 1/2")
            }
            Self::CflWave { lambda } => {
                write!(f, "CFL fali pęka: λ={lambda} > 1")
            }
            Self::BadParam => write!(f, "k, c, r, λ albo Δt nie są skończone i dodatnie"),
        }
    }
}

impl std::error::Error for FdError {}

/// Odcinek `[0, 1]`. `cells` to liczba oczek, węzłów jest o jeden więcej.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mesh {
    cells: usize,
}

impl Mesh {
    pub fn new(cells: usize) -> Result<Self, FdError> {
        if cells < 2 {
            return Err(FdError::TooCoarse { cells });
        }
        Ok(Self { cells })
    }

    pub fn cells(self) -> usize {
        self.cells
    }

    pub fn nodes(self) -> usize {
        self.cells + 1
    }

    pub fn dx(self) -> f64 {
        1.0 / self.cells as f64
    }

    pub fn x(self, i: usize) -> f64 {
        i as f64 * self.dx()
    }
}

fn finite_pos(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// `r = k Δt / Δx²`. Sitko: `r ≤ 1/2`.
pub fn heat_r(k: f64, dx: f64, dt: f64) -> Result<f64, FdError> {
    if !finite_pos(k) || !finite_pos(dx) || !finite_pos(dt) {
        return Err(FdError::BadParam);
    }
    let r = k * dt / (dx * dx);
    if !r.is_finite() {
        return Err(FdError::BadParam);
    }
    if r > CFL_HEAT {
        return Err(FdError::CflHeat { r });
    }
    Ok(r)
}

/// `λ = c Δt / Δx`. Sitko: `λ ≤ 1`.
pub fn wave_lambda(c: f64, dx: f64, dt: f64) -> Result<f64, FdError> {
    if !finite_pos(c) || !finite_pos(dx) || !finite_pos(dt) {
        return Err(FdError::BadParam);
    }
    let lambda = c * dt / dx;
    if !lambda.is_finite() {
        return Err(FdError::BadParam);
    }
    if lambda > CFL_WAVE {
        return Err(FdError::CflWave { lambda });
    }
    Ok(lambda)
}

/// `Δt` z zadanego `r`. Nie pilnuje CFL — wybuch jest lekcją.
pub fn heat_dt(k: f64, dx: f64, r: f64) -> Result<f64, FdError> {
    if !finite_pos(k) || !finite_pos(dx) || !finite_pos(r) {
        return Err(FdError::BadParam);
    }
    let dt = r * dx * dx / k;
    dt.is_finite().then_some(dt).ok_or(FdError::BadParam)
}

/// `Δt` z zadanego `λ`. Nie pilnuje CFL.
pub fn wave_dt(c: f64, dx: f64, lambda: f64) -> Result<f64, FdError> {
    if !finite_pos(c) || !finite_pos(dx) || !finite_pos(lambda) {
        return Err(FdError::BadParam);
    }
    let dt = lambda * dx / c;
    dt.is_finite().then_some(dt).ok_or(FdError::BadParam)
}

fn check_len(mesh: Mesh, u: &[f64]) -> Result<(), FdError> {
    if u.len() != mesh.nodes() {
        return Err(FdError::BadLen {
            expected: mesh.nodes(),
            got: u.len(),
        });
    }
    Ok(())
}

/// Analityczne ciepło na węzłach. Brzegi są zerem.
pub fn sample_heat(mesh: Mesh, k: f64, t: f64) -> Result<Vec<f64>, FdError> {
    if !finite_pos(k) || !t.is_finite() || t < 0.0 {
        return Err(FdError::BadParam);
    }
    Ok((0..mesh.nodes())
        .map(|i| heat_exact(k, mesh.x(i), t))
        .collect())
}

/// Analityczna fala na węzłach. Brzegi są zerem.
pub fn sample_wave(mesh: Mesh, c: f64, t: f64) -> Result<Vec<f64>, FdError> {
    if !finite_pos(c) || !t.is_finite() {
        return Err(FdError::BadParam);
    }
    Ok((0..mesh.nodes())
        .map(|i| wave_exact(c, mesh.x(i), t))
        .collect())
}

fn apply_heat(r: f64, u: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut next = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        next[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
    }
    next
}

fn apply_wave(lambda2: f64, u: &[f64], prev: &[f64]) -> Vec<f64> {
    let n = u.len();
    let mut next = vec![0.0; n];
    for i in 1..n.saturating_sub(1) {
        next[i] = 2.0 * u[i] - prev[i] + lambda2 * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
    }
    next
}

/// Jedna klatka FTCS. CFL pilnowane. Brzegi zostają zerem.
pub fn step_heat(mesh: Mesh, k: f64, dt: f64, u: &mut [f64]) -> Result<f64, FdError> {
    check_len(mesh, u)?;
    let r = heat_r(k, mesh.dx(), dt)?;
    let next = apply_heat(r, u);
    u.copy_from_slice(&next);
    Ok(r)
}

/// Jedna klatka leapfroga. CFL pilnowane. `prev` to klatka wstecz.
pub fn step_wave(
    mesh: Mesh,
    c: f64,
    dt: f64,
    u: &mut [f64],
    prev: &mut [f64],
) -> Result<f64, FdError> {
    check_len(mesh, u)?;
    check_len(mesh, prev)?;
    let lambda = wave_lambda(c, mesh.dx(), dt)?;
    let next = apply_wave(lambda * lambda, u, prev);
    prev.copy_from_slice(u);
    u.copy_from_slice(&next);
    Ok(lambda)
}

fn sprinkle_nyquist(u: &mut [f64], amp: f64) {
    let n = u.len();
    for (i, slot) in u.iter_mut().enumerate().take(n.saturating_sub(1)).skip(1) {
        *slot += amp * if i % 2 == 0 { 1.0 } else { -1.0 };
    }
}

/// Film FTCS od analitycznego `t = 0`. Nie pilnuje CFL — wybuch widać.
pub fn run_heat(mesh: Mesh, k: f64, r: f64, steps: usize) -> Result<Vec<f64>, FdError> {
    if !finite_pos(k) || !finite_pos(r) {
        return Err(FdError::BadParam);
    }
    let mut u = sample_heat(mesh, k, 0.0)?;
    if r > CFL_HEAT {
        // Czysty sinus to najniższy tryb; sitko pęka na Nyquiście.
        sprinkle_nyquist(&mut u, 0.04);
    }
    for _ in 0..steps {
        u = apply_heat(r, &u);
        if u.iter().any(|x| !x.is_finite()) {
            break;
        }
    }
    Ok(u)
}

/// Film leapfroga od analitycznego `t = 0`. Nie pilnuje CFL.
pub fn run_wave(mesh: Mesh, c: f64, lambda: f64, steps: usize) -> Result<Vec<f64>, FdError> {
    if !finite_pos(c) || !finite_pos(lambda) {
        return Err(FdError::BadParam);
    }
    let dt = wave_dt(c, mesh.dx(), lambda)?;
    let mut u = sample_wave(mesh, c, 0.0)?;
    let mut prev = sample_wave(mesh, c, -dt)?;
    if lambda > CFL_WAVE {
        sprinkle_nyquist(&mut u, 0.04);
        sprinkle_nyquist(&mut prev, 0.04);
    }
    for _ in 0..steps {
        let next = apply_wave(lambda * lambda, &u, &prev);
        if next.iter().any(|x| !x.is_finite()) {
            break;
        }
        prev = u;
        u = next;
    }
    Ok(u)
}

/// Residual FTCS analitycznego `u` na węźle: `(u^{n+1} − u^n)/Δt − k Dxx u^n`.
pub fn heat_fd_residual<F>(k: f64, x: f64, t: f64, dx: f64, dt: f64, u: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let u_t = (u(x, t + dt) - u(x, t)) / dt;
    let u_xx = (u(x + dx, t) - 2.0 * u(x, t) + u(x - dx, t)) / (dx * dx);
    u_t - k * u_xx
}

/// Residual leapfroga analitycznego `u` na węźle.
pub fn wave_fd_residual<F>(c: f64, x: f64, t: f64, dx: f64, dt: f64, u: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let u_tt = (u(x, t + dt) - 2.0 * u(x, t) + u(x, t - dt)) / (dt * dt);
    let u_xx = (u(x + dx, t) - 2.0 * u(x, t) + u(x - dx, t)) / (dx * dx);
    u_tt - c * c * u_xx
}

fn interior_rms(mesh: Mesh, mut f: impl FnMut(f64) -> f64) -> f64 {
    let mut s = 0.0;
    let n = mesh.cells.saturating_sub(1);
    if n == 0 {
        return 0.0;
    }
    for i in 1..mesh.cells {
        let v = f(mesh.x(i));
        s += v * v;
    }
    (s / n as f64).sqrt()
}

/// RMS residualu FTCS analitycznego ciepła na węzłach. Ma spaść przy zagęszczaniu.
pub fn heat_truncation_rms(mesh: Mesh, k: f64, t: f64, r: f64) -> Result<f64, FdError> {
    let dx = mesh.dx();
    let dt = heat_dt(k, dx, r)?;
    Ok(interior_rms(mesh, |x| {
        heat_fd_residual(k, x, t, dx, dt, |xx, tt| heat_exact(k, xx, tt))
    }))
}

/// RMS residualu leapfroga analitycznej fali na węzłach.
pub fn wave_truncation_rms(mesh: Mesh, c: f64, t: f64, lambda: f64) -> Result<f64, FdError> {
    let dx = mesh.dx();
    let dt = wave_dt(c, dx, lambda)?;
    Ok(interior_rms(mesh, |x| {
        wave_fd_residual(c, x, t, dx, dt, |xx, tt| wave_exact(c, xx, tt))
    }))
}

/// Ten sam residual co PINN, liczony na węźle — sędzia analitycznego sinusa.
pub fn heat_node_residual(k: f64, x: f64, t: f64) -> f64 {
    heat_residual(k, x, t, |xx, tt| heat_exact(k, xx, tt))
}

/// Residual fali z PINN na węźle.
pub fn wave_node_residual(c: f64, x: f64, t: f64) -> f64 {
    wave_residual(c, x, t, |xx, tt| wave_exact(c, xx, tt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mut s = 0.0;
        for (x, y) in a.iter().zip(b) {
            let d = x - y;
            s += d * d;
        }
        (s / a.len() as f64).sqrt()
    }

    #[test]
    fn mesh_rejects_a_single_cell() {
        assert_eq!(Mesh::new(1), Err(FdError::TooCoarse { cells: 1 }));
        let m = Mesh::new(4).unwrap();
        assert_eq!(m.nodes(), 5);
        assert!((m.dx() - 0.25).abs() < 1e-15);
        assert!((m.x(2) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn heat_cfl_rejects_a_fat_step() {
        let dx = 0.1;
        let k = K_HEAT;
        let dt_ok = heat_dt(k, dx, R_SAFE).unwrap();
        assert!((heat_r(k, dx, dt_ok).unwrap() - R_SAFE).abs() < 1e-15);
        let dt_bad = heat_dt(k, dx, 0.8).unwrap();
        match heat_r(k, dx, dt_bad) {
            Err(FdError::CflHeat { r }) => assert!(r > CFL_HEAT),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn wave_cfl_rejects_a_fat_step() {
        let dx = 0.1;
        let dt_ok = wave_dt(C_WAVE, dx, LAMBDA_SAFE).unwrap();
        assert!((wave_lambda(C_WAVE, dx, dt_ok).unwrap() - LAMBDA_SAFE).abs() < 1e-15);
        let dt_bad = wave_dt(C_WAVE, dx, 1.4).unwrap();
        match wave_lambda(C_WAVE, dx, dt_bad) {
            Err(FdError::CflWave { lambda }) => assert!(lambda > CFL_WAVE),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn step_heat_refuses_cfl_and_keeps_the_vector() {
        let mesh = Mesh::new(8).unwrap();
        let mut u = sample_heat(mesh, K_HEAT, 0.0).unwrap();
        let before = u.clone();
        let dt = heat_dt(K_HEAT, mesh.dx(), 0.9).unwrap();
        assert!(matches!(
            step_heat(mesh, K_HEAT, dt, &mut u),
            Err(FdError::CflHeat { .. })
        ));
        assert_eq!(u, before);
    }

    #[test]
    fn heat_truncation_falls_when_mesh_is_refined() {
        let t = 0.04;
        let coarse = heat_truncation_rms(Mesh::new(8).unwrap(), K_HEAT, t, R_SAFE).unwrap();
        let fine = heat_truncation_rms(Mesh::new(32).unwrap(), K_HEAT, t, R_SAFE).unwrap();
        assert!(
            fine < coarse * 0.45,
            "coarse={coarse} fine={fine} miało spaść"
        );
        assert!(fine < 8e-3, "fine={fine}");
    }

    #[test]
    fn wave_truncation_falls_when_mesh_is_refined() {
        let t = 0.05;
        let coarse = wave_truncation_rms(Mesh::new(8).unwrap(), C_WAVE, t, 0.5).unwrap();
        let fine = wave_truncation_rms(Mesh::new(32).unwrap(), C_WAVE, t, 0.5).unwrap();
        assert!(
            fine < coarse * 0.45,
            "coarse={coarse} fine={fine} miało spaść"
        );
        assert!(fine < 5e-2, "fine={fine}");
    }

    #[test]
    fn analytic_heat_residual_on_nodes_vanishes() {
        let mesh = Mesh::new(16).unwrap();
        for i in 1..mesh.cells() {
            let r = heat_node_residual(K_HEAT, mesh.x(i), 0.05);
            assert!(r.abs() < 2e-6, "R={r} at x={}", mesh.x(i));
        }
    }

    #[test]
    fn analytic_wave_residual_on_nodes_vanishes() {
        let mesh = Mesh::new(16).unwrap();
        for i in 1..mesh.cells() {
            let r = wave_node_residual(C_WAVE, mesh.x(i), 0.1);
            assert!(r.abs() < 2e-4, "R={r} at x={}", mesh.x(i));
        }
    }

    #[test]
    fn heat_ftcs_tracks_exact_under_cfl() {
        let mesh = Mesh::new(40).unwrap();
        let dt = heat_dt(K_HEAT, mesh.dx(), R_SAFE).unwrap();
        let steps = 24;
        let u = run_heat(mesh, K_HEAT, R_SAFE, steps).unwrap();
        let exact = sample_heat(mesh, K_HEAT, dt * steps as f64).unwrap();
        let err = rms_diff(&u, &exact);
        assert!(err < 2e-3, "err={err}");
        assert_eq!(u[0], 0.0);
        assert_eq!(u[mesh.cells()], 0.0);
    }

    #[test]
    fn wave_leapfrog_tracks_exact_under_cfl() {
        let mesh = Mesh::new(40).unwrap();
        let dt = wave_dt(C_WAVE, mesh.dx(), LAMBDA_SAFE).unwrap();
        let steps = 20;
        let u = run_wave(mesh, C_WAVE, LAMBDA_SAFE, steps).unwrap();
        let exact = sample_wave(mesh, C_WAVE, dt * steps as f64).unwrap();
        let err = rms_diff(&u, &exact);
        assert!(err < 5e-3, "err={err}");
        assert_eq!(u[0], 0.0);
        assert_eq!(u[mesh.cells()], 0.0);
    }

    #[test]
    fn fat_heat_step_blows_up() {
        let mesh = Mesh::new(16).unwrap();
        let u = run_heat(mesh, K_HEAT, 0.9, 8).unwrap();
        let max = u.iter().fold(0.0_f64, |a, x| a.max(x.abs()));
        assert!(max > 2.0, "CFL miało rozsadzić garnek, max={max}");
    }

    #[test]
    fn step_heat_rejects_a_short_vector() {
        let mesh = Mesh::new(4).unwrap();
        let mut u = vec![0.0; 3];
        assert_eq!(
            step_heat(mesh, K_HEAT, 1e-4, &mut u),
            Err(FdError::BadLen {
                expected: 5,
                got: 3
            })
        );
    }
}
