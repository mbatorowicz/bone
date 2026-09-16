//! Równanie pola na Schwarzschildu: Riemann, Ricci, `G_μν`, `T_μν`.
//!
//! To lewa i prawa strona z lekcji Einsteina, nie zgadywanie metryki na
//! siatce. [`Christoffel`] zostaje wzorem zamkniętym; Riemann składa z niego
//! pochodną i iloczyn ΓΓ. Ricci zszywa powtórzony wskaźnik. Tensor Einsteina
//! ustawia zszywkę naprzeciw energii: `G_μν = R_μν − (1/2) R g_μν`.
//!
//! Sygnatura (−,+,+,+), współrzędne `(t, r, θ, φ)`, jak w [`super::metric`].
//! `G = c = 1`, więc równanie pola to `G_μν = 8π T_μν`. Próżnia jest zerem.
//! Pył to `ρ u_μ u_ν`. Kerr, zderzenia i PDE na siatce nie wchodzą.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::PI;

use super::christoffel::Christoffel;
use super::metric::{MetricError, Schwarzschild};
use super::tensor::{Tensor02, Vector};

/// Krzywizna w punkcie: Riemann `R^ρ_{σμν}`, Ricci, skalar, `G_μν`.
#[derive(Clone, Copy, Debug)]
pub struct Curvature {
    riemann: [[[[f64; 4]; 4]; 4]; 4],
    ricci: [[f64; 4]; 4],
    scalar: f64,
    einstein: [[f64; 4]; 4],
    kretschmann: f64,
}

impl Curvature {
    /// `R^ρ_{σμν}`.
    pub fn riemann(self) -> [[[[f64; 4]; 4]; 4]; 4] {
        self.riemann
    }

    /// `R_μν`.
    pub fn ricci(self) -> [[f64; 4]; 4] {
        self.ricci
    }

    /// `R = g^{μν} R_μν`.
    pub fn scalar(self) -> f64 {
        self.scalar
    }

    /// `G_μν` jako waga (0,2).
    pub fn einstein(self) -> Tensor02 {
        Tensor02::from_components(self.einstein)
    }

    /// `K = R_{αβγδ} R^{αβγδ}`.
    pub fn kretschmann(self) -> f64 {
        self.kretschmann
    }
}

impl Schwarzschild {
    /// Krzywizna w `(r, θ)`. Czas i `φ` nie wchodzą: stacjonarnie i osiowo.
    pub fn curvature(self, r: f64, theta: f64) -> Result<Curvature, MetricError> {
        let gamma = gamma_full(self.christoffel(r, theta)?);
        let dgamma = gamma_partials(self, r, theta)?;
        let riemann = riemann_up(gamma, dgamma);
        let g = self.components(r, theta)?;
        let g_inv = self.inverse(r, theta)?;
        let ricci = ricci_down(riemann);
        let scalar = contract_scalar(g_inv, ricci);
        let einstein = einstein_down(ricci, scalar, g);
        let kretschmann = kretschmann_of(riemann, g, g_inv);
        Ok(Curvature {
            riemann,
            ricci,
            scalar,
            einstein,
            kretschmann,
        })
    }

    pub fn kretschmann(self, r: f64, theta: f64) -> Result<f64, MetricError> {
        Ok(self.curvature(r, theta)?.kretschmann)
    }
}

/// `T_μν = 0`. Puste pudełko na łące.
pub fn vacuum() -> Tensor02 {
    Tensor02::from_components([[0.0; 4]; 4])
}

/// Pył: `T_μν = ρ u_μ u_ν`, `u_μ = g_μν u^ν`.
pub fn dust(density: f64, four_velocity: [f64; 4], metric: Tensor02) -> Tensor02 {
    let u_down = metric.covector(Vector::from_components(four_velocity));
    let ud = u_down.components();
    let mut t = [[0.0; 4]; 4];
    for (mu, row) in t.iter_mut().enumerate() {
        for (nu, cell) in row.iter_mut().enumerate() {
            *cell = density * ud[mu] * ud[nu];
        }
    }
    Tensor02::from_components(t)
}

/// `G_μν − 8π T_μν`. Zero znaczy, że równanie pola stoi.
pub fn field_residual(einstein: Tensor02, stress_energy: Tensor02) -> Tensor02 {
    let g = einstein.components();
    let t = stress_energy.components();
    let mut r = [[0.0; 4]; 4];
    for (i, row) in r.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = g[i][j] - 8.0 * PI * t[i][j];
        }
    }
    Tensor02::from_components(r)
}

/// `Γ^ρ_{μν}` jako tablica `[ρ][μ][ν]`. Brakujące kratki są zerem.
fn gamma_full(c: Christoffel) -> [[[f64; 4]; 4]; 4] {
    let mut g = [[[0.0; 4]; 4]; 4];
    g[0][0][1] = c.gamma_t_tr;
    g[0][1][0] = c.gamma_t_tr;
    g[1][0][0] = c.gamma_r_tt;
    g[1][1][1] = c.gamma_r_rr;
    g[1][2][2] = c.gamma_r_theta_theta;
    g[1][3][3] = c.gamma_r_phi_phi;
    g[2][1][2] = c.gamma_theta_r_theta;
    g[2][2][1] = c.gamma_theta_r_theta;
    g[2][3][3] = c.gamma_theta_phi_phi;
    g[3][1][3] = c.gamma_phi_r_phi;
    g[3][3][1] = c.gamma_phi_r_phi;
    g[3][2][3] = c.gamma_phi_theta_phi;
    g[3][3][2] = c.gamma_phi_theta_phi;
    g
}

/// `∂_α Γ^ρ_{μν}` w `[α][ρ][μ][ν]`. Czas i `φ` są Killingami: te pochodne znikają.
fn gamma_partials(
    bh: Schwarzschild,
    r: f64,
    theta: f64,
) -> Result<[[[[f64; 4]; 4]; 4]; 4], MetricError> {
    let hr = (1.0e-6 * r).max(1.0e-8);
    let ht = 1.0e-6;
    let gp_r = gamma_full(bh.christoffel(r + hr, theta)?);
    let gm_r = gamma_full(bh.christoffel(r - hr, theta)?);
    let gp_th = gamma_full(bh.christoffel(r, theta + ht)?);
    let gm_th = gamma_full(bh.christoffel(r, theta - ht)?);
    let mut d = [[[[0.0; 4]; 4]; 4]; 4];
    for rho in 0..4 {
        for mu in 0..4 {
            for nu in 0..4 {
                d[1][rho][mu][nu] = (gp_r[rho][mu][nu] - gm_r[rho][mu][nu]) / (2.0 * hr);
                d[2][rho][mu][nu] = (gp_th[rho][mu][nu] - gm_th[rho][mu][nu]) / (2.0 * ht);
            }
        }
    }
    Ok(d)
}

/// `R^ρ_{σμν} = ∂_μ Γ^ρ_{νσ} − ∂_ν Γ^ρ_{μσ} + Γ^ρ_{μλ} Γ^λ_{νσ} − Γ^ρ_{νλ} Γ^λ_{μσ}`.
fn riemann_up(
    gamma: [[[f64; 4]; 4]; 4],
    dgamma: [[[[f64; 4]; 4]; 4]; 4],
) -> [[[[f64; 4]; 4]; 4]; 4] {
    let mut r = [[[[0.0; 4]; 4]; 4]; 4];
    for rho in 0..4 {
        for sig in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    let mut val = dgamma[mu][rho][nu][sig] - dgamma[nu][rho][mu][sig];
                    for lam in 0..4 {
                        val += gamma[rho][mu][lam] * gamma[lam][nu][sig]
                            - gamma[rho][nu][lam] * gamma[lam][mu][sig];
                    }
                    r[rho][sig][mu][nu] = val;
                }
            }
        }
    }
    r
}

/// `R_μν = R^λ_{μλν}`.
fn ricci_down(riemann: [[[[f64; 4]; 4]; 4]; 4]) -> [[f64; 4]; 4] {
    let mut ric = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            let mut s = 0.0;
            for lam in 0..4 {
                s += riemann[lam][mu][lam][nu];
            }
            ric[mu][nu] = s;
        }
    }
    ric
}

fn contract_scalar(g_inv: [[f64; 4]; 4], ricci: [[f64; 4]; 4]) -> f64 {
    let mut s = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            s += g_inv[mu][nu] * ricci[mu][nu];
        }
    }
    s
}

fn einstein_down(ricci: [[f64; 4]; 4], scalar: f64, g: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut e = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            e[mu][nu] = ricci[mu][nu] - 0.5 * scalar * g[mu][nu];
        }
    }
    e
}

fn kretschmann_of(riemann: [[[[f64; 4]; 4]; 4]; 4], g: [[f64; 4]; 4], g_inv: [[f64; 4]; 4]) -> f64 {
    let mut k = 0.0;
    for alpha in 0..4 {
        for beta in 0..4 {
            for gamma in 0..4 {
                for delta in 0..4 {
                    let down = g[alpha][alpha] * riemann[alpha][beta][gamma][delta];
                    let up = g_inv[alpha][alpha]
                        * g_inv[beta][beta]
                        * g_inv[gamma][gamma]
                        * g_inv[delta][delta]
                        * down;
                    k += down * up;
                }
            }
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    fn max_abs(m: [[f64; 4]; 4]) -> f64 {
        m.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max)
    }

    fn max_abs_riemann(r: [[[[f64; 4]; 4]; 4]; 4]) -> f64 {
        r.iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0, f64::max)
    }

    fn mass_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    fn textbook_kretschmann(mass: f64, r: f64) -> f64 {
        48.0 * mass * mass / r.powi(6)
    }

    #[test]
    fn flat_spherical_minkowski_has_zero_riemann() {
        let flat = Schwarzschild::new(0.0).unwrap();
        let c = flat.curvature(5.0, FRAC_PI_2).unwrap();
        assert!(
            max_abs_riemann(c.riemann()) < 1.0e-8,
            "Riemann={}",
            max_abs_riemann(c.riemann())
        );
        assert!(max_abs(c.ricci()) < 1.0e-8);
        assert!(c.scalar().abs() < 1.0e-8);
        assert!(max_abs(c.einstein().components()) < 1.0e-8);
        assert!(c.kretschmann().abs() < 1.0e-8);
    }

    #[test]
    fn kretschmann_matches_48_m2_over_r6() {
        let bh = mass_one();
        for r in [4.0, 6.0, 8.0, 12.0] {
            let got = bh.kretschmann(r, FRAC_PI_2).unwrap();
            let want = textbook_kretschmann(1.0, r);
            assert!(
                (got - want).abs() / want < 1.0e-5,
                "r={r} got={got} want={want}"
            );
        }
        let heavy = Schwarzschild::new(1.5).unwrap();
        let got = heavy.kretschmann(9.0, FRAC_PI_2).unwrap();
        let want = textbook_kretschmann(1.5, 9.0);
        assert!((got - want).abs() / want < 1.0e-5);
    }

    #[test]
    fn schwarzschild_vacuum_has_zero_einstein_tensor() {
        let bh = mass_one();
        for r in [4.0, 6.0, 10.0] {
            let c = bh.curvature(r, FRAC_PI_2).unwrap();
            assert!(
                max_abs(c.ricci()) < 1.0e-7,
                "Ricci r={r} {}",
                max_abs(c.ricci())
            );
            assert!(c.scalar().abs() < 1.0e-7);
            let g = c.einstein();
            assert!(
                max_abs(g.components()) < 1.0e-7,
                "G r={r} {}",
                max_abs(g.components())
            );
            let residual = field_residual(g, vacuum());
            assert!(max_abs(residual.components()) < 1.0e-7);
        }
    }

    #[test]
    fn riemann_is_antisymmetric_in_the_last_pair() {
        let c = mass_one().curvature(6.0, FRAC_PI_2).unwrap();
        let r = c.riemann();
        for rho in 0..4 {
            for sig in 0..4 {
                for mu in 0..4 {
                    for nu in 0..4 {
                        assert!((r[rho][sig][mu][nu] + r[rho][sig][nu][mu]).abs() < 1.0e-9);
                    }
                }
            }
        }
    }

    #[test]
    fn vacuum_residual_is_exactly_einstein_when_t_vanishes() {
        let g = Tensor02::from_components([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ]);
        let residual = field_residual(g, vacuum());
        assert_eq!(residual.components(), g.components());
    }

    #[test]
    fn dust_is_density_times_covector_outer_product() {
        let metric = Tensor02::minkowski();
        let u = [1.0, 0.0, 0.0, 0.0];
        let t = dust(2.0, u, metric);
        // u^t = 1, η_tt = −1 ⇒ u_t = −1, T_tt = ρ u_t u_t = 2.
        let want = 2.0;
        assert!((t.components()[0][0] - want).abs() < 1.0e-15);
        assert!(t.components()[1][1].abs() < 1.0e-15);
        let residual = field_residual(vacuum(), t);
        assert!((residual.components()[0][0] + 8.0 * PI * want).abs() < 1.0e-12);
    }

    #[test]
    fn horizon_is_rejected() {
        assert!(matches!(
            mass_one().curvature(2.0, FRAC_PI_2),
            Err(MetricError::AtHorizon { r, mass }) if r == 2.0 && mass == 1.0
        ));
    }

    #[test]
    fn curvature_grows_toward_the_center() {
        let bh = mass_one();
        let far = bh.kretschmann(12.0, FRAC_PI_2).unwrap();
        let near = bh.kretschmann(4.0, FRAC_PI_2).unwrap();
        assert!(near > far);
        assert!(far > 0.0);
    }
}
