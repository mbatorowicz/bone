//! Integrator KDK po ln a + stan biegu.

use crate::cosmology::Cosmology;
use crate::ics::{make_initial_state, InitialState};
use crate::pm::ParticleMesh;

#[derive(Clone, Copy, Debug)]
pub struct RunConfig {
    pub box_size: f64,
    pub n_grid: usize,
    pub pm_grid: usize,
    pub z_start: f64,
    #[allow(dead_code)]
    pub z_end: f64,
    pub dlna: f64,
    pub seed: u64,
}

impl RunConfig {
    pub fn planck18_small() -> Self {
        Self {
            box_size: 100.0,
            n_grid: 32,
            pm_grid: 64,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.02,
            seed: 42,
        }
    }

    pub fn planck18_64() -> Self {
        Self {
            box_size: 100.0,
            n_grid: 48,
            pm_grid: 64,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.015,
            seed: 42,
        }
    }

    pub fn linear() -> Self {
        Self {
            box_size: 200.0,
            n_grid: 32,
            pm_grid: 64,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.03,
            seed: 7,
        }
    }
}

pub struct Engine {
    pub cosmology: Cosmology,
    pub cfg: RunConfig,
    pub a: f64,
    pub x: Vec<[f32; 3]>,
    pub p: Vec<[f32; 3]>,
    pub acc: Vec<[f32; 3]>,
    pub mass: f32,
    pub box_size: f32,
    pub step: u64,
    pm: ParticleMesh,
    t_plus_u0: f64,
    li_integral: f64,
    last_two_t_plus_u: f64,
}

impl Engine {
    pub fn new(cosmo: Cosmology, cfg: RunConfig) -> Self {
        let ic: InitialState =
            make_initial_state(cosmo, cfg.box_size, cfg.n_grid, cfg.z_start, cfg.seed);
        let n = ic.x.len();
        let rho_bar = cosmo.mean_matter_density() as f32;
        let mut pm = ParticleMesh::new(cfg.pm_grid, ic.box_size, rho_bar);
        let mut acc = vec![[0.0; 3]; n];
        pm.deposit(&ic.x, ic.mass);
        pm.solve_forces();
        pm.gather(&ic.x, &mut acc);
        let mut eng = Self {
            cosmology: cosmo,
            cfg,
            a: ic.a,
            x: ic.x,
            p: ic.p,
            acc,
            mass: ic.mass,
            box_size: ic.box_size,
            step: 0,
            pm,
            t_plus_u0: 0.0,
            li_integral: 0.0,
            last_two_t_plus_u: 0.0,
        };
        let (t, u) = eng.energies();
        eng.t_plus_u0 = t + u;
        eng.last_two_t_plus_u = 2.0 * t + u;
        eng
    }

    pub fn redshift(&self) -> f64 {
        1.0 / self.a - 1.0
    }

    pub fn n(&self) -> usize {
        self.x.len()
    }

    fn wrap(&mut self) {
        let l = self.box_size;
        for q in &mut self.x {
            q[0] = q[0].rem_euclid(l);
            q[1] = q[1].rem_euclid(l);
            q[2] = q[2].rem_euclid(l);
        }
    }

    fn refresh_forces(&mut self) {
        self.pm.deposit(&self.x, self.mass);
        self.pm.solve_forces();
        self.pm.gather(&self.x, &mut self.acc);
    }

    pub fn step(&mut self) {
        let a1 = self.a;
        let a2 = a1 * self.cfg.dlna.max(1e-6).exp();
        let amid = (a1 * a2).sqrt();

        let k1 = self.cosmology.kick_factor(a1, amid) as f32;
        for i in 0..self.p.len() {
            self.p[i][0] += self.acc[i][0] * k1;
            self.p[i][1] += self.acc[i][1] * k1;
            self.p[i][2] += self.acc[i][2] * k1;
        }

        let dr = self.cosmology.drift_factor(a1, a2) as f32;
        for i in 0..self.x.len() {
            self.x[i][0] += self.p[i][0] * dr;
            self.x[i][1] += self.p[i][1] * dr;
            self.x[i][2] += self.p[i][2] * dr;
        }
        self.wrap();
        self.refresh_forces();

        let k2 = self.cosmology.kick_factor(amid, a2) as f32;
        for i in 0..self.p.len() {
            self.p[i][0] += self.acc[i][0] * k2;
            self.p[i][1] += self.acc[i][1] * k2;
            self.p[i][2] += self.acc[i][2] * k2;
        }

        let (t, u) = self.energies();
        let two_t_u = 2.0 * t + u;
        self.li_integral += 0.5 * (self.last_two_t_plus_u + two_t_u) * (a2.ln() - a1.ln());
        self.last_two_t_plus_u = two_t_u;

        self.a = a2;
        self.step += 1;
    }

    pub fn energies(&self) -> (f64, f64) {
        // T = Σ p² / (2 m a²)
        let a2 = (self.a * self.a) as f32;
        let mut t = 0.0f64;
        for pi in &self.p {
            t += ((pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2]) / (2.0 * self.mass * a2)) as f64;
        }
        // U ~ −½ Σ m |acc| * cell  — rząd wielkości; do LI wystarczy spójny estymator
        // U = ½ Σ m Φ, Φ z F≈−∇Φ. Używamy −½ Σ m x·F / 3 (wiriал).
        let mut u = 0.0f64;
        for i in 0..self.x.len() {
            u += (self.mass
                * (self.x[i][0] * self.acc[i][0]
                    + self.x[i][1] * self.acc[i][1]
                    + self.x[i][2] * self.acc[i][2])) as f64;
        }
        u *= 0.5;
        (t, u)
    }

    /// Residuum Layzera–Irvine'a względne.
    pub fn layzer_irvine(&self) -> f64 {
        let (t, u) = self.energies();
        let denom = (t + u.abs()).max(1e-30);
        ((t + u) - self.t_plus_u0 + self.li_integral) / denom
    }

    pub fn shade(&self, i: usize) -> f32 {
        let d = self.pm.density_at(self.x[i]);
        (0.5 + 0.15 * d).clamp(0.0, 1.0)
    }
}
