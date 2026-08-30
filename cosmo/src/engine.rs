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
    /// Domyślny preset pod formację struktur: mniejsza próbka IC, gęstsza siatka.
    pub fn planck18_small() -> Self {
        Self {
            box_size: 32.0,
            n_grid: 48,
            pm_grid: 48,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0005,
            seed: 42,
        }
    }

    pub fn planck18_64() -> Self {
        Self {
            box_size: 40.0,
            n_grid: 64,
            pm_grid: 64,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0004,
            seed: 42,
        }
    }

    pub fn linear() -> Self {
        Self {
            box_size: 200.0,
            n_grid: 32,
            pm_grid: 32,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0008,
            seed: 7,
        }
    }
}

/// Mniejszy krok przy z<5 (nieliniowość), większy przy z>20 (era liniowa).
pub fn adaptive_dlna(z: f64, base: f64) -> f64 {
    let base = base.max(1e-6);
    let scale = if z >= 20.0 {
        1.5
    } else if z <= 5.0 {
        0.5
    } else {
        0.5 + (z - 5.0) / 15.0
    };
    (base * scale).clamp(0.00008, 0.05)
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
        let mut pm = ParticleMesh::new(cfg.pm_grid, rho_bar);
        let mut acc = vec![[0.0; 3]; n];
        pm.update(&ic.x, ic.mass);
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

    pub fn cloud_center_span(&self) -> ([f32; 3], f32) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for q in &self.x {
            for d in 0..3 {
                lo[d] = lo[d].min(q[d]);
                hi[d] = hi[d].max(q[d]);
            }
        }
        if !lo[0].is_finite() {
            return ([0.0; 3], 1.0);
        }
        let center = [
            0.5 * (lo[0] + hi[0]),
            0.5 * (lo[1] + hi[1]),
            0.5 * (lo[2] + hi[2]),
        ];
        let span = (hi[0] - lo[0])
            .max(hi[1] - lo[1])
            .max(hi[2] - lo[2])
            .max(1e-3);
        (center, span)
    }

    fn refresh_forces(&mut self) {
        self.pm.update(&self.x, self.mass);
        self.pm.gather(&self.x, &mut self.acc);
    }

    pub fn step(&mut self) {
        let a1 = self.a;
        // Brak twardego stopu na z=0: a rośnie dalej (przyszłość).
        let dlna = adaptive_dlna(self.redshift(), self.cfg.dlna);
        let a2 = a1 * dlna.exp();
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
        let delta = self.pm.density_at(self.x[i]);
        // 1+δ: pustka ≈ 0, tło = 1, halo ≫ 1. log₁₀ unosi filamenty, nie zjada halo.
        let contrast = (1.0 + delta).max(1e-4);
        ((contrast.log10()) * 0.72 + 0.18).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Cosmology;

    fn max_step_frac(a: &[[f32; 3]], b: &[[f32; 3]], span: f32) -> f32 {
        let mut max_frac = 0.0f32;
        for (p, q) in a.iter().zip(b.iter()) {
            for d in 0..3 {
                max_frac = max_frac.max((q[d] - p[d]).abs() / span);
            }
        }
        max_frac
    }

    #[test]
    fn structure_preset_fits_halos() {
        let c = RunConfig::planck18_small();
        assert!(c.box_size >= 25.0 && c.box_size <= 50.0);
        assert!(c.n_grid >= 48);
        assert!((0.0003..=0.0008).contains(&c.dlna));
    }

    #[test]
    fn adaptive_slows_down_after_z5() {
        let base = 0.005;
        let hi = adaptive_dlna(30.0, base);
        let mid = adaptive_dlna(12.0, base);
        let lo = adaptive_dlna(2.0, base);
        assert!(hi > mid && mid > lo);
        assert!((hi - base * 1.5).abs() < 1e-12);
        assert!((lo - base * 0.5).abs() < 1e-12);
    }

    #[test]
    fn first_step_drift_is_small_fraction_of_box() {
        let cfg = RunConfig {
            box_size: 32.0,
            n_grid: 16,
            pm_grid: 16,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0005,
            seed: 1,
        };
        let mut eng = Engine::new(Cosmology::planck18(), cfg);
        let x0 = eng.x.clone();
        let span = eng.cloud_center_span().1;
        eng.step();
        let frac = max_step_frac(&x0, &eng.x, span);
        assert!(frac < 0.05, "max drift frac={frac}");
    }

    #[test]
    fn particles_are_not_wrapped() {
        let cfg = RunConfig {
            box_size: 32.0,
            n_grid: 16,
            pm_grid: 16,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0005,
            seed: 1,
        };
        let mut eng = Engine::new(Cosmology::planck18(), cfg);
        let l = eng.box_size;
        eng.x[0] = [-1.0, 0.0, 0.0];
        eng.p[0] = [0.0, 0.0, 0.0];
        eng.acc[0] = [0.0, 0.0, 0.0];
        eng.step();
        assert!(
            eng.x[0][0] < 0.0,
            "pozycja zawinięta do pudła: {} (L={l})",
            eng.x[0][0]
        );
    }

    #[test]
    fn late_time_drift_stays_small_versus_cloud() {
        let cfg = RunConfig {
            box_size: 32.0,
            n_grid: 16,
            pm_grid: 16,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.04,
            seed: 1,
        };
        let mut eng = Engine::new(Cosmology::planck18(), cfg);
        let mut max_frac = 0.0f32;
        for _ in 0..40 {
            let x0 = eng.x.clone();
            let span = eng.cloud_center_span().1;
            eng.step();
            max_frac = max_frac.max(max_step_frac(&x0, &eng.x, span));
        }
        assert!(eng.redshift() < 20.0, "z={}", eng.redshift());
        assert!(max_frac < 0.15, "max drift frac={max_frac}");
    }
}
