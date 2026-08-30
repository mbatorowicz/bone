//! Widmo P(k): Eisenstein & Hu 1998 (no-wiggle) + normalizacja σ₈.

use crate::cosmology::Cosmology;

pub struct PowerSpectrum {
    pub cosmology: Cosmology,
    amplitude: f64,
}

impl PowerSpectrum {
    pub fn eisenstein_hu(cosmo: Cosmology) -> Self {
        let mut p = Self {
            cosmology: cosmo,
            amplitude: 1.0,
        };
        let s8 = sigma_r(&p, 8.0);
        p.amplitude = (cosmo.sigma8 / s8).powi(2);
        p
    }

    pub fn transfer(&self, k: f64) -> f64 {
        eh98_nowiggle(&self.cosmology, k.max(1e-8))
    }

    /// P(k, a) = A k^{n_s} T² D(a)²  [ (Mpc/h)³ ]
    pub fn p(&self, k: f64, a: f64) -> f64 {
        let t = self.transfer(k);
        let d = self.cosmology.growth(a);
        self.amplitude * k.powf(self.cosmology.n_s) * t * t * d * d
    }
}

fn eh98_nowiggle(c: &Cosmology, k: f64) -> f64 {
    let h = c.h;
    let omhh = c.omega_m * h * h;
    let obhh = c.omega_b * h * h;
    let theta = c.t_cmb / 2.7;
    let s = 44.5 * (9.83 / omhh).ln() / (1.0 + 10.0 * obhh.powf(0.75)).sqrt();
    let alpha_gamma = 1.0
        - 0.328 * (431.0 * omhh).ln() * obhh / omhh
        + 0.38 * (22.3 * omhh).ln() * (obhh / omhh).powi(2);
    let gamma = c.omega_m * h;
    let ks = 0.43 * k * s;
    let gamma_eff = gamma * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + ks.powi(4)));
    let q = k * theta * theta / gamma_eff;
    let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
    let l0 = (2.0_f64.exp() + 1.8 * q).ln();
    l0 / (l0 + c0 * q * q)
}

fn top_hat(x: f64) -> f64 {
    if x < 1e-5 {
        1.0 - x * x / 10.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x * x * x)
    }
}

fn sigma_r(p: &PowerSpectrum, r: f64) -> f64 {
    let n = 256;
    let ln_kmin = (1e-4_f64).ln();
    let ln_kmax = (40.0_f64).ln();
    let dln = (ln_kmax - ln_kmin) / n as f64;
    let mut acc = 0.0;
    for i in 0..=n {
        let k = (ln_kmin + i as f64 * dln).exp();
        let w = top_hat(k * r);
        let pk = p.amplitude * k.powf(p.cosmology.n_s) * p.transfer(k).powi(2);
        let delta2 = k.powi(3) * pk / (2.0 * std::f64::consts::PI.powi(2));
        let weight = if i == 0 || i == n { 0.5 } else { 1.0 };
        acc += weight * delta2 * w * w * dln;
    }
    acc.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transfer_ir_limit() {
        let p = PowerSpectrum::eisenstein_hu(Cosmology::planck18());
        assert!((p.transfer(1e-5) - 1.0).abs() < 2e-3);
    }

    #[test]
    fn transfer_falls() {
        let p = PowerSpectrum::eisenstein_hu(Cosmology::planck18());
        assert!(p.transfer(1.0) < p.transfer(0.1));
        assert!(p.transfer(5.0) < p.transfer(1.0));
    }

    #[test]
    fn sigma8_normalized() {
        let c = Cosmology::planck18();
        let p = PowerSpectrum::eisenstein_hu(c);
        let s = sigma_r(&p, 8.0);
        assert!((s - c.sigma8).abs() < 1e-4, "sigma8={s}");
    }
}
