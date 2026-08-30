//! Tło ΛCDM: E(a), D(a), czynniki drift/kick, wiek.

use crate::units::{critical_density_0, GYR_PER_CODE_TIME, H100};

#[derive(Clone, Copy, Debug)]
pub struct Cosmology {
    pub h: f64,
    pub omega_m: f64,
    pub omega_b: f64,
    pub omega_l: f64,
    pub omega_r: f64,
    pub omega_k: f64,
    pub n_s: f64,
    pub sigma8: f64,
    pub t_cmb: f64,
}

impl Cosmology {
    /// Planck 2018 TT,TE,EE+lowE+lensing+BAO (Aghanim et al.).
    pub fn planck18() -> Self {
        let h = 0.6736;
        let omega_m = 0.3153;
        let omega_b = 0.0493;
        let t_cmb = 2.7255;
        let omega_r = omega_r_from_cmb(h, t_cmb);
        let omega_l = 1.0 - omega_m - omega_r;
        Self {
            h,
            omega_m,
            omega_b,
            omega_l,
            omega_r,
            omega_k: 0.0,
            n_s: 0.9649,
            sigma8: 0.8111,
            t_cmb,
        }
    }

    #[allow(dead_code)]
    pub fn eds() -> Self {
        Self {
            h: 0.7,
            omega_m: 1.0,
            omega_b: 0.05,
            omega_l: 0.0,
            omega_r: 0.0,
            omega_k: 0.0,
            n_s: 1.0,
            sigma8: 0.8,
            t_cmb: 2.7255,
        }
    }

    #[allow(dead_code)]
    pub fn validate(self) -> Result<Self, String> {
        if self.omega_m <= 0.0 {
            return Err(format!("Omega_m musi być dodatnie, dostałem {}", self.omega_m));
        }
        if self.h <= 0.0 {
            return Err(format!("h musi być dodatnie, dostałem {}", self.h));
        }
        Ok(self)
    }

    pub fn e(&self, a: f64) -> f64 {
        let a2 = a * a;
        let a3 = a2 * a;
        (self.omega_r / (a2 * a2)
            + self.omega_m / a3
            + self.omega_k / a2
            + self.omega_l)
            .sqrt()
    }

    pub fn h_of_a(&self, a: f64) -> f64 {
        H100 * self.e(a)
    }

    #[allow(dead_code)]
    pub fn omega_m_of_a(&self, a: f64) -> f64 {
        self.omega_m / (a * a * a * self.e(a).powi(2))
    }

    pub fn mean_matter_density(&self) -> f64 {
        self.omega_m * critical_density_0(self.h)
    }

    /// D(a)/D(1), D(1)=1. D ∝ H ∫ da'/(a' H³).
    pub fn growth(&self, a: f64) -> f64 {
        let i = |aa: f64| {
            if aa <= 0.0 {
                return 0.0;
            }
            let h = self.h_of_a(aa);
            1.0 / (aa * h).powi(3)
        };
        let num = self.h_of_a(a) * gl_integral(1e-6, a, 16, i);
        let den = self.h_of_a(1.0) * gl_integral(1e-6, 1.0, 16, i);
        num / den
    }

    pub fn growth_rate(&self, a: f64) -> f64 {
        let dln = 1e-3_f64;
        let a1 = (a * (-dln).exp()).max(1e-4);
        let a2 = (a * dln.exp()).min(1.2);
        let d1 = self.growth(a1).ln();
        let d2 = self.growth(a2).ln();
        (d2 - d1) / (a2.ln() - a1.ln())
    }

    /// Wiek w Gyr od a=0 do a.
    pub fn age_gyr(&self, a: f64) -> f64 {
        let i = |aa: f64| 1.0 / (aa.max(1e-8) * self.h_of_a(aa));
        let t_code = gl_integral(1e-6, a, 32, i);
        t_code * GYR_PER_CODE_TIME / self.h
    }

    pub fn drift_factor(&self, a1: f64, a2: f64) -> f64 {
        if (a2 - a1).abs() < 1e-14 {
            return 0.0;
        }
        gl_integral(a1, a2, 12, |a| 1.0 / (a.powi(3) * self.h_of_a(a)))
    }

    pub fn kick_factor(&self, a1: f64, a2: f64) -> f64 {
        if (a2 - a1).abs() < 1e-14 {
            return 0.0;
        }
        gl_integral(a1, a2, 12, |a| 1.0 / (a * a * self.h_of_a(a)))
    }
}

fn omega_r_from_cmb(h: f64, t_cmb: f64) -> f64 {
    let n_eff = 3.046;
    let theta = t_cmb / 2.7255;
    let omega_g_h2 = 2.472e-5 * theta.powi(4);
    let omega_g = omega_g_h2 / (h * h);
    omega_g * (1.0 + 0.227_107_317 * n_eff)
}

/// Złożenie 8-punktowego Gaussa–Legendre'a na [lo, hi] (`panels` paneli).
fn gl_integral(lo: f64, hi: f64, panels: usize, f: impl Fn(f64) -> f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    const X8: [f64; 8] = [
        -0.960_289_856_497_536_3,
        -0.796_666_477_413_626_7,
        -0.525_532_409_916_329_0,
        -0.183_434_642_495_649_8,
        0.183_434_642_495_649_8,
        0.525_532_409_916_329_0,
        0.796_666_477_413_626_7,
        0.960_289_856_497_536_3,
    ];
    const W8: [f64; 8] = [
        0.101_228_536_290_376_3,
        0.222_381_034_453_374_5,
        0.313_706_645_877_887_3,
        0.362_683_783_378_362_0,
        0.362_683_783_378_362_0,
        0.313_706_645_877_887_3,
        0.222_381_034_453_374_5,
        0.101_228_536_290_376_3,
    ];
    let n = panels.max(1);
    let mut acc = 0.0;
    for i in 0..n {
        let a = lo + (hi - lo) * i as f64 / n as f64;
        let b = lo + (hi - lo) * (i + 1) as f64 / n as f64;
        let mid = 0.5 * (a + b);
        let half = 0.5 * (b - a);
        acc += X8
            .iter()
            .zip(W8.iter())
            .map(|(x, w)| w * f(mid + half * x))
            .sum::<f64>()
            * half;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_today_is_one() {
        let c = Cosmology::planck18();
        assert!((c.e(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eds_hubble() {
        let c = Cosmology::eds();
        let a: f64 = 0.3;
        let expect = H100 * a.powf(-1.5);
        assert!((c.h_of_a(a) - expect).abs() / expect < 1e-12);
    }

    #[test]
    fn eds_growth_is_a() {
        let c = Cosmology::eds();
        for a in [0.1, 0.3, 0.6, 1.0] {
            let d = c.growth(a);
            assert!((d - a).abs() < 2e-3, "a={a} D={d}");
            let f = c.growth_rate(a);
            assert!((f - 1.0).abs() < 2e-2, "a={a} f={f}");
        }
    }

    #[test]
    fn planck_age() {
        let age = Cosmology::planck18().age_gyr(1.0);
        assert!((age - 13.80).abs() < 0.08, "age={age}");
    }

    #[test]
    fn drift_additive() {
        let c = Cosmology::planck18();
        let a = c.drift_factor(0.1, 0.4);
        let b = c.drift_factor(0.4, 0.8);
        let d = c.drift_factor(0.1, 0.8);
        assert!((a + b - d).abs() / d.abs().max(1e-20) < 1e-8);
    }

    #[test]
    fn rejects_negative_omega() {
        let mut c = Cosmology::planck18();
        c.omega_m = -0.1;
        assert!(c.validate().is_err());
    }
}
