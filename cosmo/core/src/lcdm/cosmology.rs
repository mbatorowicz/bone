//! Tło ΛCDM: `E(a)`, czynnik wzrostu `D(a)`, wiek, czynniki dryfu i kopnięcia.
//!
//! Wszystkie całki liczy jedno złożone Gaussa–Legendre'a. Wybór nie jest obojętny:
//! `D(a)` i wiek mają całki z osobliwością całkowalną przy `a → 0`, a metoda
//! Gaussa nie próbkuje końców przedziału, więc nie wchodzi w nią wprost.

use serde::{Deserialize, Serialize};

use crate::lcdm::units::{CRITICAL_DENSITY_0, GYR_PER_CODE_TIME, H100};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
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

impl Default for Cosmology {
    fn default() -> Self {
        Self::planck18()
    }
}

impl Cosmology {
    /// Planck 2018: TT,TE,EE+lowE+lensing+BAO (Aghanim i in. 2020).
    ///
    /// `Ω_Λ` nie jest wpisane z tabeli, a domknięte do jedności razem z gęstością
    /// promieniowania. Wpisanie obu z osobna dawałoby `Ω_tot ≠ 1` na czwartym
    /// miejscu i cichą krzywiznę w modelu, który ma być płaski.
    pub fn planck18() -> Self {
        let h = 0.6736;
        let omega_m = 0.3153;
        let t_cmb = 2.7255;
        let omega_r = omega_r_from_cmb(h, t_cmb);
        Self {
            h,
            omega_m,
            omega_b: 0.0493,
            omega_l: 1.0 - omega_m - omega_r,
            omega_r,
            omega_k: 0.0,
            n_s: 0.9649,
            sigma8: 0.8111,
            t_cmb,
        }
    }

    /// Einstein–de Sitter: `Ω_m = 1`, bez Λ. Model odniesienia dla testów, bo ma
    /// rozwiązania analityczne: `H ∝ a^{-3/2}`, `D = a`, `f = 1`.
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

    /// # Errors
    /// Gdy parametry nie opisują wszechświata, który da się policzyć.
    pub fn validate(self) -> Result<Self, String> {
        for (name, value) in [
            ("Ω_m", self.omega_m),
            ("h", self.h),
            ("T_CMB", self.t_cmb),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} musi być dodatnie i skończone, jest {value}"));
            }
        }
        if self.omega_b > self.omega_m {
            return Err(format!(
                "barionów więcej niż całej materii: Ω_b={} > Ω_m={}",
                self.omega_b, self.omega_m
            ));
        }
        Ok(self)
    }

    /// `E(a) = H(a)/H₀`.
    pub fn e(&self, a: f64) -> f64 {
        let a2 = a * a;
        (self.omega_r / (a2 * a2) + self.omega_m / (a2 * a) + self.omega_k / a2 + self.omega_l)
            .sqrt()
    }

    pub fn h_of_a(&self, a: f64) -> f64 {
        H100 * self.e(a)
    }

    pub fn omega_m_of_a(&self, a: f64) -> f64 {
        self.omega_m / (a * a * a * self.e(a).powi(2))
    }

    /// Średnia gęstość materii — stała w komowych jednostkach, więc bez argumentu.
    pub fn mean_matter_density(&self) -> f64 {
        self.omega_m * CRITICAL_DENSITY_0
    }

    /// `D(a)/D(1)`, znormalizowane do jedności dziś. `D ∝ H ∫ da/(aH)³`.
    pub fn growth(&self, a: f64) -> f64 {
        let integrand = |aa: f64| {
            if aa <= 0.0 {
                return 0.0;
            }
            1.0 / (aa * self.h_of_a(aa)).powi(3)
        };
        let num = self.h_of_a(a) * gauss_legendre(1e-6, a, 16, integrand);
        let den = self.h_of_a(1.0) * gauss_legendre(1e-6, 1.0, 16, integrand);
        num / den
    }

    /// `f = dlnD/dlna`, liczone różnicą centralną w logarytmie.
    pub fn growth_rate(&self, a: f64) -> f64 {
        let dln = 1e-3_f64;
        let a1 = (a * (-dln).exp()).max(1e-6);
        let a2 = a * dln.exp();
        (self.growth(a2).ln() - self.growth(a1).ln()) / (a2.ln() - a1.ln())
    }

    /// Wiek wszechświata w Gyr od `a = 0` do `a`.
    pub fn age_gyr(&self, a: f64) -> f64 {
        let integrand = |aa: f64| 1.0 / (aa.max(1e-8) * self.h_of_a(aa));
        gauss_legendre(1e-6, a, 32, integrand) * GYR_PER_CODE_TIME / self.h
    }

    /// `∫ da/(a³H)` — mnożnik pędu w kroku dryfu.
    pub fn drift_factor(&self, a1: f64, a2: f64) -> f64 {
        if (a2 - a1).abs() < 1e-14 {
            return 0.0;
        }
        gauss_legendre(a1, a2, 12, |a| 1.0 / (a.powi(3) * self.h_of_a(a)))
    }

    /// `∫ da/(a²H)` — mnożnik przyspieszenia w kroku kopnięcia.
    pub fn kick_factor(&self, a1: f64, a2: f64) -> f64 {
        if (a2 - a1).abs() < 1e-14 {
            return 0.0;
        }
        gauss_legendre(a1, a2, 12, |a| 1.0 / (a * a * self.h_of_a(a)))
    }
}

/// `Ω_r` z temperatury CMB: fotony plus trzy rodziny neutrin relatywistycznych.
fn omega_r_from_cmb(h: f64, t_cmb: f64) -> f64 {
    const N_EFF: f64 = 3.046;
    /// `(7/8)·(4/11)^{4/3}` — udział jednej rodziny neutrin względem fotonów.
    const NEUTRINO_SHARE: f64 = 0.227_107_317;
    let theta = t_cmb / 2.7255;
    let omega_gamma = 2.472e-5 * theta.powi(4) / (h * h);
    omega_gamma * (1.0 + NEUTRINO_SHARE * N_EFF)
}

/// Złożone 8-punktowe Gaussa–Legendre'a na `[lo, hi]`, podzielone na `panels` paneli.
fn gauss_legendre(lo: f64, hi: f64, panels: usize, f: impl Fn(f64) -> f64) -> f64 {
    if hi <= lo {
        return 0.0;
    }
    const NODES: [f64; 8] = [
        -0.960_289_856_497_536_3,
        -0.796_666_477_413_626_7,
        -0.525_532_409_916_329,
        -0.183_434_642_495_649_8,
        0.183_434_642_495_649_8,
        0.525_532_409_916_329,
        0.796_666_477_413_626_7,
        0.960_289_856_497_536_3,
    ];
    const WEIGHTS: [f64; 8] = [
        0.101_228_536_290_376_3,
        0.222_381_034_453_374_5,
        0.313_706_645_877_887_3,
        0.362_683_783_378_362,
        0.362_683_783_378_362,
        0.313_706_645_877_887_3,
        0.222_381_034_453_374_5,
        0.101_228_536_290_376_3,
    ];
    let n = panels.max(1);
    let width = (hi - lo) / n as f64;
    let half = 0.5 * width;
    (0..n)
        .map(|i| {
            let mid = lo + width * (i as f64 + 0.5);
            NODES
                .iter()
                .zip(WEIGHTS.iter())
                .map(|(x, w)| w * f(mid + half * x))
                .sum::<f64>()
                * half
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_today_is_one() {
        assert!((Cosmology::planck18().e(1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn planck18_is_flat() {
        let c = Cosmology::planck18();
        let total = c.omega_m + c.omega_l + c.omega_r + c.omega_k;
        assert!((total - 1.0).abs() < 1e-12, "Ω_tot = {total}");
    }

    #[test]
    fn eds_hubble_follows_the_analytic_law() {
        let c = Cosmology::eds();
        let a = 0.3_f64;
        let expected = H100 * a.powf(-1.5);
        assert!((c.h_of_a(a) - expected).abs() / expected < 1e-12);
    }

    /// W EdS wzrost jest dokładnie liniowy: `D = a`, `f = 1`. To jedyny sprawdzian
    /// całej kwadratury dla `D`, bo istnieje wynik analityczny do porównania.
    #[test]
    fn eds_growth_equals_the_scale_factor() {
        let c = Cosmology::eds();
        for a in [0.1, 0.3, 0.6, 1.0] {
            assert!((c.growth(a) - a).abs() < 2e-3, "a={a} D={}", c.growth(a));
            assert!(
                (c.growth_rate(a) - 1.0).abs() < 2e-2,
                "a={a} f={}",
                c.growth_rate(a)
            );
        }
    }

    /// W ΛCDM wzrost musi być POWOLNIEJSZY niż `a` — ciemna energia go tłumi.
    #[test]
    fn lambda_slows_the_growth_down() {
        let c = Cosmology::planck18();
        assert!((c.growth(1.0) - 1.0).abs() < 1e-12);
        assert!(c.growth(0.5) > 0.5, "D(0,5) = {}", c.growth(0.5));
        assert!(c.growth_rate(1.0) < 0.95, "f(1) = {}", c.growth_rate(1.0));
    }

    #[test]
    fn planck18_age_matches_the_published_value() {
        let age = Cosmology::planck18().age_gyr(1.0);
        assert!((age - 13.80).abs() < 0.08, "wiek = {age}");
    }

    #[test]
    fn age_grows_with_the_scale_factor() {
        let c = Cosmology::planck18();
        let mut previous = 0.0;
        for a in [0.02, 0.1, 0.3, 0.7, 1.0] {
            let age = c.age_gyr(a);
            assert!(age > previous, "wiek zmalał przy a={a}");
            previous = age;
        }
    }

    /// Czynniki kroku muszą być addytywne po przedziałach — inaczej wynik
    /// całkowania zależałby od tego, na ile kroków podzielono ten sam odcinek.
    #[test]
    fn step_factors_are_additive() {
        let c = Cosmology::planck18();
        let drift_whole = c.drift_factor(0.1, 0.8);
        let drift_split = c.drift_factor(0.1, 0.4) + c.drift_factor(0.4, 0.8);
        assert!((drift_split - drift_whole).abs() / drift_whole < 1e-8);

        let kick_whole = c.kick_factor(0.1, 0.8);
        let kick_split = c.kick_factor(0.1, 0.4) + c.kick_factor(0.4, 0.8);
        assert!((kick_split - kick_whole).abs() / kick_whole < 1e-8);
    }

    #[test]
    fn empty_interval_gives_no_step() {
        let c = Cosmology::planck18();
        assert_eq!(c.drift_factor(0.5, 0.5), 0.0);
        assert_eq!(c.kick_factor(0.5, 0.5), 0.0);
    }

    #[test]
    fn radiation_dominates_early() {
        let c = Cosmology::planck18();
        // Równość materia–promieniowanie wypada przy a ≈ Ω_r/Ω_m ≈ 1/3400.
        let a_eq = c.omega_r / c.omega_m;
        assert!(
            (1e-5..1e-3).contains(&a_eq),
            "równość przy a = {a_eq}, czyli Ω_r jest bez sensu"
        );
    }

    #[test]
    fn rejects_impossible_parameters() {
        let base = Cosmology::planck18();
        for broken in [
            Cosmology { omega_m: -0.1, ..base },
            Cosmology { h: 0.0, ..base },
            Cosmology { t_cmb: -1.0, ..base },
            Cosmology { omega_b: 0.9, ..base },
        ] {
            assert!(broken.validate().is_err());
        }
        assert!(base.validate().is_ok());
    }

    /// Węzły i wagi muszą być te właściwe. Kwadratura 8-punktowa całkuje wielomiany
    /// do stopnia 15 DOKŁADNIE, więc literówka w którejkolwiek stałej ujawnia się tu
    /// od razu — a w wynikach kosmologicznych ujawniłaby się jako parę procent
    /// przekłamania wieku wszechświata, czego nie da się odróżnić od innych parametrów.
    #[test]
    fn quadrature_is_exact_for_polynomials() {
        for (power, exact) in [(1i32, 0.5), (7, 0.125), (15, 1.0 / 16.0)] {
            let got = gauss_legendre(0.0, 1.0, 1, |x| x.powi(power));
            assert!((got - exact).abs() < 1e-14, "∫x^{power} = {got}, ma być {exact}");
        }
    }

    #[test]
    fn quadrature_handles_a_smooth_transcendental() {
        let got = gauss_legendre(0.0, std::f64::consts::PI, 4, f64::sin);
        assert!((got - 2.0).abs() < 1e-12, "∫sin = {got}");
    }

    /// Całki tła mają w podcałkowej ułamkową potęgę `a` (`a^{1/2}` w erze materii),
    /// więc zbieżność trzeba sprawdzić na tym, co naprawdę jest liczone, a nie na
    /// wielomianie. Ta niegładkość przy `a → 0` ogranicza dokładność do ~10⁻⁶
    /// względnie, czyli kilku dziesięciotysięcznych gigaroku — dużo poniżej
    /// niepewności samych parametrów Plancka.
    #[test]
    fn background_integrals_are_converged() {
        let c = Cosmology::planck18();
        let integrand = |a: f64| 1.0 / (a.max(1e-8) * c.h_of_a(a));
        let coarse = gauss_legendre(1e-6, 1.0, 32, integrand);
        let fine = gauss_legendre(1e-6, 1.0, 256, integrand);
        assert!(
            (coarse - fine).abs() / fine < 1e-5,
            "wiek: {coarse} vs {fine}"
        );
    }
}
