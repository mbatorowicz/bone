//! Widmo mocy `P(k)`: funkcja przejścia Eisensteina i Hu (1998, wariant bez oscylacji)
//! znormalizowana przez `σ₈`.
//!
//! Wariant „no-wiggle" nie ma oscylacji barionowych (BAO). Jest to świadome
//! uproszczenie, a nie brak: oscylacje mają amplitudę kilku procent i skalę 150 Mpc,
//! więc w próbce 32 Mpc/h nie mieści się ani jeden ich okres. Za to wzór jest
//! zamknięty i nie wymaga rozwiązywania równań Boltzmanna.

use crate::lcdm::cosmology::Cosmology;

pub struct PowerSpectrum {
    pub cosmology: Cosmology,
    /// Amplituda dobrana tak, by `σ(8 Mpc/h) = σ₈`.
    amplitude: f64,
}

impl PowerSpectrum {
    pub fn eisenstein_hu(cosmology: Cosmology) -> Self {
        // `σ_R` skaluje się jak `√amplituda`, więc jedno przejście wystarcza:
        // wyliczamy `σ₈` przy amplitudzie jednostkowej i podnosimy do kwadratu iloraz.
        let mut spectrum = Self {
            cosmology,
            amplitude: 1.0,
        };
        let unnormalized = spectrum.sigma_r(8.0);
        spectrum.amplitude = (cosmology.sigma8 / unnormalized).powi(2);
        spectrum
    }

    /// `T(k)`, dążące do jedności dla `k → 0`.
    pub fn transfer(&self, k: f64) -> f64 {
        eisenstein_hu_nowiggle(&self.cosmology, k.max(1e-8))
    }

    /// `P(k, a) = A k^{n_s} T²(k) D²(a)` w [(Mpc/h)³].
    pub fn p(&self, k: f64, a: f64) -> f64 {
        let d = self.cosmology.growth(a);
        self.p_today(k) * d * d
    }

    fn p_today(&self, k: f64) -> f64 {
        let t = self.transfer(k);
        self.amplitude * k.powf(self.cosmology.n_s) * t * t
    }

    /// `σ(R)` dziś: wariancja gęstości wygładzonej kulą o promieniu `R`.
    ///
    /// Całkowanie po `ln k`, bo `Δ²(k) = k³P/(2π²)` jest funkcją wolnozmienną
    /// w logarytmie i rozciąga się na cztery dekady; równomierna siatka w `k`
    /// wymagałaby setek tysięcy punktów na tę samą dokładność.
    pub fn sigma_r(&self, r: f64) -> f64 {
        const PANELS: usize = 256;
        let ln_lo = 1e-4_f64.ln();
        let ln_hi = 40.0_f64.ln();
        let dln = (ln_hi - ln_lo) / PANELS as f64;
        let sum: f64 = (0..=PANELS)
            .map(|i| {
                let k = (ln_lo + i as f64 * dln).exp();
                let w = top_hat(k * r);
                let delta2 = k.powi(3) * self.p_today(k) / (2.0 * std::f64::consts::PI.powi(2));
                let trapezoid = if i == 0 || i == PANELS { 0.5 } else { 1.0 };
                trapezoid * delta2 * w * w * dln
            })
            .sum();
        sum.sqrt()
    }
}

/// Funkcja przejścia bez oscylacji, wzory (28)–(31) z Eisenstein & Hu 1998.
fn eisenstein_hu_nowiggle(c: &Cosmology, k: f64) -> f64 {
    let h = c.h;
    let om_h2 = c.omega_m * h * h;
    let ob_h2 = c.omega_b * h * h;
    let theta = c.t_cmb / 2.7;
    // Skala horyzontu dźwięku [Mpc].
    let s = 44.5 * (9.83 / om_h2).ln() / (1.0 + 10.0 * ob_h2.powf(0.75)).sqrt();
    let alpha = 1.0 - 0.328 * (431.0 * om_h2).ln() * ob_h2 / om_h2
        + 0.38 * (22.3 * om_h2).ln() * (ob_h2 / om_h2).powi(2);
    let ks = 0.43 * k * s;
    let gamma_eff = c.omega_m * h * (alpha + (1.0 - alpha) / (1.0 + ks.powi(4)));
    let q = k * theta * theta / gamma_eff;
    let l = (2.0_f64.exp() + 1.8 * q).ln();
    let c0 = 14.2 + 731.0 / (1.0 + 62.5 * q);
    l / (l + c0 * q * q)
}

/// Transformata Fouriera kuli o promieniu jednostkowym.
///
/// Postać jawna ma w liczniku `sin x − x cos x ≈ x³/3`, czyli różnicę dwóch liczb
/// rzędu `x`. Błąd bezwzględny tego odejmowania jest rzędu `ε·x`, więc błąd względny
/// wyniku rośnie jak `ε/x²` — przy `x = 10⁻³` to już `10⁻¹⁰`, a przy `x = 10⁻⁵` całe
/// `10⁻⁶`. Dla małych `x` liczymy więc z rozwinięcia.
///
/// Punkt przejścia MUSI leżeć tam, gdzie oba wyrażenia są równie dokładne, a nie tam,
/// gdzie rozwinięcie przestaje być „oczywiście dobre": błąd rozwinięcia rośnie jak
/// `x⁶/15120`, błąd postaci jawnej maleje jak `ε/x²`, i zrównują się przy `x ≈ 0,05`.
/// Próg przesunięty w stronę małych `x` (np. `10⁻⁵`) zostawia postaci jawnej błąd
/// względny `10⁻⁶`, czyli robi w tym miejscu skok funkcji.
const TOP_HAT_SERIES_LIMIT: f64 = 0.05;

fn top_hat(x: f64) -> f64 {
    let x2 = x * x;
    if x.abs() < TOP_HAT_SERIES_LIMIT {
        1.0 - x2 / 10.0 + x2 * x2 / 280.0
    } else {
        3.0 * (x.sin() - x * x.cos()) / (x2 * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planck() -> PowerSpectrum {
        PowerSpectrum::eisenstein_hu(Cosmology::planck18())
    }

    #[test]
    fn transfer_tends_to_one_on_large_scales() {
        assert!((planck().transfer(1e-5) - 1.0).abs() < 2e-3);
    }

    #[test]
    fn transfer_falls_monotonically() {
        let p = planck();
        let mut previous = f64::INFINITY;
        for k in [0.01, 0.1, 1.0, 5.0, 20.0] {
            let t = p.transfer(k);
            assert!(t < previous, "T({k}) = {t} nie zmalało");
            previous = t;
        }
    }

    /// Normalizacja to jedyny powód, dla którego amplituda istnieje. Gdyby była
    /// spartaczona, całe widmo miałoby złą skalę, a warunki początkowe zbyt słaby
    /// albo zbyt silny kontrast — czego na obrazku nie da się odróżnić od fizyki.
    #[test]
    fn sigma8_is_normalized_to_the_requested_value() {
        let c = Cosmology::planck18();
        let s = PowerSpectrum::eisenstein_hu(c).sigma_r(8.0);
        assert!((s - c.sigma8).abs() < 1e-4, "σ₈ = {s}");
    }

    #[test]
    fn sigma_falls_with_the_smoothing_radius() {
        let p = planck();
        assert!(p.sigma_r(2.0) > p.sigma_r(8.0));
        assert!(p.sigma_r(8.0) > p.sigma_r(30.0));
    }

    /// `P` musi rosnąć z `a` dokładnie jak `D²` — to ta zależność przenosi
    /// normalizację z „dziś" na chwilę startu symulacji.
    #[test]
    fn power_grows_as_the_square_of_the_growth_factor() {
        let p = planck();
        let a = 0.02;
        let d = p.cosmology.growth(a);
        let ratio = p.p(0.3, a) / p.p(0.3, 1.0);
        assert!((ratio - d * d).abs() / (d * d) < 1e-12, "iloraz {ratio}");
    }

    /// Obie gałęzie muszą się zszywać bez skoku, bo `σ_R` całkuje po całym zakresie
    /// `kR` i przechodzi przez punkt przejścia. Skok w tym miejscu byłby błędem
    /// normalizacji, którego nie widać w żadnym pojedynczym wyniku.
    ///
    /// Rozbieżność w punkcie zszycia to pierwszy pominięty wyraz szeregu, `x⁶/15120`,
    /// czyli przy `x = 0,05` około `10⁻¹²`. Progu 10⁻¹¹ nie da się zacieśnić bez
    /// dopisania kolejnego wyrazu, a nie ma po co: `σ₈` jest znane na trzy cyfry.
    #[test]
    fn top_hat_branches_agree_at_the_switch() {
        let x = TOP_HAT_SERIES_LIMIT;
        let series = 1.0 - x * x / 10.0 + x.powi(4) / 280.0;
        let closed = 3.0 * (x.sin() - x * x.cos()) / x.powi(3);
        assert!((series - closed).abs() < 1e-11, "{series} vs {closed}");
        let next_term = x.powi(6) / 15_120.0;
        assert!((series - closed).abs() < 2.0 * next_term, "rozbieżność większa \
            niż pominięty wyraz szeregu — coś jeszcze się nie zgadza");
    }

    #[test]
    fn top_hat_is_one_at_zero_and_symmetric() {
        assert!((top_hat(0.0) - 1.0).abs() < 1e-15);
        assert!((top_hat(0.3) - top_hat(-0.3)).abs() < 1e-15);
        assert!((top_hat(3.0) - top_hat(-3.0)).abs() < 1e-15);
    }

    /// Pierwsze zero okna wypada przy `tan x = x`, czyli `x ≈ 4,4934`. To sprawdza
    /// gałąź jawną w zakresie, w którym naprawdę pracuje.
    #[test]
    fn top_hat_has_its_first_zero_in_the_right_place() {
        assert!(top_hat(4.4) > 0.0);
        assert!(top_hat(4.6) < 0.0);
        assert!(top_hat(4.4934).abs() < 1e-4);
    }
}
