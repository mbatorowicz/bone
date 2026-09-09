//! Porównanie wzrostu: `δ_rms(a) / D(a)`.
//!
//! W reżimie liniowym amplituda zaburzeń rośnie jak czynnik wzrostu tła.
//! Iloraz jest wtedy stały — to liczba, której kosmolog szuka obok chmury,
//! nie residuum energii. Wykres jest serią punktów z silnika; ten moduł
//! tylko je składa, nic nie całkuje.

use crate::lcdm::cosmology::Cosmology;

#[derive(Clone, Debug, Default)]
pub struct Series {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

/// Jedna próbka z biegu: skala, zmierzone `σ(δ)` i iloraz przez `D(a)`.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub a: f64,
    pub delta_rms: f64,
    pub ratio: f64,
}

impl Sample {
    pub fn new(a: f64, delta_rms: f64, d_of_a: f64) -> Self {
        Self {
            a,
            delta_rms,
            ratio: delta_rms / d_of_a.max(1e-30),
        }
    }
}

/// Serie do panelu: krzywa tła `D(a)` oraz zmierzony iloraz wzdłuż biegu.
#[derive(Clone, Debug)]
pub struct Chart {
    pub growth: Series,
    pub ratio: Series,
    pub delta_rms: f64,
    pub d_of_a: f64,
    pub ratio_now: f64,
}

impl Chart {
    /// Krzywa tła jeszcze przed startem — suwak `z` ma zmieniać `D(a)` od razu.
    pub fn theoretical(cosmo: Cosmology, z_start: f64) -> Self {
        let a_start = 1.0 / (1.0 + z_start.max(0.0));
        let d = cosmo.growth(a_start);
        Self {
            growth: growth_curve(cosmo, a_start, 1.0, 64),
            ratio: Series::default(),
            delta_rms: 0.0,
            d_of_a: d,
            ratio_now: 0.0,
        }
    }

    pub fn from_run(
        cosmo: Cosmology,
        z_start: f64,
        log: &[Sample],
        now_a: f64,
        now_delta: f64,
    ) -> Self {
        let d = cosmo.growth(now_a);
        Self {
            growth: growth_curve(cosmo, 1.0 / (1.0 + z_start.max(0.0)), 1.0, 64),
            ratio: Series {
                xs: log.iter().map(|s| s.a).collect(),
                ys: log.iter().map(|s| s.ratio).collect(),
            },
            delta_rms: now_delta,
            d_of_a: d,
            ratio_now: now_delta / d.max(1e-30),
        }
    }
}

/// `D(a)` równo w `ln a`, bo wzrost jest potęgowy i równomierna siatka w `a`
/// zjada wczesny wszechświat.
pub fn growth_curve(cosmo: Cosmology, a_min: f64, a_max: f64, n: usize) -> Series {
    let n = n.max(2);
    let lo = a_min.max(1e-4);
    let hi = a_max.max(lo * 1.01);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let ln_lo = lo.ln();
    let ln_span = hi.ln() - ln_lo;
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        let a = (ln_lo + t * ln_span).exp();
        xs.push(a);
        ys.push(cosmo.growth(a));
    }
    Series { xs, ys }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_curve_rises_to_unity_today() {
        let s = growth_curve(Cosmology::planck18(), 0.02, 1.0, 24);
        assert_eq!(s.xs.len(), s.ys.len());
        for w in s.ys.windows(2) {
            assert!(w[1] > w[0], "D zmalało: {} → {}", w[0], w[1]);
        }
        assert!((s.ys.last().copied().unwrap_or(0.0) - 1.0).abs() < 2e-3);
    }

    #[test]
    fn theoretical_chart_has_no_measured_ratio() {
        let chart = Chart::theoretical(Cosmology::planck18(), 49.0);
        assert!(chart.ratio.xs.is_empty());
        assert!(chart.d_of_a > 0.0);
        assert!(chart.d_of_a < 0.05, "D(z=49) = {}", chart.d_of_a);
    }

    #[test]
    fn sample_ratio_is_delta_over_growth() {
        let s = Sample::new(0.5, 0.12, 0.6);
        assert!((s.ratio - 0.2).abs() < 1e-12);
    }
}
