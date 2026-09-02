//! Deterministyczny generator liczb losowych dla warunków początkowych.
//!
//! ChaCha8 zamiast generatora domyślnego, bo strumień musi być powtarzalny między
//! wersjami biblioteki i między platformami — inaczej „to samo ziarno" przestaje
//! znaczyć „ten sam bieg", a wtedy porównywanie dwóch przebiegów przestaje mieć
//! sens. `rand::thread_rng` tej gwarancji nie daje.
//!
//! Rozkład normalny jest tu policzony wprost metodą Boxa-Mullera, zamiast przez
//! zależność zewnętrzną. Powód jest ten sam: transformacja z dwóch liczb
//! jednostajnych jest zapisana w trzech linijkach i jej wynik nie zmieni się przy
//! aktualizacji, a losowanie warunku początkowego jest miejscem, w którym cicha
//! zmiana strumienia unieważnia zapisane biegi.

use rand::Rng as _;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::vec3::{vec3, Vec3};

pub struct Rng {
    inner: ChaCha8Rng,
    /// Druga liczba z pary Boxa-Mullera; wyrzucanie jej byłoby marnowaniem połowy
    /// wywołań, a przy 120 tys. cząstek × 3 składowe to zauważalna różnica.
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn seeded(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
            spare_normal: None,
        }
    }

    /// Jednostajna z [0, 1).
    pub fn unit(&mut self) -> f64 {
        self.inner.gen::<f64>()
    }

    /// Jednostajna z [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Standardowa normalna N(0, 1).
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        // u1 odsunięte od zera, bo ln(0) = −∞
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        self.spare_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    pub fn normal(&mut self, mean: f64, sigma: f64) -> f64 {
        mean + sigma * self.standard_normal()
    }

    pub fn normal_vec(&mut self, mean: f64, sigma: f64) -> Vec3 {
        vec3(
            self.normal(mean, sigma),
            self.normal(mean, sigma),
            self.normal(mean, sigma),
        )
    }

    /// Kierunek rozłożony jednostajnie po sferze.
    ///
    /// Trzy niezależne normalne dają rozkład sferycznie symetryczny, więc po
    /// normalizacji wychodzi rozkład jednostajny na sferze. Losowanie kątów
    /// niezależnie (θ, φ jednostajne) daje zagęszczenie przy biegunach — to
    /// najczęstszy błąd w tym miejscu i dlatego jest tu wypisany.
    pub fn unit_vector(&mut self) -> Vec3 {
        loop {
            let v = self.normal_vec(0.0, 1.0);
            let n = v.norm();
            if n > 1e-12 {
                return v / n;
            }
        }
    }

    /// `k` różnych indeksów z zakresu `0..n`, bez powtórzeń.
    ///
    /// Częściowe tasowanie Fishera-Yatesa: koszt O(k), a nie O(n), więc próbkowanie
    /// 512 cząstek z miliona nie wymaga zbudowania miliona elementów. Mapa
    /// podmienionych pozycji zastępuje pełną tablicę permutacji.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        let k = k.min(n);
        let mut swapped: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut out = Vec::with_capacity(k);
        for i in 0..k {
            let j = i + (self.unit() * (n - i) as f64) as usize;
            let j = j.min(n - 1);
            let value = *swapped.get(&j).unwrap_or(&j);
            out.push(value);
            let at_i = *swapped.get(&i).unwrap_or(&i);
            swapped.insert(j, at_i);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_stream() {
        let mut a = Rng::seeded(7);
        let mut b = Rng::seeded(7);
        for _ in 0..64 {
            assert_eq!(a.unit(), b.unit());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Rng::seeded(1);
        let mut b = Rng::seeded(2);
        let differs = (0..32).any(|_| a.unit() != b.unit());
        assert!(differs);
    }

    #[test]
    fn uniform_stays_in_range() {
        let mut r = Rng::seeded(3);
        for _ in 0..1000 {
            let v = r.uniform(-2.0, 5.0);
            assert!((-2.0..5.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn normal_has_expected_moments() {
        let mut r = Rng::seeded(11);
        let n = 20_000;
        let samples: Vec<f64> = (0..n).map(|_| r.standard_normal()).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.05, "średnia={mean}");
        assert!((var - 1.0).abs() < 0.05, "wariancja={var}");
    }

    #[test]
    fn unit_vectors_are_normalized_and_isotropic() {
        let mut r = Rng::seeded(5);
        let n = 20_000;
        let mut sum = crate::vec3::ZERO;
        for _ in 0..n {
            let v = r.unit_vector();
            assert!((v.norm() - 1.0).abs() < 1e-12);
            sum += v;
        }
        // Rozkład izotropowy ma zerową średnią; odchylenie maleje jak 1/√N.
        let anisotropy = sum.norm() / n as f64;
        assert!(anisotropy < 0.05, "anizotropia {anisotropy}");
    }

    #[test]
    fn sampled_indices_are_unique_and_in_range() {
        let mut r = Rng::seeded(13);
        let picked = r.sample_indices(1000, 200);
        assert_eq!(picked.len(), 200);
        let unique: std::collections::BTreeSet<_> = picked.iter().copied().collect();
        assert_eq!(unique.len(), 200, "indeksy się powtarzają");
        assert!(picked.iter().all(|i| *i < 1000));
    }

    #[test]
    fn sampling_more_than_available_returns_everything() {
        let mut r = Rng::seeded(17);
        let picked = r.sample_indices(10, 50);
        assert_eq!(picked.len(), 10);
        let unique: std::collections::BTreeSet<_> = picked.iter().copied().collect();
        assert_eq!(unique.len(), 10);
    }
}
