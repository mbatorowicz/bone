//! Dokładne stany związane atomu wodoropodobnego — równanie Schrödingera, nie
//! model Bohra.
//!
//! Funkcja radialna i harmoniczne sferyczne są tu policzone z wielomianów, nie
//! z tablicy. Dzięki temu `n, l, m, Z` są suwakami, a nie listą obrazków. Wzory
//! są sprawdzane na zamkniętych postaciach `1s`, `2s`, `2p` i na normowaniu;
//! jeśli rekurencja się rozjedzie, test to złapie zanim panel pokaże ładny błąd.

use std::f64::consts::{PI, TAU};

/// Czy `(n, l, m)` jest dozwolone. `n ≥ 1`, `0 ≤ l < n`, `|m| ≤ l`.
pub fn quantum_ok(n: u32, l: u32, m: i32) -> bool {
    n >= 1 && l < n && m.unsigned_abs() <= l
}

pub fn spectroscopic(l: u32) -> char {
    match l {
        0 => 's',
        1 => 'p',
        2 => 'd',
        3 => 'f',
        4 => 'g',
        5 => 'h',
        _ => '?',
    }
}

/// Etykieta orbitalu w konwencji chemicznej: `2p_z`, `3d_{x²−y²}`.
pub fn orbital_label(n: u32, l: u32, m: i32) -> String {
    let letter = spectroscopic(l);
    if l == 0 {
        return format!("{n}{letter}");
    }
    let suffix = match (l, m) {
        (1, 0) => "_z",
        (1, 1) => "_x",
        (1, -1) => "_y",
        (2, 0) => "_z²",
        (2, 1) => "_xz",
        (2, -1) => "_yz",
        (2, 2) => "_{x²−y²}",
        (2, -2) => "_xy",
        _ if m == 0 => "_0",
        _ if m > 0 => return format!("{n}{letter}_c{m}"),
        _ => return format!("{n}{letter}_s{}", m.unsigned_abs()),
    };
    format!("{n}{letter}{suffix}")
}

/// Energia `E_n = −μ Z² / (2 n²)` hartree. `mu` to stosunek zredukowanej masy
/// do `m_e`; `1` znaczy nieskończone jądro.
pub fn energy_hartree(z: f64, n: u32, mu: f64) -> f64 {
    let n = n.max(1) as f64;
    -mu * z * z / (2.0 * n * n)
}

/// Promień orbity Bohra `n² a₀ / Z` — **nie** wartość oczekiwana `⟨r⟩`.
pub fn bohr_radius(z: f64, n: u32) -> f64 {
    let n = n.max(1) as f64;
    let z = z.abs().max(1e-12);
    n * n / z
}

/// `⟨r⟩_{nl} = (a₀ / 2Z) [3n² − l(l+1)]`. Zależy od `n` i `l`, nie od `m`.
pub fn mean_radius(z: f64, n: u32, l: u32) -> f64 {
    let n = n.max(1) as f64;
    let l = l as f64;
    let z = z.abs().max(1e-12);
    (3.0 * n * n - l * (l + 1.0)) / (2.0 * z)
}

fn log_factorial(n: u32) -> f64 {
    (2..=n).map(|k| (k as f64).ln()).sum()
}

/// Uogólniony wielomian Laguerre’a `L_k^{(α)}(x)` rekurencją.
pub fn laguerre(k: u32, alpha: u32, x: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let a = alpha as f64;
    let mut prev = 1.0;
    let mut curr = 1.0 + a - x;
    if k == 1 {
        return curr;
    }
    for i in 1..k {
        let ii = i as f64;
        let next = (((2.0 * ii + 1.0 + a - x) * curr) - ((ii + a) * prev)) / (ii + 1.0);
        prev = curr;
        curr = next;
    }
    curr
}

/// Funkcja radialna `R_{nl}(r)` w jednostkach atomowych, jądro o ładunku `z`.
pub fn radial(z: f64, n: u32, l: u32, r: f64) -> f64 {
    if !quantum_ok(n, l, 0) || r < 0.0 || !r.is_finite() || z <= 0.0 {
        return 0.0;
    }
    if l > 0 && r == 0.0 {
        return 0.0;
    }
    let n_f = n as f64;
    let rho = 2.0 * z * r / n_f;
    let k = n - l - 1;
    let alpha = 2 * l + 1;
    let log_norm = 1.5 * (2.0 * z / n_f).ln() + 0.5 * log_factorial(k)
        - 0.5 * (2.0 * n_f).ln()
        - 0.5 * log_factorial(n + l);
    let log_weight = -0.5 * rho
        + if l == 0 || rho <= 0.0 {
            0.0
        } else {
            f64::from(l) * rho.ln()
        };
    // W ogonie `ρ^{n−1} e^{−ρ/2}` zawsze wygrywa z wielomianem. Gdybyśmy
    // pomnożyli najpierw `L` przez `e^{−ρ/2}`, dostałoby się `inf * 0 = NaN`.
    if log_norm + log_weight < -700.0 {
        return 0.0;
    }
    let poly = laguerre(k, alpha, rho);
    if !poly.is_finite() {
        return 0.0;
    }
    let signed_log = poly.abs().max(f64::MIN_POSITIVE).ln() + log_norm + log_weight;
    if signed_log < -700.0 {
        return 0.0;
    }
    poly.signum() * signed_log.exp()
}

/// Gęstość radialna `P(r) = r² R²` — prawdopodobieństwo w powłoce `dr`.
pub fn radial_probability(z: f64, n: u32, l: u32, r: f64) -> f64 {
    let rnl = radial(z, n, l, r);
    r * r * rnl * rnl
}

/// `P_l^{|m|}(x)` **bez** fazy Condona–Shortleya. Dodatnie `P_m^m` dają
/// chemiczne `p_x`, `p_y`, `p_z` bez ręcznego odwracania znaku.
pub fn assoc_legendre(l: u32, m_abs: u32, x: f64) -> f64 {
    let x = x.clamp(-1.0, 1.0);
    if m_abs > l {
        return 0.0;
    }
    let somx2 = (1.0 - x * x).max(0.0);
    let mut pmm = 1.0;
    if m_abs > 0 {
        let mut fact = 1.0;
        for _ in 1..=m_abs {
            pmm *= fact * somx2.sqrt();
            fact += 2.0;
        }
    }
    if l == m_abs {
        return pmm;
    }
    let mut pmmp1 = x * (2 * m_abs + 1) as f64 * pmm;
    if l == m_abs + 1 {
        return pmmp1;
    }
    let mut pll = 0.0;
    for ll in (m_abs + 2)..=l {
        pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m_abs - 1) as f64 * pmm) / (ll - m_abs) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

fn spherical_norm(l: u32, m_abs: u32) -> f64 {
    let num = log_factorial(l - m_abs);
    let den = log_factorial(l + m_abs);
    ((2 * l + 1) as f64 / (4.0 * PI) * (num - den).exp()).sqrt()
}

/// Rzeczywista harmoniczna sferyczna (konwencja chemiczna).
///
/// `m > 0` → człon cosinusowy (`p_x`, `d_{xz}`, …), `m < 0` → sinusowy
/// (`p_y`, `d_{yz}`, …), `m = 0` → `Y_{l0}`.
pub fn real_harmonic(l: u32, m: i32, theta: f64, phi: f64) -> f64 {
    let m_abs = m.unsigned_abs();
    if m_abs > l {
        return 0.0;
    }
    let x = theta.cos();
    let p = assoc_legendre(l, m_abs, x);
    let n = spherical_norm(l, m_abs);
    if m == 0 {
        n * p
    } else if m > 0 {
        std::f64::consts::SQRT_2 * n * p * (m_abs as f64 * phi).cos()
    } else {
        std::f64::consts::SQRT_2 * n * p * (m_abs as f64 * phi).sin()
    }
}

/// Rzeczywista funkcja falowa `ψ_{nlm} = R_{nl}(r) Y_{lm}(θ,φ)`.
pub fn wavefunction(z: f64, n: u32, l: u32, m: i32, x: f64, y: f64, zc: f64) -> f64 {
    if !quantum_ok(n, l, m) {
        return 0.0;
    }
    let r = (x * x + y * y + zc * zc).sqrt();
    let theta = if r == 0.0 { 0.0 } else { (zc / r).acos() };
    let phi = y.atan2(x);
    radial(z, n, l, r) * real_harmonic(l, m, theta, phi)
}

pub fn density(z: f64, n: u32, l: u32, m: i32, x: f64, y: f64, zc: f64) -> f64 {
    let psi = wavefunction(z, n, l, m, x, y, zc);
    psi * psi
}

/// Kąty ze współrzędnych kartezjańskich; w początku układu `θ = 0`.
pub fn angles(x: f64, y: f64, zc: f64) -> (f64, f64, f64) {
    let r = (x * x + y * y + zc * zc).sqrt();
    let theta = if r == 0.0 { 0.0 } else { (zc / r).clamp(-1.0, 1.0).acos() };
    (r, theta, y.atan2(x))
}

/// Okres bicia dwóch poziomów `2π ħ / |E_a − E_b|` w jednostkach atomowych.
pub fn beat_period(e_a: f64, e_b: f64) -> Option<f64> {
    let d = (e_a - e_b).abs();
    if d < 1e-18 {
        None
    } else {
        Some(TAU / d)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_radial_functions_match() {
        let r = 0.7_f64;
        let r10 = 2.0 * (-r).exp();
        assert!((radial(1.0, 1, 0, r) - r10).abs() < 1e-12, "{}", radial(1.0, 1, 0, r));

        let r20 = (1.0 / std::f64::consts::SQRT_2) * (1.0 - r / 2.0) * (-r / 2.0).exp();
        assert!((radial(1.0, 2, 0, r) - r20).abs() < 1e-12);

        let r21 = (r / 24.0_f64.sqrt()) * (-r / 2.0).exp();
        assert!((radial(1.0, 2, 1, r) - r21).abs() < 1e-12);
    }

    #[test]
    fn r10_at_origin_is_two() {
        assert!((radial(1.0, 1, 0, 0.0) - 2.0).abs() < 1e-12);
        assert_eq!(radial(1.0, 2, 1, 0.0), 0.0);
    }

    #[test]
    fn he_plus_squeezes_the_1s_orbital() {
        // R_Z(r) = Z^{3/2} R_1(Zr); w zerze He⁺ jest 2·2^{3/2} = 4√2.
        let he = radial(2.0, 1, 0, 0.0);
        assert!((he - 4.0 * std::f64::consts::SQRT_2).abs() < 1e-12, "{he}");
    }

    fn integrate_radial(z: f64, n: u32, l: u32) -> f64 {
        let r_max = 40.0 * bohr_radius(z, n);
        let steps = 4_000;
        let dr = r_max / steps as f64;
        let mut acc = 0.0;
        for i in 0..=steps {
            let r = i as f64 * dr;
            let w = if i == 0 || i == steps {
                1.0
            } else if i % 2 == 0 {
                2.0
            } else {
                4.0
            };
            acc += w * radial_probability(z, n, l, r);
        }
        acc * dr / 3.0
    }

    #[test]
    fn radial_functions_are_normalized() {
        for (z, n, l) in [(1.0, 1, 0), (1.0, 2, 0), (1.0, 2, 1), (1.0, 3, 2), (2.0, 1, 0)] {
            let norm = integrate_radial(z, n, l);
            assert!(
                (norm - 1.0).abs() < 2e-3,
                "Z={z} n={n} l={l}  ∫r²R² = {norm}"
            );
        }
    }

    #[test]
    fn mean_radius_matches_the_textbook_formula() {
        assert!((mean_radius(1.0, 1, 0) - 1.5).abs() < 1e-12);
        assert!((mean_radius(1.0, 2, 1) - 5.0).abs() < 1e-12);
        assert!((mean_radius(2.0, 1, 0) - 0.75).abs() < 1e-12);
        assert!(bohr_radius(1.0, 1) < mean_radius(1.0, 1, 0));
    }

    #[test]
    fn energy_goes_as_minus_z_squared_over_n_squared() {
        assert!((energy_hartree(1.0, 1, 1.0) + 0.5).abs() < 1e-15);
        assert!((energy_hartree(2.0, 1, 1.0) + 2.0).abs() < 1e-15);
        assert!((energy_hartree(1.0, 2, 1.0) + 0.125).abs() < 1e-15);
    }

    #[test]
    fn y00_is_isotropic_and_y10_peaks_on_z() {
        let y00 = 1.0 / (2.0 * PI.sqrt());
        assert!((real_harmonic(0, 0, 0.4, 1.2) - y00).abs() < 1e-12);
        let y10_north = (3.0 / (4.0 * PI)).sqrt();
        assert!((real_harmonic(1, 0, 0.0, 0.0) - y10_north).abs() < 1e-12);
        assert!(real_harmonic(1, 0, PI / 2.0, 0.0).abs() < 1e-12);
        // p_x na osi x: θ=π/2, φ=0.
        let px = (3.0 / (4.0 * PI)).sqrt();
        assert!((real_harmonic(1, 1, PI / 2.0, 0.0) - px).abs() < 1e-12);
        // p_y na osi y: θ=π/2, φ=π/2.
        assert!((real_harmonic(1, -1, PI / 2.0, PI / 2.0) - px).abs() < 1e-12);
    }

    #[test]
    fn p_orbital_vanishes_at_the_nucleus_and_has_two_lobes() {
        assert_eq!(density(1.0, 2, 1, 0, 0.0, 0.0, 0.0), 0.0);
        let north = density(1.0, 2, 1, 0, 0.0, 0.0, 4.0);
        let south = density(1.0, 2, 1, 0, 0.0, 0.0, -4.0);
        let equator = density(1.0, 2, 1, 0, 4.0, 0.0, 0.0);
        assert!((north - south).abs() / north < 1e-12);
        assert!(north > 10.0 * equator, "{north} vs {equator}");
        let psi_n = wavefunction(1.0, 2, 1, 0, 0.0, 0.0, 4.0);
        let psi_s = wavefunction(1.0, 2, 1, 0, 0.0, 0.0, -4.0);
        assert!(psi_n * psi_s < 0.0, "płaty 2p_z muszą mieć przeciwny znak");
    }

    #[test]
    fn forbidden_quantum_numbers_are_rejected() {
        assert!(!quantum_ok(0, 0, 0));
        assert!(!quantum_ok(1, 1, 0));
        assert!(!quantum_ok(2, 1, 2));
        assert!(quantum_ok(3, 2, -2));
        assert_eq!(wavefunction(1.0, 1, 1, 0, 1.0, 0.0, 0.0), 0.0);
    }

    #[test]
    fn labels_use_chemistry_names() {
        assert_eq!(orbital_label(1, 0, 0), "1s");
        assert_eq!(orbital_label(2, 1, 0), "2p_z");
        assert_eq!(orbital_label(2, 1, 1), "2p_x");
        assert_eq!(orbital_label(3, 2, 0), "3d_z²");
    }

    #[test]
    fn one_s_two_p_beat_is_a_fraction_of_a_femtosecond() {
        let t = beat_period(energy_hartree(1.0, 1, 1.0), energy_hartree(1.0, 2, 1.0)).unwrap();
        assert!((t - std::f64::consts::TAU / 0.375).abs() < 1e-12);
        assert!(crate::qm::units::atomic_to_fs(t) < 1.0);
        assert!(beat_period(-0.5, -0.5).is_none());
    }

    #[test]
    fn far_tail_is_zero_not_nan() {
        let v = radial(1.0, 8, 1, 2_000.0);
        assert!(v.is_finite(), "{v}");
        assert!(v.abs() < 1e-40, "{v}");
    }
}
