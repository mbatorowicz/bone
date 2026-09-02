//! Kinematyka szczególnej teorii względności.
//!
//! Stanem cząstki jest pęd `p`, nie prędkość. Dzięki temu
//!
//! ```text
//! γ = E/(mc²) = √(1 + (|p|/mc)²)
//! ```
//!
//! jest zawsze skończone i ≥ 1, niezależnie od tego, jak duża jest siła. Żaden
//! clamp prędkości nie jest potrzebny — |v| → c asymptotycznie z samej definicji.
//! Wariant trzymający `v` wymagałby przycinania β² poniżej jedności, co maskuje
//! błędy całkowania zamiast im zapobiegać.

use crate::vec3::Vec3;

/// Masa, poniżej której cząstka przestaje mieć sens jako obiekt masywny.
///
/// Nie jest to zabezpieczenie przed dzieleniem przez zero „na wszelki wypadek":
/// przy `m = 0` wzór γ = E/(mc²) nie ma granicy, bo cząstka bezmasowa nie ma
/// układu spoczynkowego. Podłoga zamienia to na cząstkę bardzo lekką, czyli stan,
/// który dalsza arytmetyka umie opisać.
pub const MIN_MASS: f64 = 1e-12;

/// Prędkość, powyżej której warunek początkowy jest odrzucany jako nieosiągalny.
pub const MAX_INITIAL_BETA: f64 = 0.95;

pub fn rest_mass(mass: f64) -> f64 {
    mass.max(MIN_MASS)
}

/// `E/c = √(|p|² + (mc)²)`, liczone przez `hypot` — bez pośredniego kwadratu.
///
/// `hypot` nie przepełnia się dla dużych argumentów, więc pęd rzędu 1e200 daje
/// nadal poprawną energię, a nie `inf`.
pub fn energy_over_c(mass: f64, momentum: Vec3, c: f64) -> f64 {
    momentum.norm().hypot(rest_mass(mass) * c)
}

/// `γ = E/(mc²) = √(|p|² + (mc)²)/(mc)`. Zawsze ≥ 1, nigdy NaN.
pub fn gamma(mass: f64, momentum: Vec3, c: f64) -> f64 {
    energy_over_c(mass, momentum, c) / (rest_mass(mass) * c)
}

/// `v = p c²/E = p c/√(|p|² + (mc)²)`.
///
/// Liczone bez pośrednictwa γ, więc pozostaje poprawne nawet wtedy, gdy samo γ
/// wykracza poza zakres liczb zmiennoprzecinkowych. Gwarancja: |v| ≤ c zawsze,
/// a |v| < c dla każdego fizycznie sensownego pędu.
pub fn velocity(mass: f64, momentum: Vec3, c: f64) -> Vec3 {
    momentum * (c / energy_over_c(mass, momentum, c))
}

/// `p = γmv`.
///
/// Prędkości ≥ c są błędem wywołującego, nie stanem fizycznym, więc funkcja
/// zwraca `Err` zamiast po cichu przycinać. Ciche przycięcie tutaj oznaczałoby, że
/// zamówiony warunek początkowy różni się od policzonego i nikt się o tym nie
/// dowie.
pub fn momentum(mass: f64, velocity: Vec3, c: f64) -> Result<Vec3, SuperluminalError> {
    let beta2 = velocity.norm_squared() / (c * c);
    // NaN jest wyłapane osobno, a nie przez zaprzeczenie `beta2 < 1.0`: prędkość
    // nieoznaczona też nie jest dopuszczalnym wejściem, a przy zaprzeczeniu łatwo
    // przy kolejnej zmianie zgubić ten przypadek, bo nie widać go w kodzie.
    if beta2.is_nan() || beta2 >= 1.0 {
        return Err(SuperluminalError {
            beta: if beta2.is_nan() { f64::NAN } else { beta2.sqrt() },
        });
    }
    Ok(velocity * (rest_mass(mass) / (1.0 - beta2).sqrt()))
}

/// `T = (γ−1)mc²`.
///
/// Energia spoczynkowa jest pominięta, bo jest stałą ruchu i zagłuszyłaby dryf
/// o interesującej nas skali.
pub fn kinetic_energy(mass: f64, momentum: Vec3, c: f64) -> f64 {
    (gamma(mass, momentum, c) - 1.0) * rest_mass(mass) * c * c
}

/// `β = |p|/√(|p|² + (mc)²)`.
pub fn speed_over_c(mass: f64, momentum: Vec3, c: f64) -> f64 {
    momentum.norm() / energy_over_c(mass, momentum, c)
}

/// Przytnij prędkość do `MAX_INITIAL_BETA·c`, zwracając informację, czy zadziałało.
///
/// Używane wyłącznie przy składaniu warunku początkowego. Zwracany znacznik nie
/// jest ozdobą: obcięta orbita kołowa nie jest już kołowa, a układ może
/// wystartować z energią dodatnią i po prostu się rozlecieć. Wywołujący ma
/// obowiązek powiedzieć o tym wprost.
pub fn clamp_initial_speed(velocity: Vec3, c: f64) -> (Vec3, bool) {
    let limit = MAX_INITIAL_BETA * c;
    let speed = velocity.norm();
    if speed > limit && speed > 0.0 {
        (velocity * (limit / speed), true)
    } else {
        (velocity, false)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SuperluminalError {
    pub beta: f64,
}

impl std::fmt::Display for SuperluminalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prędkość początkowa β={:.4} ≥ 1 — pęd byłby nieskończony",
            self.beta
        )
    }
}

impl std::error::Error for SuperluminalError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    const C: f64 = 30.0;

    #[test]
    fn gamma_is_one_at_rest() {
        assert!((gamma(1.0, crate::vec3::ZERO, C) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn gamma_never_below_one() {
        for p in [1e-30, 1.0, 1e6, 1e200] {
            let g = gamma(1.0, vec3(p, 0.0, 0.0), C);
            assert!(g >= 1.0 && !g.is_nan(), "p={p} γ={g}");
        }
    }

    #[test]
    fn speed_stays_below_c() {
        for p in [1.0, 1e3, 1e12, 1e200] {
            let v = velocity(1.0, vec3(p, p, p), C).norm();
            assert!(v <= C, "p={p} |v|={v} > c");
        }
    }

    #[test]
    fn momentum_and_velocity_are_inverses() {
        for beta in [0.0, 0.1, 0.5, 0.9, 0.999] {
            let v = vec3(beta * C, 0.0, 0.0);
            let p = momentum(2.0, v, C).expect("β<1");
            let back = velocity(2.0, p, C);
            assert!((back.x - v.x).abs() < 1e-9, "β={beta}");
        }
    }

    #[test]
    fn momentum_rejects_lightspeed() {
        assert!(momentum(1.0, vec3(C, 0.0, 0.0), C).is_err());
        assert!(momentum(1.0, vec3(2.0 * C, 0.0, 0.0), C).is_err());
    }

    /// W granicy nierelatywistycznej `T` musi zgadzać się z ½mv² — to jedyny
    /// niezależny sprawdzian, że czynniki `c` są na swoich miejscach.
    #[test]
    fn kinetic_energy_matches_classical_limit() {
        let m = 3.0;
        let v = vec3(0.001 * C, 0.0, 0.0);
        let p = momentum(m, v, C).unwrap();
        let t = kinetic_energy(m, p, C);
        let classical = 0.5 * m * v.norm_squared();
        assert!(
            (t - classical).abs() / classical < 1e-5,
            "T={t} vs ½mv²={classical}"
        );
    }

    #[test]
    fn beta_and_velocity_agree() {
        let p = vec3(4.0, -2.0, 1.0);
        let b = speed_over_c(1.5, p, C);
        let v = velocity(1.5, p, C).norm() / C;
        assert!((b - v).abs() < 1e-12);
    }

    #[test]
    fn clamp_reports_when_it_bites() {
        let (v, hit) = clamp_initial_speed(vec3(0.5 * C, 0.0, 0.0), C);
        assert!(!hit && (v.norm() - 0.5 * C).abs() < 1e-12);
        let (v, hit) = clamp_initial_speed(vec3(10.0 * C, 0.0, 0.0), C);
        assert!(hit);
        assert!((v.norm() - MAX_INITIAL_BETA * C).abs() < 1e-12);
    }

    #[test]
    fn massless_particle_does_not_produce_nan() {
        let g = gamma(0.0, vec3(1.0, 0.0, 0.0), C);
        assert!(g.is_finite() && g >= 1.0, "γ={g}");
    }
}
