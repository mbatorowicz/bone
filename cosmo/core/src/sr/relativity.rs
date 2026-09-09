//! Kinematyka N-ciał: Newton albo szczególna teoria względności.
//!
//! Stanem cząstki jest pęd `p`, nie prędkość. Leapfrog jest ten sam
//! (`dp/dt = F`, `dx/dt = v(p)`); przełącznik decyduje tylko, jak pęd
//! przechodzi na prędkość i energię.
//!
//! ```text
//! Newton:  v = p/m,           E = |p|²/(2m)
//! SR:      v = p c²/E,        E = (γ−1)mc²,   γ = √(1 + (|p|/mc)²)
//! ```
//!
//! W SR |v| → c asymptotycznie z samej definicji — żaden clamp prędkości nie jest
//! potrzebny. W Newtonie |v| może przekroczyć c: to jest wynik, nie błąd.

use serde::{Deserialize, Serialize};

use crate::vec3::Vec3;

/// Wybór kinematyki. Siła zostaje newtonowska; zmienia się tylko `p ↔ v` i energia.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kinematics {
    Newton,
    #[default]
    Sr,
}

impl Kinematics {
    pub const ALL: [Kinematics; 2] = [Self::Newton, Self::Sr];

    pub fn label(self) -> &'static str {
        match self {
            Self::Newton => "Newton",
            Self::Sr => "SR",
        }
    }

    /// `v(p)`: Newton `p/m`, SR `p c²/E`.
    pub fn velocity(self, mass: f64, momentum: Vec3, c: f64) -> Vec3 {
        match self {
            Self::Newton => momentum / rest_mass(mass),
            Self::Sr => velocity(mass, momentum, c),
        }
    }

    /// `p(v)`: Newton zawsze `mv`; SR odrzuca |v| ≥ c.
    pub fn momentum(self, mass: f64, velocity: Vec3, c: f64) -> Result<Vec3, SuperluminalError> {
        match self {
            Self::Newton => Ok(velocity * rest_mass(mass)),
            Self::Sr => momentum(mass, velocity, c),
        }
    }

    /// Energia kinetyczna: Newton `|p|²/2m`, SR `(γ−1)mc²`.
    pub fn kinetic_energy(self, mass: f64, momentum: Vec3, c: f64) -> f64 {
        match self {
            Self::Newton => momentum.norm_squared() / (2.0 * rest_mass(mass)),
            Self::Sr => kinetic_energy(mass, momentum, c),
        }
    }

    /// `γ`: w Newtonie zawsze 1 — ta kinematyka nie ma czynnika Lorentza.
    pub fn gamma(self, mass: f64, momentum: Vec3, c: f64) -> f64 {
        match self {
            Self::Newton => 1.0,
            Self::Sr => gamma(mass, momentum, c),
        }
    }

    /// `|v|/c`. W Newtonie może być > 1.
    pub fn speed_over_c(self, mass: f64, momentum: Vec3, c: f64) -> f64 {
        match self {
            Self::Newton => momentum.norm() / (rest_mass(mass) * c.max(1e-300)),
            Self::Sr => speed_over_c(mass, momentum, c),
        }
    }

    /// Przytnij warunek początkowy do `MAX_INITIAL_BETA·c`. Newton nie przycina:
    /// nadświetlność jest tam dozwolonym wynikiem, nie błędem wejścia.
    pub fn clamp_initial(self, velocity: Vec3, c: f64) -> (Vec3, bool) {
        match self {
            Self::Newton => (velocity, false),
            Self::Sr => clamp_initial_speed(velocity, c),
        }
    }
}

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

    /// Przy małym β oba wzory energii muszą się zgadzać — inaczej przełącznik
    /// kłamałby już w granicy, w której Newton i SR są tym samym.
    #[test]
    fn small_beta_newton_and_sr_energies_agree() {
        let m = 3.0;
        let v = vec3(0.001 * C, 0.0, 0.0);
        let p_n = Kinematics::Newton.momentum(m, v, C).unwrap();
        let p_s = Kinematics::Sr.momentum(m, v, C).unwrap();
        let t_n = Kinematics::Newton.kinetic_energy(m, p_n, C);
        let t_s = Kinematics::Sr.kinetic_energy(m, p_s, C);
        assert!(
            (t_n - t_s).abs() / t_n.max(t_s) < 1e-5,
            "Newton {t_n} vs SR {t_s}"
        );
        let classical = 0.5 * m * v.norm_squared();
        assert!((t_n - classical).abs() / classical < 1e-12);
    }

    /// Przy dużym pędzie Newton daje |v| > c; SR nigdy.
    #[test]
    fn large_beta_only_sr_stays_below_c() {
        let m = 1.0;
        let p = vec3(10.0 * m * C, 0.0, 0.0);
        let v_n = Kinematics::Newton.velocity(m, p, C).norm();
        let v_s = Kinematics::Sr.velocity(m, p, C).norm();
        assert!(v_n > C, "Newton |v|={v_n} miało przekroczyć c={C}");
        assert!(v_s < C, "SR |v|={v_s} nie wolno przekroczyć c");
        assert_eq!(Kinematics::Newton.gamma(m, p, C), 1.0);
        assert!(Kinematics::Newton.speed_over_c(m, p, C) > 1.0);
        assert!(Kinematics::Sr.speed_over_c(m, p, C) < 1.0);
    }

    #[test]
    fn newton_accepts_superluminal_initial_velocity() {
        let v = vec3(2.0 * C, 0.0, 0.0);
        assert!(Kinematics::Newton.momentum(1.0, v, C).is_ok());
        assert!(Kinematics::Sr.momentum(1.0, v, C).is_err());
        let (clamped, hit) = Kinematics::Newton.clamp_initial(v, C);
        assert!(!hit && (clamped.x - v.x).abs() < 1e-15);
    }
}
