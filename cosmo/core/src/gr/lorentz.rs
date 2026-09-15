//! Boost Lorentza między układami inercjalnymi, w jednostkach `c = 1`.
//!
//! Współrzędna czasowa jest już `ct`, więc wzory z lekcji wkleja się bez
//! dodatkowej stałej. `β = v/c` jest bezwymiarowe, `|β| < 1` jest warunkiem
//! istnienia transformacji — nie przycinamy go po cichu.
//!
//! To nie jest [`crate::sr::relativity`]. Tam stanem cząstki jest pęd, a `γ`
//! liczy się z `E/(mc²)`. Tu nie ma masy ani siły: jest tylko adres wydarzenia
//! i zmiana układu, w którym ten adres odczytujemy.

use std::ops::Sub;

use crate::vec3::{vec3, Vec3};

/// Znacznik na scenie: kiedy (`ct`) i gdzie (`x, y, z`).
///
/// Cztery liczby trzymamy razem, bo boost miesza czas z miejscem. Osobna para
/// `(t, x)` kusiłaby, żeby transformować tylko jedną składową — a wtedy
/// jednoczesność i interwał rozjadą się przy pierwszej animacji.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event {
    pub ct: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Event {
    pub fn new(ct: f64, x: f64, y: f64, z: f64) -> Self {
        Self { ct, x, y, z }
    }

    /// Zdarzenie na osi `ct`–`x`, jak w lekcjach 1–3.
    pub fn on_axis(ct: f64, x: f64) -> Self {
        Self::new(ct, x, 0.0, 0.0)
    }

    pub fn position(self) -> Vec3 {
        vec3(self.x, self.y, self.z)
    }

    /// Interwał Minkowskiego względem początku: `s² = (ct)² − x² − y² − z²`.
    ///
    /// Znak minus przy przestrzeni to cała STW w jednym miejscu. `s² > 0` to
    /// para jak tyknięcie zegara, `s² = 0` to światło, `s² < 0` to linijka.
    pub fn interval_sq(self) -> f64 {
        self.ct * self.ct - self.x * self.x - self.y * self.y - self.z * self.z
    }

    /// `s²` między dwoma zdarzeniami — ten sam wzór na różnicy współrzędnych.
    pub fn interval_sq_to(self, other: Self) -> f64 {
        (self - other).interval_sq()
    }
}

impl Sub for Event {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            ct: self.ct - other.ct,
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// `|β| ≥ 1` albo `NaN` — transformacja Lorentza wtedy nie istnieje.
#[derive(Clone, Copy, Debug)]
pub struct Superluminal {
    pub beta: f64,
}

impl std::fmt::Display for Superluminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "prędkość β={:.4} ≥ 1 — transformacja Lorentza nie istnieje",
            self.beta
        )
    }
}

impl std::error::Error for Superluminal {}

fn reject_beta2(beta2: f64) -> Result<(), Superluminal> {
    if beta2.is_nan() || beta2 >= 1.0 {
        Err(Superluminal {
            beta: if beta2.is_nan() {
                f64::NAN
            } else {
                beta2.sqrt()
            },
        })
    } else {
        Ok(())
    }
}

fn reject_beta_1d(beta: f64) -> Result<(), Superluminal> {
    if beta.is_nan() || beta.abs() >= 1.0 {
        Err(Superluminal { beta })
    } else {
        Ok(())
    }
}

fn gamma_from_beta2(beta2: f64) -> Result<f64, Superluminal> {
    reject_beta2(beta2)?;
    Ok(1.0 / (1.0 - beta2).sqrt())
}

/// `γ = 1 / √(1 − β²)`. Zawsze ≥ 1, gdy istnieje.
pub fn gamma(beta: f64) -> Result<f64, Superluminal> {
    reject_beta_1d(beta)?;
    gamma_from_beta2(beta * beta)
}

/// `γ` z wektora `β`. `|β|` jest prędkością układu w jednostkach `c`.
pub fn gamma_from_beta(beta: Vec3) -> Result<f64, Superluminal> {
    gamma_from_beta2(beta.norm_squared())
}

/// Czas laboratoryjny, w którym poruszający się zegar odmierza `proper_time`.
///
/// `Δt = γ Δτ`. Przy `β = 0` jest `γ = 1` i oba czasy są tym samym.
pub fn dilated_time(proper_time: f64, beta: f64) -> Result<f64, Superluminal> {
    Ok(gamma(beta)? * proper_time)
}

/// Długość wzdłuż ruchu: `L = L₀ / γ`.
///
/// Kierunek prostopadły do `v` nie wchodzi do tego wzoru — patrz [`contract_rod`].
pub fn contracted_length(proper_length: f64, beta: f64) -> Result<f64, Superluminal> {
    Ok(proper_length / gamma(beta)?)
}

/// Linijka o wektorze spoczynkowym `proper`, mierzona gdy leci z prędkością `beta`.
///
/// Składowa równoległa do `β` kurczy się przez `γ`, prostopadła zostaje.
/// Inaczej animacja „spłaszczałaby" linijkę we wszystkie strony.
pub fn contract_rod(proper: Vec3, beta: Vec3) -> Result<Vec3, Superluminal> {
    let beta2 = beta.norm_squared();
    let g = gamma_from_beta2(beta2)?;
    if beta2 == 0.0 {
        return Ok(proper);
    }
    let parallel = beta * (beta.dot(proper) / beta2);
    Ok(parallel / g + (proper - parallel))
}

/// Złożenie dwóch boostów wzdłuż tej samej osi: `(β₁ + β₂) / (1 + β₁ β₂)`.
///
/// Wynik zostaje `|β| < 1`, o ile oba składniki są dopuszczalne. To ten sam
/// wzór, który składa dwie transformacje [`boost_x`] w jedną.
pub fn compose_boost_1d(beta1: f64, beta2: f64) -> Result<f64, Superluminal> {
    reject_beta_1d(beta1)?;
    reject_beta_1d(beta2)?;
    Ok((beta1 + beta2) / (1.0 + beta1 * beta2))
}

/// Boost wzdłuż `x`: układ primowany jedzie z prędkością `β` względem naszego.
///
/// ```text
/// ct' = γ (ct − β x)
/// x'  = γ (x  − β ct)
/// y'  = y
/// z'  = z
/// ```
pub fn boost_x(event: Event, beta: f64) -> Result<Event, Superluminal> {
    boost(event, vec3(beta, 0.0, 0.0))
}

/// Boost o dowolny `β`, `|β| < 1`.
///
/// Rozkład na składową równoległą i prostopadłą jest zapisany wprost:
/// `(γ−1)/β²` zamiast dzielenia przez małe `|β|`, gdy boost zbliża się do zera.
pub fn boost(event: Event, beta: Vec3) -> Result<Event, Superluminal> {
    let beta2 = beta.norm_squared();
    let g = gamma_from_beta2(beta2)?;
    if beta2 == 0.0 {
        return Ok(event);
    }
    let r = event.position();
    let br = beta.dot(r);
    let factor = (g - 1.0) / beta2;
    let r_prime = r + beta * (factor * br - g * event.ct);
    Ok(Event {
        ct: g * (event.ct - br),
        x: r_prime.x,
        y: r_prime.y,
        z: r_prime.z,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    fn assert_event_close(got: Event, want: Event, tol: f64) {
        assert!(
            (got.ct - want.ct).abs() < tol
                && (got.x - want.x).abs() < tol
                && (got.y - want.y).abs() < tol
                && (got.z - want.z).abs() < tol,
            "got {got:?} want {want:?}"
        );
    }

    #[test]
    fn gamma_is_one_at_rest() {
        assert!((gamma(0.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((gamma(-0.0).unwrap() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn gamma_has_exact_textbook_values() {
        assert!((gamma(0.6).unwrap() - 1.25).abs() < 1e-15);
        assert!((gamma(-0.6).unwrap() - 1.25).abs() < 1e-15);
        assert!((gamma(0.8).unwrap() - 5.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn gamma_grows_with_speed() {
        let g1 = gamma(0.3).unwrap();
        let g2 = gamma(0.9).unwrap();
        assert!(g2 > g1 && g1 > 1.0);
    }

    #[test]
    fn gamma_rejects_lightspeed_and_beyond() {
        assert!(gamma(1.0).is_err());
        assert!(gamma(-1.0).is_err());
        assert!(gamma(1.2).is_err());
        assert!(gamma(f64::NAN).is_err());
        assert!(gamma_from_beta(vec3(0.8, 0.8, 0.0)).is_err());
    }

    #[test]
    fn boost_rejects_superluminal_velocity() {
        let e = Event::on_axis(0.0, 1.0);
        assert!(boost_x(e, 1.0).is_err());
        assert!(boost(e, vec3(0.0, 0.0, 1.0)).is_err());
    }

    #[test]
    fn rest_boost_is_identity() {
        let e = Event::new(2.0, -1.0, 0.5, 3.0);
        assert_eq!(boost_x(e, 0.0).unwrap(), e);
        assert_eq!(boost(e, vec3(0.0, 0.0, 0.0)).unwrap(), e);
        assert!((dilated_time(4.0, 0.0).unwrap() - 4.0).abs() < 1e-15);
        assert!((contracted_length(4.0, 0.0).unwrap() - 4.0).abs() < 1e-15);
    }

    #[test]
    fn boost_x_matches_textbook_formula() {
        let beta = 0.6;
        let g = 1.25;
        let e = Event::on_axis(2.0, 1.0);
        let got = boost_x(e, beta).unwrap();
        let want = Event::on_axis(g * (e.ct - beta * e.x), g * (e.x - beta * e.ct));
        assert_event_close(got, want, 1e-15);
    }

    #[test]
    fn boost_along_x_hat_matches_boost_x() {
        let e = Event::new(1.5, -0.4, 0.2, -0.7);
        let a = boost_x(e, 0.4).unwrap();
        let b = boost(e, vec3(0.4, 0.0, 0.0)).unwrap();
        assert_event_close(a, b, 1e-15);
        assert!((a.y - e.y).abs() < 1e-15 && (a.z - e.z).abs() < 1e-15);
    }

    #[test]
    fn simultaneous_events_at_different_x_split_in_time() {
        let left = Event::on_axis(0.0, -1.0);
        let right = Event::on_axis(0.0, 1.0);
        let beta = 0.6;
        let left_p = boost_x(left, beta).unwrap();
        let right_p = boost_x(right, beta).unwrap();
        assert!(
            (left_p.ct - right_p.ct).abs() > 1.0,
            "pioruny miały stracić wspólną chwilę, ct' {} i {}",
            left_p.ct,
            right_p.ct
        );
        let g = gamma(beta).unwrap();
        assert!((left_p.ct - g * beta).abs() < 1e-15);
        assert!((right_p.ct + g * beta).abs() < 1e-15);
    }

    #[test]
    fn interval_is_invariant_under_1d_and_3d_boost() {
        let a = Event::new(4.0, 1.0, -2.0, 0.5);
        let b = Event::new(1.0, -0.5, 0.25, 2.0);
        let s2 = a.interval_sq_to(b);
        let beta = vec3(0.2, -0.4, 0.1);
        let a_p = boost(a, beta).unwrap();
        let b_p = boost(b, beta).unwrap();
        assert!((a_p.interval_sq_to(b_p) - s2).abs() < 1e-12);
        let a_x = boost_x(a, 0.7).unwrap();
        let b_x = boost_x(b, 0.7).unwrap();
        assert!((a_x.interval_sq_to(b_x) - s2).abs() < 1e-12);
    }

    #[test]
    fn lightlike_stays_lightlike() {
        let on_light = Event::on_axis(5.0, 5.0);
        assert!(on_light.interval_sq().abs() < 1e-15);
        let moved = boost_x(on_light, 0.3).unwrap();
        assert!(moved.interval_sq().abs() < 1e-12);

        let diagonal = Event::new(5.0, 3.0, 4.0, 0.0);
        assert!(diagonal.interval_sq().abs() < 1e-15);
        let moved3 = boost(diagonal, vec3(0.1, 0.2, -0.15)).unwrap();
        assert!(moved3.interval_sq().abs() < 1e-12);
    }

    #[test]
    fn boost_there_and_back_restores_event() {
        let e = Event::new(2.0, 1.0, -0.5, 0.3);
        let beta = vec3(0.2, 0.4, -0.1);
        let there = boost(e, beta).unwrap();
        let back = boost(there, -beta).unwrap();
        assert_event_close(back, e, 1e-12);
        assert_event_close(boost_x(boost_x(e, 0.8).unwrap(), -0.8).unwrap(), e, 1e-12);
    }

    #[test]
    fn collinear_boost_composition_matches_velocity_addition() {
        let e = Event::new(3.0, -1.0, 0.4, 0.2);
        let b1 = 0.3;
        let b2 = 0.5;
        let composed = compose_boost_1d(b1, b2).unwrap();
        assert!(composed.abs() < 1.0);
        let stepwise = boost_x(boost_x(e, b1).unwrap(), b2).unwrap();
        let direct = boost_x(e, composed).unwrap();
        assert_event_close(stepwise, direct, 1e-12);
        assert!((compose_boost_1d(0.6, -0.6).unwrap()).abs() < 1e-15);
    }

    #[test]
    fn compose_boost_rejects_superluminal_parts() {
        assert!(compose_boost_1d(1.0, 0.1).is_err());
        assert!(compose_boost_1d(0.1, -1.5).is_err());
    }

    #[test]
    fn dilated_time_is_gamma_times_proper() {
        let beta = 0.8;
        let tau = 2.0;
        let dt = dilated_time(tau, beta).unwrap();
        let g = gamma(beta).unwrap();
        assert!((dt / tau - g).abs() < 1e-15);
        assert!(dt > tau);
    }

    #[test]
    fn contracted_length_times_gamma_is_proper() {
        let beta = 0.8;
        let l0 = 3.0;
        let l = contracted_length(l0, beta).unwrap();
        assert!((l * gamma(beta).unwrap() - l0).abs() < 1e-15);
        assert!(l < l0);
    }

    #[test]
    fn rod_contracts_only_along_motion() {
        let beta = vec3(0.8, 0.0, 0.0);
        let along = vec3(3.0, 0.0, 0.0);
        let across = vec3(0.0, 3.0, 0.0);
        let got_along = contract_rod(along, beta).unwrap();
        let got_across = contract_rod(across, beta).unwrap();
        assert!((got_along.x - contracted_length(3.0, 0.8).unwrap()).abs() < 1e-15);
        assert_eq!(got_across, across);
        let mixed = vec3(4.0, 2.0, -1.0);
        let got = contract_rod(mixed, beta).unwrap();
        assert!((got.x - mixed.x / gamma(0.8).unwrap()).abs() < 1e-15);
        assert!((got.y - mixed.y).abs() < 1e-15 && (got.z - mixed.z).abs() < 1e-15);
    }
}
