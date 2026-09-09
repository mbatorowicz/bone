//! Wariacja helu: najprostsza funkcja próbna, nie Hartree–Fock.
//!
//! Dwa elektrony w iloczynie orbitali `1s` z jednym parametrem `ζ`:
//!
//! ```text
//! ψ(r₁, r₂) = e^{−ζ r₁} e^{−ζ r₂}
//! ```
//!
//! Hamiltonian (jądro nieskończone, jednostki Hartree’a):
//!
//! ```text
//! H = −½∇₁² − ½∇₂² − Z/r₁ − Z/r₂ + 1/r₁₂ ,   Z = 2
//! ```
//!
//! Średnia na znormalizowanym `1s` daje `E(ζ) = ζ² − 2Z ζ + (5/8) ζ`.
//! Minimum przy `Z = 2` wypada w `ζ = 27/16`, `E = −729/256` hartree.
//! Dokładny nietraktacyjny stan podstawowy (Pekeris) to około `−2.9037` Ha,
//! więc błąd jest rzędu dwóch procent — to jest wynik modelu, nie usterka.

/// Wykładnik minimalizujący `E(ζ)` dla `Z = 2`.
pub const ZETA: f64 = 27.0 / 16.0;

/// `E(27/16) = −729/256` hartree.
pub const ENERGY_HARTREE: f64 = -729.0 / 256.0;

/// Nietraktacyjny stan podstawowy He, jądro nieskończone (Pekeris).
pub const EXACT_ENERGY_HARTREE: f64 = -2.903_724;

pub fn energy_of_zeta(zeta: f64, z: f64) -> f64 {
    zeta * zeta - 2.0 * z * zeta + 0.625 * zeta
}

pub fn energy_error() -> f64 {
    (ENERGY_HARTREE - EXACT_ENERGY_HARTREE) / EXACT_ENERGY_HARTREE.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variational_helium_energy_is_minus_729_over_256() {
        let e = energy_of_zeta(ZETA, 2.0);
        assert!((e - ENERGY_HARTREE).abs() < 1e-12, "{e}");
        assert!((ENERGY_HARTREE - (-2.847_656_25)).abs() < 1e-12);
        let err = energy_error();
        assert!(
            (0.015..0.025).contains(&err),
            "błąd wariacji powinien być ~2%, jest {err}"
        );
    }

    #[test]
    fn zeta_minimizes_the_trial_energy() {
        let e0 = energy_of_zeta(ZETA, 2.0);
        for d in [-0.05, -0.01, 0.01, 0.05] {
            let e = energy_of_zeta(ZETA + d, 2.0);
            assert!(e > e0, "ζ+{d}: {e} nie jest powyżej minimum {e0}");
        }
    }
}
