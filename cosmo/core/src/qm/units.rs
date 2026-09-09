//! Jednostki atomowe — przeliczniki na eV, nm i fs.
//!
//! Silnik liczy w jednostkach Hartree’a: `ħ = m_e = e = 4πε₀ = 1`. Długość to
//! promień Bohra, energia — hartree, czas — `ħ/E_h`. Stałe żyją w
//! [`crate::constants`] (CODATA 2022); ten moduł tylko je pokazuje na zewnątrz.

pub use crate::constants::{
    ALPHA, ATOMIC_TIME_FS, ATOMIC_TIME_S, BOHR_FM, BOHR_M, HARTREE_EV, HC_EV_NM,
    PROTON_ELECTRON_MASS, RYDBERG_INF_EV,
};

pub fn hartree_to_ev(e: f64) -> f64 {
    e * HARTREE_EV
}

pub fn ev_to_hartree(e: f64) -> f64 {
    e / HARTREE_EV
}

pub fn atomic_to_fs(t: f64) -> f64 {
    t * ATOMIC_TIME_FS
}

pub fn bohr_to_fm(r: f64) -> f64 {
    r * BOHR_FM
}

/// Długość fali w nm z różnicy energii w eV. Zero i ujemne `delta` oddają `None`,
/// bo to nie jest przejście, tylko ta sama lub wyższa studnia.
pub fn wavelength_nm(delta_ev: f64) -> Option<f64> {
    if delta_ev > 0.0 && delta_ev.is_finite() {
        Some(HC_EV_NM / delta_ev)
    } else {
        None
    }
}

/// Zredukowana masa elektronu przy jądrze o masie `nucleus_over_electron` w
/// jednostkach `m_e`. Dla nieskończonego jądra wraca 1.
pub fn reduced_mass_ratio(nucleus_over_electron: f64) -> f64 {
    if !nucleus_over_electron.is_finite() || nucleus_over_electron <= 0.0 {
        return 1.0;
    }
    nucleus_over_electron / (nucleus_over_electron + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_ionization_with_reduced_mass_matches_experiment() {
        let mu = reduced_mass_ratio(PROTON_ELECTRON_MASS);
        let ie = RYDBERG_INF_EV * mu;
        // NIST: 13,59844 eV. Nieskończone jądro daje 13,606, czyli o 0,05% za dużo.
        assert!((ie - 13.598_44).abs() < 0.001, "{ie}");
        let infinite = RYDBERG_INF_EV;
        assert!(infinite > ie);
        assert!((infinite - ie) / infinite > 4e-4);
    }

    #[test]
    fn balmer_h_alpha_is_red() {
        let mu = reduced_mass_ratio(PROTON_ELECTRON_MASS);
        let e3 = -RYDBERG_INF_EV * mu / 9.0;
        let e2 = -RYDBERG_INF_EV * mu / 4.0;
        let lambda = wavelength_nm(e3 - e2).unwrap();
        assert!(
            (lambda - 656.3).abs() < 0.5,
            "Hα = {lambda} nm, oczekiwane 656,3"
        );
    }

    #[test]
    fn conversions_are_finite_and_oriented() {
        assert!(hartree_to_ev(1.0) > 27.0);
        assert!(atomic_to_fs(1.0) > 0.0);
    }

    /// α w atomach i α w cząstkach to ta sama liczba. Dwa wpisy rozjechałyby
    /// się przy aktualizacji CODATA i nikt by tego nie zauważył na orbitalu.
    #[test]
    fn fine_structure_constant_is_shared_with_particles() {
        assert!((ALPHA - crate::sm::units::ALPHA_EM).abs() < 1e-18);
    }
}
