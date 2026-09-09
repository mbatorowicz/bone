//! Jednostki atomowe i stałe, z których liczone są energie, długości i czasy.
//!
//! Silnik liczy w jednostkach Hartree’a: `ħ = m_e = e = 4πε₀ = 1`. Długość to
//! promień Bohra, energia — hartree, czas — `ħ/E_h`. Na zewnątrz (panel, CLI)
//! liczby są podawane w eV, nm i fs, bo tak się je porównuje z tablicami.

/// Energia Hartree w elektronowoltach (CODATA 2018).
pub const HARTREE_EV: f64 = 27.211_386_245_988;

/// Promień Bohra w metrach.
pub const BOHR_M: f64 = 5.291_772_109_03e-11;

/// Promień Bohra w femtometrach — ta sama miara co w modelu cząstek.
pub const BOHR_FM: f64 = BOHR_M * 1.0e15;

/// Czas atomowy `ħ/E_h` w sekundach.
pub const ATOMIC_TIME_S: f64 = 2.418_884_326_585_7e-17;

/// Czas atomowy w femtosekundach.
pub const ATOMIC_TIME_FS: f64 = ATOMIC_TIME_S * 1.0e15;

/// Stała struktury subtelnej.
pub const ALPHA: f64 = 7.297_352_569_3e-3;

/// Stosunek masy protonu do masy elektronu.
pub const PROTON_ELECTRON_MASS: f64 = 1_836.152_673_43;

/// `hc` w eV·nm — do przeliczania energii przejścia na długość fali.
pub const HC_EV_NM: f64 = 1_239.841_93;

/// Rydberg nieskończonej masy jądra, w eV. Doświadczalny wodór jest mniejszy
/// o czynnik zredukowanej masy; różnica jest mierzona, nie ukrywana.
pub const RYDBERG_INF_EV: f64 = 0.5 * HARTREE_EV;

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
}
