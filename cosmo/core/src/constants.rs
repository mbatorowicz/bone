//! Jedno źródło stałych fizycznych.
//!
//! Modele `sm` i `qm` liczą w różnych układach jednostek, ale α, ħc i masy
//! nukleonów to te same liczby. Wpisywanie ich osobno w każdym module dawałoby
//! okazję, żeby po poprawce z CODATA rozjechały się o ostatnią cyfrę — i nic by
//! tego nie złapało, bo oba zestawy przechodziłyby swoje lokalne testy.
//!
//! # Skąd co pochodzi
//!
//! - **SI 2019** — `c`, `h`, `e`, `ħ` są dokładne z definicji. `ħc` w MeV·fm
//!   i `hc` w eV·nm wynikają z nich bez pomiaru.
//! - **CODATA 2022** — stałe, które nadal się mierzy: α, hartree, Bohr,
//!   stosunek mas, masy e/p/n.
//! - **PDG 2025** — masy i szerokości rezonansów (W, Z, H, t). Czas życia
//!   rezonansu **nie jest tu wpisany**: `τ = ħ/Γ`.
//!
//! Wartości pochodne (sprzężenie Coulomba, Rydberg, klasyczny promień elektronu,
//! zasięg słabego) są wyprowadzane. Osobny wpis rozjechałby się z resztą
//! tablicy przy pierwszej poprawce.

/// Prędkość światła, dokładna z definicji metra.
pub const C_M_S: f64 = 299_792_458.0;

/// Ile femtometrów przebywa światło w sekundzie — `c` w fm/s.
pub const FM_PER_SECOND: f64 = 2.997_924_58e23;

/// `ħc` w MeV·fm. Dokładne z definicji `ħ` i `c` (SI 2019).
pub const HBAR_C_MEV_FM: f64 = 197.326_980_4;

/// `ħ` w MeV·s. Dokładne z definicji `ħ` i `e` (SI 2019).
///
/// Zamienia szerokość rezonansu `Γ` na średni czas życia: `τ = ħ/Γ`.
pub const HBAR_MEV_S: f64 = 6.582_119_569e-22;

/// `hc` w eV·nm. Dokładne z definicji `h`, `c` i `e`.
pub const HC_EV_NM: f64 = 1_239.841_984_332;

/// Stała struktury subtelnej `α` — CODATA 2022.
pub const ALPHA: f64 = 7.297_352_564_3e-3;

/// Energia Hartree w eV — CODATA 2022.
pub const HARTREE_EV: f64 = 27.211_386_245_981;

/// Promień Bohra w metrach — CODATA 2022.
pub const BOHR_M: f64 = 5.291_772_105_44e-11;

/// Promień Bohra w femtometrach — ta sama miara co w modelu cząstek.
pub const BOHR_FM: f64 = BOHR_M * 1.0e15;

/// Czas atomowy `ħ/E_h` w sekundach — CODATA 2022.
pub const ATOMIC_TIME_S: f64 = 2.418_884_326_586_4e-17;

/// Czas atomowy w femtosekundach.
pub const ATOMIC_TIME_FS: f64 = ATOMIC_TIME_S * 1.0e15;

/// Stosunek masy protonu do masy elektronu — CODATA 2022.
pub const PROTON_ELECTRON_MASS: f64 = 1_836.152_673_426;

/// Masa elektronu w MeV — CODATA 2022.
pub const ELECTRON_MASS_MEV: f64 = 0.510_998_950_69;

/// Masa protonu w MeV — CODATA 2022.
pub const PROTON_MASS_MEV: f64 = 938.272_089_43;

/// Masa neutronu w MeV — CODATA 2022.
pub const NEUTRON_MASS_MEV: f64 = 939.565_421_94;

/// Masa Plancka w MeV — `√(ħc/G)`, CODATA 2022.
pub const PLANCK_MASS_MEV: f64 = 1.220_890e22;

/// Rydberg nieskończonej masy jądra, w eV. `E_h/2`.
pub const RYDBERG_INF_EV: f64 = 0.5 * HARTREE_EV;

/// Sprzężenie Coulomba w MeV·fm: `V = COULOMB · z₁z₂ / r`.
pub const COULOMB: f64 = ALPHA * HBAR_C_MEV_FM;

/// Klasyczny promień elektronu `r_e = α ħc / m_e c²` w fm.
///
/// Wyprowadzony, nie wpisany: inna α albo inna masa elektronu ma zmienić
/// przekrój anihilacji, a nie zostawić tu starą liczbę.
pub const CLASSICAL_ELECTRON_RADIUS_FM: f64 = COULOMB / ELECTRON_MASS_MEV;

/// Sprzężenie grawitacyjne w jednostkach cząstek: `G = ħc / M_Pl²`.
pub const GRAVITY_MEV_FM: f64 = HBAR_C_MEV_FM / (PLANCK_MASS_MEV * PLANCK_MASS_MEV);

/// Masa bozonu W w MeV — PDG 2025.
pub const W_MASS_MEV: f64 = 80_369.2;

/// Masa bozonu Z w MeV — PDG 2025.
pub const Z_MASS_MEV: f64 = 91_188.0;

/// Masa bozonu Higgsa w MeV — PDG 2025.
pub const HIGGS_MASS_MEV: f64 = 125_200.0;

/// Masa kwarka szczytowego (pomiar bezpośredni) w MeV — PDG 2025.
pub const TOP_MASS_MEV: f64 = 172_560.0;

/// Szerokość W w MeV — PDG 2025, `Γ = 2,14 GeV`.
pub const W_WIDTH_MEV: f64 = 2_140.0;

/// Szerokość Z w MeV — PDG 2025.
pub const Z_WIDTH_MEV: f64 = 2_495.5;

/// Szerokość Higgsa w MeV — przewidywanie SM (LHCHXSWG) przy `m_H = 125,20 GeV`.
///
/// Doświadczalna szerokość jest rzędu kilku MeV z niepewnością większą niż
/// sama wartość; wpisanie pomiaru udawałoby dokładność, której nie ma.
pub const HIGGS_WIDTH_MEV: f64 = 4.07;

/// Szerokość kwarka szczytowego w MeV — PDG 2025, `Γ = 1,42 GeV`.
pub const TOP_WIDTH_MEV: f64 = 1_420.0;

/// `sin²θ_W` w `MSbar` przy `M_Z` — PDG 2025.
pub const SIN2_THETA_W: f64 = 0.231_22;

/// Temperatura CMB w kelwinach — Fixsen 2009, wartość używana przez Plancka 2018.
pub const T_CMB: f64 = 2.7255;

/// Średni czas życia z szerokości rezonansu: `τ = ħ/Γ`, wynik w sekundach.
pub const fn mean_life_s(width_mev: f64) -> f64 {
    HBAR_MEV_S / width_mev
}

pub const W_MEAN_LIFE_S: f64 = mean_life_s(W_WIDTH_MEV);
pub const Z_MEAN_LIFE_S: f64 = mean_life_s(Z_WIDTH_MEV);
pub const HIGGS_MEAN_LIFE_S: f64 = mean_life_s(HIGGS_WIDTH_MEV);
pub const TOP_MEAN_LIFE_S: f64 = mean_life_s(TOP_WIDTH_MEV);

#[cfg(test)]
mod tests {
    use super::*;

    /// Klasyczny promień musi wyjść z α i masy, i zgadzać się z CODATA 2022.
    #[test]
    fn classical_radius_matches_codata_2022() {
        let published = 2.817_940_320_5;
        assert!(
            (CLASSICAL_ELECTRON_RADIUS_FM - published).abs() < 1e-9,
            "r_e = {CLASSICAL_ELECTRON_RADIUS_FM} fm vs CODATA {published}"
        );
    }

    /// Rydberg to połowa hartree — tożsamość, nie osobny pomiar.
    #[test]
    fn rydberg_is_half_a_hartree() {
        const { assert!(RYDBERG_INF_EV == 0.5 * HARTREE_EV) };
        assert!((RYDBERG_INF_EV - 13.605_693_122_990).abs() < 1e-12);
    }

    /// Czas życia W musi wyjść z szerokości, nie z osobnej liczby.
    #[test]
    fn resonance_lifetime_is_hbar_over_width() {
        const { assert!(W_MEAN_LIFE_S == HBAR_MEV_S / W_WIDTH_MEV) };
        assert!((W_MEAN_LIFE_S - 3.076e-25).abs() / 3.076e-25 < 0.01);
        assert!((Z_MEAN_LIFE_S - 2.638e-25).abs() / 2.638e-25 < 0.01);
        assert!((HIGGS_MEAN_LIFE_S - 1.62e-22).abs() / 1.62e-22 < 0.02);
    }

    #[test]
    fn electroweak_masses_are_ordered() {
        const { assert!(W_MASS_MEV < Z_MASS_MEV) };
        const { assert!(Z_MASS_MEV < HIGGS_MASS_MEV) };
        const { assert!(HIGGS_MASS_MEV < TOP_MASS_MEV) };
    }
}
