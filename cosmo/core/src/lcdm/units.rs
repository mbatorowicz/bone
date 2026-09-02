//! Układ jednostek: Mpc/h, 10¹⁰ M☉/h, km/s.
//!
//! Ten sam układ, którego używa GADGET, i to nie z sentymentu: w nim stała
//! grawitacji ma wartość rzędu jedności (43,0), więc siły i potencjały nie
//! rozjeżdżają się na skalę, przy której `f32` na siatce traciłby cyfry znaczące.

/// Kilometry w megaparseku — most między [Mpc] i [km/s].
pub const MPC_KM: f64 = 3.085_677_581_491_367e19;

/// Sekundy w gigaroku.
pub const GYR_S: f64 = 3.155_76e16;

/// `G` w [Mpc/h · (km/s)² / (10¹⁰ M☉/h)].
pub const G: f64 = 43.0071;

/// `H₀/h` w jednostkach kodu, czyli 100 km/s/(Mpc/h).
pub const H100: f64 = 100.0;

/// Ile gigalat trwa jednostka czasu kodu — wynik trzeba jeszcze podzielić przez `h`.
pub const GYR_PER_CODE_TIME: f64 = MPC_KM / GYR_S;

/// Gęstość krytyczna dziś, `3H₀²/(8πG)`, w [10¹⁰ M☉/h / (Mpc/h)³].
///
/// Nie zależy od `h` i nie jest to zbieg okoliczności: `H₀ = 100h`, a jednostki masy
/// i długości też noszą `h`, więc wszystkie potęgi się skracają. Dlatego jest to stała,
/// a nie funkcja `h` — parametr, którego wynik nie używa, zachęcałby do „poprawienia"
/// tej wielkości przez `h²`.
pub const CRITICAL_DENSITY_0: f64 = 3.0 * H100 * H100 / (8.0 * std::f64::consts::PI * G);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_density_matches_the_textbook_value() {
        // 2,7754·10¹⁰ M☉/h / (Mpc/h)³ = 2,7754·10¹¹ M☉/h / Mpc³.
        let expected = 27.754;
        assert!(
            (CRITICAL_DENSITY_0 - expected).abs() / expected < 1e-4,
            "ρ_kryt = {CRITICAL_DENSITY_0}"
        );
    }

    /// Jednostka czasu kodu to Mpc/(km/s) ≈ 978 Gyr, więc `1/H₀ = 9,78/h` Gyr.
    /// Test pilnuje, że przeliczenie nie zostało odwrócone — odwrotność dałaby
    /// wiek wszechświata rzędu 10⁻⁵ Gyr i nikt by tego nie nazwał wynikiem.
    #[test]
    fn code_time_unit_is_about_a_thousand_gigayears() {
        assert!(
            (GYR_PER_CODE_TIME - 977.79).abs() < 0.1,
            "jednostka czasu = {GYR_PER_CODE_TIME}"
        );
        let hubble_time_gyr = GYR_PER_CODE_TIME / H100;
        assert!((hubble_time_gyr - 9.778).abs() < 0.01, "1/H₀ = {hubble_time_gyr}");
    }
}
