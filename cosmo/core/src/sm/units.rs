//! Jednostki i stałe sprzężenia dla modelu cząstek.
//!
//! Cały moduł `sm` liczy w układzie, w którym **`c = 1`**:
//!
//! ```text
//! energia, masa, pęd   MeV        (pęd naprawdę w MeV/c, masa w MeV/c²)
//! długość              fm
//! czas                 fm/c
//! ładunek              e          (przechowywany jako liczba całkowita trzecich)
//! ```
//!
//! Wybór `c = 1` nie jest kosmetyką ani lenistwem. W modelach `sr` i `lcdm` `c` jest
//! parametrem, bo tam wolno je zmieniać — to nastawy badanego układu. Tutaj cząstki
//! mają masy zmierzone w laboratorium, więc `c` jest przelicznikiem między MeV
//! a MeV/c², a nie pokrętłem. Trzymanie go jako zmiennej dawałoby okazję do biegu
//! z masą elektronu i cudzą prędkością światła, czyli do liczb, których nie da się
//! porównać z niczym.
//!
//! # Skąd biorą się przeliczniki
//!
//! Jedyną stałą wymiarową, jakiej potrzeba, jest `ħc = 197,3269804 MeV·fm`. Ona
//! zamienia „odwrotność energii" na długość i odwrotnie, więc każde sprzężenie
//! poniżej sprowadza się do liczby bezwymiarowej pomnożonej przez `ħc`.

/// `ħc` — CODATA 2018, wartość dokładna z definicji `ħ` i `c`.
pub const HBAR_C: f64 = 197.326_980_4;

/// Ile femtometrów przebywa światło w sekundzie. Zamienia czasy życia z PDG
/// (sekundy) na czas symulacji (fm/c).
pub const FM_PER_SECOND: f64 = 2.997_924_58e23;

/// Stała struktury subtelnej `α = e²/(4πε₀ħc)` — CODATA 2018.
pub const ALPHA_EM: f64 = 7.297_352_569_3e-3;

/// Sprzężenie Coulomba w tych jednostkach: `V = COULOMB · z₁z₂ / r`,
/// gdzie `z` jest ładunkiem w jednostkach `e`, `r` w fm, a `V` wychodzi w MeV.
///
/// Liczbowo `≈ 1,44 MeV·fm` — czyli dwa protony w odległości femtometra dzieli
/// bariera rzędu megaelektronowolta. To jest ta sama „1,44", która w chemii pojawia
/// się jako `14,4 eV·Å`; zgodność obu zapisów jest testowana.
pub const COULOMB: f64 = ALPHA_EM * HBAR_C;

/// Masa Plancka w MeV — `√(ħc/G)` przeliczone z CODATA 2018.
pub const PLANCK_MASS: f64 = 1.220_890e22;

/// Sprzężenie grawitacyjne: `V = −GRAVITY · m₁m₂ / r` (masy w MeV, `r` w fm).
///
/// Wynika z `G = ħc/M_Pl²`. Rząd wielkości `10⁻⁴²` nie jest pomyłką — to jest
/// właśnie ta liczba, przez którą grawitacja w fizyce cząstek nie występuje.
/// Moduł [`crate::sm::forces`] liczy ją mimo to i **mierzy** jej udział, zamiast
/// zakładać, że jest zaniedbywalna.
pub const GRAVITY: f64 = HBAR_C / (PLANCK_MASS * PLANCK_MASS);

/// Sprzężenie silne w skali ~1 GeV. Nie jest stałą — biegnie ze skalą energii —
/// więc każda pojedyncza liczba jest przybliżeniem obowiązującym w jednym zakresie.
pub const ALPHA_S: f64 = 0.30;

/// Napięcie struny `σ` w MeV/fm: człon liniowy potencjału Cornella.
///
/// Z sieciowego `σ ≈ 0,18 GeV²` przez podzielenie przez `ħc`. To jest energia
/// potrzebna na rozciągnięcie pary kwarków o femtometr — prawie GeV, czyli tyle,
/// ile waży proton. Stąd bierze się uwięzienie: rozdzielenie kwarków jest droższe
/// niż wyprodukowanie nowej pary.
pub const STRING_TENSION: f64 = 0.18e6 / HBAR_C;

/// Masa bozonu W w MeV — wyznacza zasięg oddziaływania słabego.
pub const W_MASS: f64 = 80_377.0;

/// Sprzężenie słabe `α_w = g²/4π = α/sin²θ_W` przy `sin²θ_W = 0,2312`.
pub const ALPHA_WEAK: f64 = ALPHA_EM / 0.231_21;

/// Zasięg oddziaływania słabego `λ_W = ħc/M_W` w fm.
///
/// Liczbowo `≈ 0,0025 fm` — dwa i pół tysiąca razy mniej niż promień protonu.
/// Dlatego oddziaływanie słabe praktycznie nie jest siłą: na każdej odległości,
/// którą ta symulacja rozdziela, jego wkład do pędu jest wygaszony wykładniczo.
/// Jego widocznym skutkiem są **rozpady**, i tam właśnie jest policzone —
/// w [`crate::sm::decays`].
pub const WEAK_RANGE: f64 = HBAR_C / W_MASS;

/// Klasyczny promień elektronu `r_e = α ħc / m_e c²` w fm. Skala przekroju
/// czynnego na anihilację `e⁺e⁻ → γγ`.
pub const CLASSICAL_ELECTRON_RADIUS: f64 = 2.817_940_326_2;

/// Zamień czas życia z tablic PDG (sekundy) na czas symulacji (fm/c).
pub fn lifetime_to_fm(seconds: f64) -> f64 {
    seconds * FM_PER_SECOND
}

/// Zamień czas symulacji (fm/c) na sekundy — do komunikatów dla człowieka.
pub fn fm_to_seconds(fm: f64) -> f64 {
    fm / FM_PER_SECOND
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sprzężenie Coulomba musi zgadzać się z zapisem znanym z chemii: `14,4 eV·Å`.
    /// To dwa zupełnie różne układy jednostek, więc zgodność łapie pomyłkę o rząd
    /// wielkości w `ħc` albo w `α`.
    #[test]
    fn coulomb_matches_the_chemists_number() {
        // 1 MeV·fm = 10⁶ eV · 10⁻⁵ Å = 10 eV·Å
        let ev_angstrom = COULOMB * 10.0;
        assert!(
            (ev_angstrom - 14.4).abs() < 0.05,
            "{ev_angstrom} eV·Å zamiast 14,4"
        );
    }

    /// Iloraz grawitacji do Coulomba dla dwóch protonów jest jedną z najczęściej
    /// cytowanych liczb w fizyce: `≈ 8·10⁻³⁷`. Jest niezależnym sprawdzianem tego,
    /// że `GRAVITY` i `COULOMB` mieszkają w tym samym układzie jednostek — a to
    /// jedyny warunek, żeby wolno je było dodać do siebie w jednej pętli sił.
    #[test]
    fn gravity_to_coulomb_ratio_for_two_protons() {
        let proton = 938.272_088_16;
        let ratio = GRAVITY * proton * proton / COULOMB;
        assert!(
            (8.0e-37..9.0e-37).contains(&ratio),
            "grawitacja/Coulomb = {ratio:e}, a ma być ~8·10⁻³⁷"
        );
    }

    /// Napięcie struny wyrażone „po fizycznemu": prawie 1 GeV na femtometr.
    #[test]
    fn string_tension_is_about_one_gev_per_fermi() {
        assert!(
            (850.0..1000.0).contains(&STRING_TENSION),
            "σ = {STRING_TENSION} MeV/fm"
        );
    }

    /// Zasięg słabego musi być o trzy rzędy mniejszy od protonu — to jest powód,
    /// dla którego rozdziela się je na siłę (nieistotną) i rozpady (istotne).
    #[test]
    fn weak_range_is_far_below_the_proton() {
        const { assert!(WEAK_RANGE < 0.005) };
        const { assert!(WEAK_RANGE > 0.001) };
    }

    #[test]
    fn lifetime_conversion_round_trips() {
        let muon_seconds = 2.196_981_1e-6;
        let fm = lifetime_to_fm(muon_seconds);
        assert!((fm_to_seconds(fm) - muon_seconds).abs() / muon_seconds < 1e-15);
        // Mion przelatuje ~659 metrów na czas życia — stąd 6,6·10¹⁷ fm.
        assert!((6.0e17..7.0e17).contains(&fm), "τ_μ = {fm} fm/c");
    }

    /// Promień elektronu musi wynikać ze sprzężenia, a nie być niezależnie wpisaną
    /// liczbą, która może się rozjechać z resztą tablicy.
    #[test]
    fn classical_radius_follows_from_the_coupling() {
        let electron_mass = 0.510_998_950_0;
        let derived = COULOMB / electron_mass;
        assert!(
            (derived - CLASSICAL_ELECTRON_RADIUS).abs() < 1e-6,
            "r_e wyprowadzone {derived} vs wpisane {CLASSICAL_ELECTRON_RADIUS}"
        );
    }
}
