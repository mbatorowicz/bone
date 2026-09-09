//! Kinematyka relatywistyczna przy `c = 1`, z cząstkami bezmasowymi włącznie.
//!
//! Model `sr` trzyma `c` jako parametr i podpiera masę podłogą [`MIN_MASS`], bo tam
//! cząstka bezmasowa nie występuje i nie ma jej po co obsługiwać. Tutaj występuje:
//! foton, gluon i (w przyjętym przybliżeniu) neutrino mają masę dokładnie zero,
//! a rozpad `π⁰ → γγ` produkuje je w co drugim kroku.
//!
//! Zero w mianowniku nie pojawia się nigdzie, bo wszystko jest wyrażone przez
//! **energię**, a nie przez `γ`:
//!
//! ```text
//! E = √(|p|² + m²)          skończone dla m = 0
//! v = p/E                   |v| = 1 dokładnie, gdy m = 0
//! T = E − m                 dla m = 0 to po prostu |p|
//! ```
//!
//! `γ = E/m` istnieje tylko dla cząstek masywnych i dlatego zwracane jest jako
//! [`Option`], a nie jako nieskończoność. Nieskończoność przeszłaby przez arytmetykę
//! po cichu i wypłynęła jako `NaN` trzy działania dalej, w miejscu, które nie ma
//! z tym nic wspólnego.
//!
//! [`MIN_MASS`]: crate::sr::relativity::MIN_MASS

use crate::vec3::{Vec3, ZERO};

/// `E = √(|p|² + m²)`, liczone przez `hypot`.
///
/// `hypot` nie przepełnia się przy dużych argumentach, więc pęd rzędu `10³⁰⁰ MeV`
/// nadal daje poprawną energię zamiast `inf`.
pub fn energy(mass: f64, momentum: Vec3) -> f64 {
    momentum.norm().hypot(mass)
}

/// Energia kinetyczna `T = E − m`.
///
/// Dla cząstki bezmasowej cała energia jest kinetyczna, więc `T = |p|`. Odjęcie
/// energii spoczynkowej nie jest kosmetyką: `E` zmienia się przy rozpadach (masa
/// spoczynkowa przechodzi w energię potomków), więc bilans mieszałby dwie rzeczy.
pub fn kinetic_energy(mass: f64, momentum: Vec3) -> f64 {
    energy(mass, momentum) - mass
}

/// `v = p/E` w jednostkach `c`. Zawsze `|v| ≤ 1`, z równością tylko dla `m = 0`.
pub fn velocity(mass: f64, momentum: Vec3) -> Vec3 {
    let e = energy(mass, momentum);
    if e == 0.0 {
        return ZERO;
    }
    momentum / e
}

/// `β = |v|/c`. Zawsze `≤ 1`.
///
/// Powyżej `γ ≈ 10⁸` różnica `1 − β` jest mniejsza niż precyzja `f64` i wynik
/// zaokrągla się do dokładnie `1`. Cząstka nadal ma skończoną masę i skończoną
/// energię — to arytmetyka przestaje odróżniać jej prędkość od `c`, a nie fizyka.
/// Nie przycinamy tego do `1 − ε`: udawana różnica byłaby zmyślona, a jedyne, co
/// od niej zależy, to wygląd wydruku.
pub fn beta(mass: f64, momentum: Vec3) -> f64 {
    let e = energy(mass, momentum);
    if e == 0.0 {
        return 0.0;
    }
    momentum.norm() / e
}

/// `γ = E/m`. `None` dla cząstki bezmasowej, która nie ma układu spoczynkowego.
pub fn gamma(mass: f64, momentum: Vec3) -> Option<f64> {
    if mass <= 0.0 {
        return None;
    }
    Some(energy(mass, momentum) / mass)
}

/// `p = γmβ` — pęd cząstki masywnej o zadanej prędkości (w jednostkach `c`).
///
/// # Errors
/// Gdy `|β| ≥ 1`. Cicho przycinać nie wolno: warunek początkowy różniłby się wtedy
/// od zamówionego i nikt by się o tym nie dowiedział.
pub fn momentum_from_beta(mass: f64, beta: Vec3) -> Result<Vec3, Superluminal> {
    let b2 = beta.norm_squared();
    if b2.is_nan() || b2 >= 1.0 {
        return Err(Superluminal {
            beta: if b2.is_nan() { f64::NAN } else { b2.sqrt() },
        });
    }
    Ok(beta * (mass / (1.0 - b2).sqrt()))
}

/// Pęd cząstki o zadanej energii kinetycznej, w zadanym kierunku.
///
/// Działa też dla `m = 0`, gdzie `|p| = T`. To jest naturalny sposób zadawania
/// warunku początkowego w fizyce cząstek: energię wiązki zna się, a pęd wylicza.
pub fn momentum_from_kinetic(mass: f64, kinetic: f64, direction: Vec3) -> Vec3 {
    let e = mass + kinetic.max(0.0);
    let p = (e * e - mass * mass).max(0.0).sqrt();
    match direction.norm() {
        0.0 => ZERO,
        n => direction * (p / n),
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Superluminal {
    pub beta: f64,
}

impl std::fmt::Display for Superluminal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prędkość β={:.4} ≥ 1 — pęd byłby nieskończony", self.beta)
    }
}

impl std::error::Error for Superluminal {}

/// Czteropęd `(E, p)`. Istnieje po to, żeby rozpad miał czym rachować bilans.
///
/// Rozpad odbywa się w układzie spoczynkowym cząstki macierzystej, a wynik trzeba
/// przenieść do układu laboratorium. Bez czteropędu trzeba by ręcznie pilnować, żeby
/// energia i pęd zostały pchnięte tym samym `β` — a to jest dokładnie ten rodzaj
/// pomyłki, który daje wynik „prawie zachowany".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FourMomentum {
    pub energy: f64,
    pub momentum: Vec3,
}

impl FourMomentum {
    pub fn new(energy: f64, momentum: Vec3) -> Self {
        Self { energy, momentum }
    }

    /// Czteropęd cząstki o danej masie i pędzie — energia wynika z powłoki masy.
    pub fn on_shell(mass: f64, momentum: Vec3) -> Self {
        Self {
            energy: energy(mass, momentum),
            momentum,
        }
    }

    /// Cząstka spoczywająca.
    pub fn at_rest(mass: f64) -> Self {
        Self {
            energy: mass,
            momentum: ZERO,
        }
    }

    /// Masa niezmiennicza `√(E² − |p|²)`.
    ///
    /// Ujemny wynik pod pierwiastkiem (możliwy przy sumowaniu i błędach
    /// zaokrągleń dla stanu bezmasowego) jest ścinany do zera, bo `m² < 0` nie
    /// opisuje niczego, co ta symulacja może wyprodukować.
    pub fn invariant_mass(self) -> f64 {
        (self.energy * self.energy - self.momentum.norm_squared())
            .max(0.0)
            .sqrt()
    }

    /// Prędkość układu, w którym ten czteropęd spoczywa.
    pub fn boost_velocity(self) -> Vec3 {
        if self.energy == 0.0 {
            return ZERO;
        }
        self.momentum / self.energy
    }

    /// Pchnięcie Lorentza o prędkość `beta` (w jednostkach `c`).
    ///
    /// Konwencja: `beta` jest prędkością układu spoczynkowego względem laboratorium,
    /// więc pchnięcie czteropędu z układu spoczynkowego tą prędkością daje czteropęd
    /// w laboratorium. Rozkład na składową równoległą i prostopadłą jest zapisany
    /// wprost, bo wersja „macierz 4×4" byłaby dłuższa i trudniejsza do sprawdzenia.
    pub fn boost(self, beta: Vec3) -> Self {
        let b2 = beta.norm_squared();
        if b2 <= 0.0 {
            return self;
        }
        let gamma = 1.0 / (1.0 - b2).sqrt();
        let bp = beta.dot(self.momentum);
        // (γ−1)/β² zamiast (γ−1)/|β|·(1/|β|) — jeden pierwiastek mniej i brak
        // dzielenia przez małą liczbę, gdy β dąży do zera.
        let factor = (gamma - 1.0) / b2;
        Self {
            energy: gamma * (self.energy + bp),
            momentum: self.momentum + beta * (factor * bp + gamma * self.energy),
        }
    }
}

impl std::ops::Add for FourMomentum {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            energy: self.energy + other.energy,
            momentum: self.momentum + other.momentum,
        }
    }
}

impl std::iter::Sum for FourMomentum {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::new(0.0, ZERO), |a, b| a + b)
    }
}

/// Pęd każdego z produktów rozpadu dwuciałowego, w układzie spoczynkowym matki.
///
/// ```text
/// p* = √(λ(M², m₁², m₂²)) / (2M),   λ(a,b,c) = a² + b² + c² − 2ab − 2bc − 2ca
/// ```
///
/// `None`, gdy `M < m₁ + m₂` — rozpad jest wtedy zabroniony energetycznie i nie ma
/// „prawie dozwolonego" wariantu, który dałoby się policzyć mimo wszystko.
///
/// Postać `λ` liczona jest jako iloczyn `(M²−(m₁+m₂)²)(M²−(m₁−m₂)²)`, a nie z sumy
/// sześciu składników. Dla `M` bliskiego progowi suma jest różnicą dużych liczb
/// i traci wszystkie cyfry znaczące, a iloczyn ma pierwszy czynnik mały z natury —
/// czyli dokładnie tam, gdzie wynik jest najczulszy, jest też najdokładniejszy.
pub fn two_body_momentum(parent: f64, m1: f64, m2: f64) -> Option<f64> {
    if parent < m1 + m2 {
        return None;
    }
    let s = parent * parent;
    let sum = m1 + m2;
    let diff = m1 - m2;
    let lambda = (s - sum * sum) * (s - diff * diff);
    Some(lambda.max(0.0).sqrt() / (2.0 * parent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    const ELECTRON: f64 = 0.510_998_95;
    const MUON: f64 = 105.658_375_5;

    #[test]
    fn massless_particle_travels_at_exactly_one() {
        let p = vec3(3.0, -4.0, 12.0);
        assert_eq!(energy(0.0, p), p.norm());
        assert!((beta(0.0, p) - 1.0).abs() < 1e-15);
        assert!((velocity(0.0, p).norm() - 1.0).abs() < 1e-15);
        assert!(gamma(0.0, p).is_none());
    }

    /// Cząstka masywna nie może przekroczyć `c` przy ŻADNYM pędzie — także takim,
    /// przy którym `γ` wykracza poza zakres podwójnej precyzji.
    #[test]
    fn massive_particle_never_exceeds_one() {
        for magnitude in [1e-6, 1.0, 1e3, 1e12, 1e200] {
            let p = vec3(magnitude, magnitude, magnitude);
            let b = beta(ELECTRON, p);
            assert!(b <= 1.0 && b.is_finite(), "|p|={magnitude} β={b}");
            assert!(velocity(ELECTRON, p).norm() <= 1.0, "|p|={magnitude}");
        }
    }

    /// Dopóki `γ` mieści się w precyzji `f64`, nierówność jest OSTRA. Powyżej
    /// `γ ≈ 10⁸` różnica `1 − β` schodzi poniżej `10⁻¹⁶` i zaokrągla się do zera —
    /// `β` wychodzi wtedy dokładnie `1`. To jest granica arytmetyki, nie fizyki,
    /// i lepiej ją mieć zapisaną w teście niż odkrywać ją w wyniku biegu.
    #[test]
    fn below_the_precision_limit_the_inequality_is_strict() {
        for magnitude in [1e-6, 1.0, 1e3, 1e6] {
            let b = beta(ELECTRON, vec3(magnitude, 0.0, 0.0));
            assert!(b < 1.0, "|p|={magnitude} β={b}");
        }
        // A tuż nad granicą — już nie. Bez tej asercji test wyżej wyglądałby na
        // słabszy, niż jest naprawdę.
        assert_eq!(beta(ELECTRON, vec3(1e12, 0.0, 0.0)), 1.0);
    }

    /// Granica nierelatywistyczna: `T → ½mv²`. Jedyny niezależny sprawdzian, że
    /// czynniki są na miejscach, skoro `c` nie występuje jawnie.
    #[test]
    fn kinetic_energy_matches_the_classical_limit() {
        let b = vec3(1e-4, 0.0, 0.0);
        let p = momentum_from_beta(MUON, b).unwrap();
        let t = kinetic_energy(MUON, p);
        let classical = 0.5 * MUON * b.norm_squared();
        assert!((t - classical).abs() / classical < 1e-6, "T={t} vs {classical}");
    }

    #[test]
    fn momentum_and_velocity_are_inverses() {
        for b in [0.0, 0.1, 0.5, 0.9, 0.999, 0.999_999] {
            let want = vec3(b, 0.0, 0.0);
            let p = momentum_from_beta(MUON, want).unwrap();
            let got = velocity(MUON, p);
            assert!((got.x - b).abs() < 1e-12, "β={b} wróciło jako {}", got.x);
        }
    }

    #[test]
    fn superluminal_input_is_rejected() {
        assert!(momentum_from_beta(MUON, vec3(1.0, 0.0, 0.0)).is_err());
        assert!(momentum_from_beta(MUON, vec3(0.8, 0.8, 0.0)).is_err());
        assert!(momentum_from_beta(MUON, vec3(f64::NAN, 0.0, 0.0)).is_err());
    }

    #[test]
    fn kinetic_input_works_for_massless_and_massive() {
        let dir = vec3(0.0, 0.0, 2.0);
        let photon = momentum_from_kinetic(0.0, 7.0, dir);
        assert!((photon.norm() - 7.0).abs() < 1e-12);
        assert!((energy(0.0, photon) - 7.0).abs() < 1e-12);

        let muon = momentum_from_kinetic(MUON, 900.0, dir);
        assert!((energy(MUON, muon) - (MUON + 900.0)).abs() < 1e-9);
        // Kierunek musi zostać zachowany, a długość podmieniona.
        assert!(muon.x == 0.0 && muon.y == 0.0 && muon.z > 0.0);
    }

    /// Pchnięcie cząstki spoczywającej ma dać podręcznikowe `E = γm`, `p = γmβ`.
    #[test]
    fn boosting_a_particle_at_rest_gives_gamma_m() {
        let b: f64 = 0.6;
        let gamma = 1.0 / (1.0 - b * b).sqrt();
        let moved = FourMomentum::at_rest(MUON).boost(vec3(b, 0.0, 0.0));
        assert!((moved.energy - gamma * MUON).abs() / MUON < 1e-12);
        assert!((moved.momentum.x - gamma * MUON * b).abs() / MUON < 1e-12);
        assert!(moved.momentum.y.abs() < 1e-12 && moved.momentum.z.abs() < 1e-12);
    }

    /// Pchnięcie i pchnięcie przeciwne muszą wrócić do punktu wyjścia. To łapie
    /// pomyłkę w znaku i w członie `(γ−1)/β²` naraz.
    #[test]
    fn boost_is_reversible() {
        let start = FourMomentum::on_shell(MUON, vec3(30.0, -12.0, 5.0));
        let b = vec3(0.3, -0.5, 0.11);
        let there = start.boost(b);
        let back = there.boost(-b);
        assert!((back.energy - start.energy).abs() / start.energy < 1e-12);
        assert!((back.momentum - start.momentum).norm() / start.momentum.norm() < 1e-12);
    }

    /// Masa niezmiennicza jest niezmiennicza — to jest definicja poprawnego
    /// pchnięcia i jedyny test, który nie zależy od układu odniesienia.
    #[test]
    fn invariant_mass_survives_any_boost() {
        let start = FourMomentum::on_shell(MUON, vec3(4.0, 0.0, -9.0));
        for b in [
            vec3(0.9, 0.0, 0.0),
            vec3(0.0, -0.7, 0.0),
            vec3(0.4, 0.4, 0.4),
            vec3(0.0, 0.0, 0.999),
        ] {
            let moved = start.boost(b);
            let m = moved.invariant_mass();
            assert!((m - MUON).abs() / MUON < 1e-10, "β={b:?} dało m={m}");
        }
    }

    /// Foton musi zostać bezmasowy w każdym układzie, choć jego energia się zmienia
    /// (to jest przesunięcie Dopplera).
    #[test]
    fn a_photon_stays_massless_and_shifts_in_energy() {
        let photon = FourMomentum::on_shell(0.0, vec3(0.0, 0.0, 10.0));
        let towards = photon.boost(vec3(0.0, 0.0, 0.5));
        let away = photon.boost(vec3(0.0, 0.0, -0.5));
        assert!(towards.invariant_mass() < 1e-9);
        assert!(away.invariant_mass() < 1e-9);
        assert!(towards.energy > photon.energy, "brak przesunięcia ku niebieskiemu");
        assert!(away.energy < photon.energy, "brak przesunięcia ku czerwieni");
        // Iloczyn energii przesuniętych w obie strony to niezmiennik Dopplera.
        assert!((towards.energy * away.energy - 100.0).abs() < 1e-9);
    }

    /// Pchnięcie prędkością układu spoczynkowego musi zatrzymać cząstkę.
    #[test]
    fn boosting_by_its_own_velocity_brings_a_particle_to_rest() {
        let p = FourMomentum::on_shell(MUON, vec3(200.0, -50.0, 30.0));
        let at_rest = p.boost(-p.boost_velocity());
        assert!(at_rest.momentum.norm() / p.momentum.norm() < 1e-12);
        assert!((at_rest.energy - MUON).abs() / MUON < 1e-12);
    }

    /// Rozpad `π⁰ → γγ`: dwa fotony po `m/2` każdy. Liczba znana bez rachunku,
    /// więc sprawdza wzór na `p*` w najprostszym możliwym przypadku.
    #[test]
    fn two_body_momentum_of_a_symmetric_massless_pair_is_half_the_mass() {
        let pi0 = 134.9768;
        let p = two_body_momentum(pi0, 0.0, 0.0).unwrap();
        assert!((p - pi0 / 2.0).abs() < 1e-12);
    }

    /// `π⁺ → μ⁺ ν`: pęd mionu to 29,79 MeV/c — wartość z tablic PDG, policzona
    /// niezależnie od tego kodu.
    #[test]
    fn two_body_momentum_matches_the_pion_decay_from_tables() {
        let p = two_body_momentum(139.570_39, MUON, 0.0).unwrap();
        assert!((p - 29.792).abs() < 0.01, "p* = {p} zamiast 29,79 MeV/c");
    }

    #[test]
    fn a_forbidden_decay_has_no_momentum() {
        assert!(two_body_momentum(ELECTRON, MUON, 0.0).is_none());
        // Dokładnie na progu rozpad jest dozwolony, a produkty spoczywają.
        let threshold = two_body_momentum(2.0 * MUON, MUON, MUON).unwrap();
        assert!(threshold.abs() < 1e-9, "p* = {threshold} zamiast zera");
    }

    /// Tuż nad progiem wynik musi pozostać dokładny — to jest miejsce, w którym
    /// naiwna postać `λ` jako sumy sześciu członów traci wszystkie cyfry.
    #[test]
    fn near_threshold_the_momentum_stays_accurate() {
        let m = 100.0;
        let excess = 1e-9;
        let parent = 2.0 * m + excess;
        let p = two_body_momentum(parent, m, m).unwrap();
        // Nadwyżka `ΔM` dzieli się po równo, więc każda cząstka dostaje `T = ΔM/2`,
        // a nierelatywistycznie `p = √(2mT) = √(m·ΔM)`.
        let expected = (m * excess).sqrt();
        assert!(
            (p - expected).abs() / expected < 1e-4,
            "p* = {p}, a przybliżenie progowe daje {expected}"
        );
    }

    #[test]
    fn four_momenta_add_and_sum() {
        let a = FourMomentum::on_shell(MUON, vec3(1.0, 2.0, 3.0));
        let b = FourMomentum::on_shell(0.0, vec3(-1.0, 0.5, 0.0));
        let total: FourMomentum = [a, b].into_iter().sum();
        assert_eq!(total, a + b);
        assert!((total.energy - (a.energy + b.energy)).abs() < 1e-12);
        // Masa niezmiennicza pary jest większa niż suma mas — energia względna też waży.
        assert!(total.invariant_mass() > MUON);
    }
}
