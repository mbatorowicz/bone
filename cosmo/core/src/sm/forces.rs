//! Cztery oddziaływania w jednej pętli — i zmierzony udział każdego z nich.
//!
//! # Coulomb jedzie na solverze grawitacji
//!
//! To nie jest sztuczka oszczędnościowa, tylko stwierdzenie faktu: prawo Coulomba
//! i prawo powszechnego ciążenia to **to samo równanie**.
//!
//! ```text
//! F = −G mᵢmⱼ r̂ / r²        F = +k qᵢqⱼ r̂ / r²
//! ```
//!
//! Różnią się tylko tym, co wstawić w miejsce „ładunku", i znakiem stałej. Solver
//! z [`crate::sr::backends`] liczy `F_i = −g m_i Σ m_j …`, więc wywołany
//! z ładunkami zamiast mas i z `g = −k` daje Coulomba co do znaku i co do wartości —
//! z odpychaniem ładunków jednoimiennych włącznie. Dotyczy to obu solverów:
//! dokładnego `O(N²)` i siatkowego, bo splot z jądrem jest liniowy, a ujemna
//! „gęstość" jest w nim legalna.
//!
//! Praktyczna konsekwencja: chmura obojętna nie ma monopola, więc solver siatkowy
//! widzi tylko ekranowanie — to jest fizycznie poprawne i to jest powód, dla którego
//! plazma daje się liczyć na siatce, a chmura naładowana nie.
//!
//! # Dwa solvery, nie jeden
//!
//! Gdy włączone są **i** grawitacja, **i** Coulomb, powstają dwa osobne solwery.
//! Siatka trzyma widmo jądra w pamięci podręcznej z kluczem zawierającym `g`, więc
//! jeden solver wołany naprzemiennie z dwiema stałymi przebudowywałby jądro
//! dwukrotnie **w każdym kroku** — a przebudowa to dwie transformaty na podwojonej
//! siatce, czyli najdroższa rzecz w całym kroku.
//!
//! # Silne i słabe idą pętlą po parach
//!
//! Oba są krótkozasięgowe, więc siatka nie ma czego przyspieszać, a zysk z niej
//! byłby ujemny: oczko siatki jest większe niż zasięg tych oddziaływań.

use crate::sm::config::ForceConfig;
use crate::sm::particles::Particle;
use crate::sm::state::State;
use crate::sm::units;
use crate::sr::backends::{Backend, Exact};
use crate::sr::config::BackendKind;
use crate::sr::state::Field;
use crate::mesh::Mesh;
use crate::vec3::{Vec3, ZERO};

/// Które oddziaływanie. Kolejność jest kolejnością kolumn w diagnostyce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interaction {
    Electromagnetic,
    Strong,
    Weak,
    Gravitational,
}

impl Interaction {
    pub const ALL: [Interaction; 4] = [
        Self::Electromagnetic,
        Self::Strong,
        Self::Weak,
        Self::Gravitational,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Electromagnetic => "elektromagnetyczne",
            Self::Strong => "silne",
            Self::Weak => "słabe",
            Self::Gravitational => "grawitacja",
        }
    }

    pub fn short(self) -> &'static str {
        match self {
            Self::Electromagnetic => "EM",
            Self::Strong => "silne",
            Self::Weak => "słabe",
            Self::Gravitational => "graw.",
        }
    }

    /// Kolumna w tablicach [`FieldSet::energy`] i [`FieldSet::rms`].
    pub fn index(self) -> usize {
        self as usize
    }
}

/// Wynik jednego policzenia sił.
#[derive(Clone, Debug)]
pub struct FieldSet {
    /// Suma sił wszystkich włączonych oddziaływań, MeV/fm.
    pub force: Vec<Vec3>,
    /// Energia potencjalna każdego oddziaływania z osobna, MeV.
    pub energy: [f64; 4],
    /// Średnia kwadratowa siły każdego oddziaływania, MeV/fm.
    ///
    /// Ta tablica jest powodem, dla którego wyłączone oddziaływanie nadal ma tu
    /// swoją kolumnę: dopiero porównanie liczb pozwala **pokazać**, że grawitacja
    /// jest 36 rzędów wielkości słabsza od Coulomba, zamiast to oświadczyć.
    pub rms: [f64; 4],
    /// Zmiękczenie, którym solver naprawdę liczył (siatka nie zejdzie poniżej oczka).
    pub effective_softening: f64,
}

impl FieldSet {
    fn zeros(n: usize, softening: f64) -> Self {
        Self {
            force: vec![ZERO; n],
            energy: [0.0; 4],
            rms: [0.0; 4],
            effective_softening: softening,
        }
    }

    pub fn total_potential_energy(&self) -> f64 {
        self.energy.iter().sum()
    }

    pub fn energy_of(&self, interaction: Interaction) -> f64 {
        self.energy[interaction.index()]
    }

    pub fn rms_of(&self, interaction: Interaction) -> f64 {
        self.rms[interaction.index()]
    }

    /// Największa siła działająca na pojedynczą cząstkę — do doboru kroku.
    pub fn max_force(&self) -> f64 {
        self.force.iter().map(|f| f.norm()).fold(0.0, f64::max)
    }
}

/// Zestaw solverów dla jednego biegu.
pub struct Forces {
    cfg: ForceConfig,
    electromagnetic: Option<Box<dyn Backend>>,
    gravitational: Option<Box<dyn Backend>>,
}

impl Forces {
    pub fn new(cfg: ForceConfig, n_particles: usize) -> Self {
        Self {
            electromagnetic: cfg.coulomb.then(|| make_backend(&cfg, n_particles)),
            gravitational: cfg.gravity.then(|| make_backend(&cfg, n_particles)),
            cfg,
        }
    }

    pub fn config(&self) -> &ForceConfig {
        &self.cfg
    }

    /// Podmień nastawy, które wolno zmieniać w trakcie biegu.
    ///
    /// Włączenie oddziaływania w trakcie wymaga zbudowania solvera, więc dzieje się
    /// tu, a nie w silniku. Zmiana `grid` czy `backend` w locie jest świadomie
    /// nieobsługiwana: solver siatkowy trzyma stan (pudło, widmo jądra), a podmiana
    /// go w połowie biegu dałaby skok siły nie do odróżnienia od fizyki.
    pub fn apply_runtime(&mut self, live: &ForceConfig, n_particles: usize) {
        self.cfg.coulomb = live.coulomb;
        self.cfg.gravity = live.gravity;
        self.cfg.strong = live.strong;
        self.cfg.weak = live.weak;
        self.cfg.softening = live.softening;
        self.cfg.short_range_cutoff = live.short_range_cutoff;

        if self.cfg.coulomb && self.electromagnetic.is_none() {
            self.electromagnetic = Some(make_backend(&self.cfg, n_particles));
        }
        if self.cfg.gravity && self.gravitational.is_none() {
            self.gravitational = Some(make_backend(&self.cfg, n_particles));
        }
    }

    pub fn describe(&self) -> String {
        let mut on: Vec<&str> = Vec::new();
        if self.cfg.coulomb {
            on.push(Interaction::Electromagnetic.short());
        }
        if self.cfg.strong {
            on.push(Interaction::Strong.short());
        }
        if self.cfg.weak {
            on.push(Interaction::Weak.short());
        }
        if self.cfg.gravity {
            on.push(Interaction::Gravitational.short());
        }
        let solver = match self.electromagnetic.as_ref().or(self.gravitational.as_ref()) {
            Some(backend) => backend.describe(),
            None => "bez solvera".to_string(),
        };
        if on.is_empty() {
            return format!("brak oddziaływań ({solver})");
        }
        format!("{} ({solver})", on.join(" + "))
    }

    /// Czy którykolwiek włączony solver liczy w przybliżeniu.
    pub fn approximate(&self) -> bool {
        [&self.electromagnetic, &self.gravitational]
            .into_iter()
            .flatten()
            .any(|b| b.approximate())
    }

    /// Policz siły i energie potencjalne dla podanego stanu.
    pub fn compute(&mut self, state: &State) -> FieldSet {
        let n = state.n();
        // Bez włączonego oddziaływania zmiękczenie nie jest niczym — nie ma siły,
        // którą miałoby zmiękczać. Zerowa wartość mówi diagnostyce i doborowi kroku,
        // że w tym biegu nie ma skali długości, względem której cokolwiek mierzyć;
        // wpisanie tu `cfg.softening` dawałoby ostrzeżenia o stłoczeniu w gazie,
        // w którym cząstki się nie widzą.
        let announced = if self.cfg.none_enabled() {
            0.0
        } else {
            self.cfg.softening
        };
        let mut out = FieldSet::zeros(n, announced);
        if n < 2 {
            return out;
        }
        let eps = self.cfg.softening;

        if let Some(backend) = self.electromagnetic.as_mut() {
            // Znak: solver liczy `F = −g·qᵢ·Σqⱼ…`, więc `g = −k` daje `F = +k qᵢqⱼ…`,
            // czyli odpychanie ładunków jednoimiennych.
            let field = backend.compute(&state.positions, state.charges(), -units::COULOMB, eps);
            out.effective_softening = backend.effective_softening(eps);
            accumulate(&mut out, Interaction::Electromagnetic, &field, state.charges());
        }

        if let Some(backend) = self.gravitational.as_mut() {
            let field = backend.compute(&state.positions, state.masses(), units::GRAVITY, eps);
            out.effective_softening = out.effective_softening.max(backend.effective_softening(eps));
            accumulate(&mut out, Interaction::Gravitational, &field, state.masses());
        }

        if self.cfg.strong {
            let field = strong_field(state, eps);
            accumulate_pairs(&mut out, Interaction::Strong, &field);
        }

        if self.cfg.weak {
            let field = weak_field(state, self.weak_cutoff());
            accumulate_pairs(&mut out, Interaction::Weak, &field);
        }

        out
    }

    /// Zasięg, poza którym oddziaływanie słabe jest ścinane do zera.
    ///
    /// `40λ_W` to `e⁻⁴⁰ ≈ 4·10⁻¹⁸` reszty — poniżej precyzji `f64` względem członu
    /// wiodącego, więc obcięcie nie jest przybliżeniem, tylko pominięciem zera.
    fn weak_cutoff(&self) -> f64 {
        (40.0 * units::WEAK_RANGE).min(self.cfg.short_range_cutoff)
    }
}

fn make_backend(cfg: &ForceConfig, n_particles: usize) -> Box<dyn Backend> {
    let kind = match cfg.backend {
        BackendKind::Auto if crate::sr::backends::prefer_exact(n_particles, cfg.grid) => {
            BackendKind::Exact
        }
        BackendKind::Auto => BackendKind::Mesh,
        explicit => explicit,
    };
    match kind {
        BackendKind::Exact => Box::new(Exact::new()),
        BackendKind::Mesh | BackendKind::Auto => Box::new(Mesh::new(cfg.grid, cfg.box_margin)),
    }
}

fn accumulate(out: &mut FieldSet, which: Interaction, field: &Field, weights: &[f64]) {
    let index = which.index();
    out.energy[index] = field.energy(weights);
    out.rms[index] = rms(&field.force);
    for (total, part) in out.force.iter_mut().zip(field.force.iter()) {
        *total += *part;
    }
}

fn accumulate_pairs(out: &mut FieldSet, which: Interaction, field: &PairField) {
    let index = which.index();
    out.energy[index] = field.energy;
    out.rms[index] = rms(&field.force);
    for (total, part) in out.force.iter_mut().zip(field.force.iter()) {
        *total += *part;
    }
}

fn rms(force: &[Vec3]) -> f64 {
    if force.is_empty() {
        return 0.0;
    }
    (force.iter().map(|f| f.norm_squared()).sum::<f64>() / force.len() as f64).sqrt()
}

/// Siła i energia z pętli po parach.
struct PairField {
    force: Vec<Vec3>,
    energy: f64,
}

/// Współczynnik członu jednogluonowego dla pary kolorowych cząstek.
///
/// Kolor nie jest tu śledzony — cząstka ma tylko informację, że **jakiś** kolor
/// niesie. Wybieramy więc kanał **najbardziej przyciągający**, bo to on decyduje
/// o istnieniu stanu związanego:
///
/// - kwark z antykwarkiem w singlecie koloru: `C = 4/3`;
/// - dwa kwarki w antytryplecie (tak wiążą się bariony): `C = 2/3`.
///
/// Przybliżenie jest dobre dla hadronu w stanie podstawowym i **zawodzi dla układów
/// wielokwarkowych**, w których kanały odpychające są równie ważne. Wtedy ta
/// symulacja pokazuje układ silniej związany, niż jest naprawdę.
fn color_factor(a: Particle, b: Particle) -> f64 {
    if a.anti == b.anti {
        2.0 / 3.0
    } else {
        4.0 / 3.0
    }
}

/// Potencjał Cornella między cząstkami kolorowymi.
///
/// ```text
/// V(r) = −C·α_s·ħc/r + σ·r
/// ```
///
/// Człon liniowy **nie jest obcinany zasięgiem**. Obcięcie byłoby wygodne
/// obliczeniowo i byłoby kłamstwem dokładnie o tę własność, dla której ten
/// potencjał się liczy: struna nie słabnie z odległością. W naturze pęka, tworząc
/// nową parę kwark-antykwark — a produkcji par ten model nie ma, więc struna
/// rozciąga się tu w nieskończoność. To jest **jedyne** miejsce, w którym uwięzienie
/// jest tu przesadzone, i wolimy je mieć przesadzone niż zniesione.
fn strong_field(state: &State, softening: f64) -> PairField {
    let n = state.n();
    let mut force = vec![ZERO; n];
    let mut energy = 0.0;
    let eps2 = softening * softening;

    let colored: Vec<usize> = (0..n).filter(|i| state.kinds[*i].colored()).collect();
    for (position, &i) in colored.iter().enumerate() {
        for &j in &colored[position + 1..] {
            let d = state.positions[i] - state.positions[j];
            let r = (d.norm_squared() + eps2).sqrt();
            let c = color_factor(state.kinds[i], state.kinds[j]);
            let coulomb_like = c * units::ALPHA_S * units::HBAR_C;

            energy += -coulomb_like / r + units::STRING_TENSION * r;
            // dV/dr = +C·α_s·ħc/r² + σ, a siła to −dV/dr wzdłuż r̂ — obie składowe
            // przyciągające, więc łączny znak jest ujemny przy r̂ = (xᵢ−xⱼ)/r.
            let magnitude = coulomb_like / (r * r) + units::STRING_TENSION;
            let pull = d * (-magnitude / r);
            force[i] += pull;
            force[j] -= pull;
        }
    }
    PairField { force, energy }
}

/// Oddziaływanie słabe jako potencjał Yukawy o zasięgu `ħc/M_W`.
///
/// ```text
/// V(r) = −α_w·ħc·e^(−r/λ)/r,    λ = ħc/M_W ≈ 0,0025 fm
/// ```
///
/// Dwa zastrzeżenia, oba istotne:
///
/// 1. **Oddziaływanie słabe nie jest potencjałem statycznym.** Wymiana ciężkiego
///    bozonu między fermionami zmienia zapach i nie sprowadza się do siły
///    zachowawczej. Ta postać jest oszacowaniem co do rzędu wielkości, a nie
///    obliczeniem — i jest tu po to, żeby ten rząd wielkości **pokazać**.
/// 2. **Widocznym skutkiem oddziaływania słabego są rozpady**, nie przekaz pędu.
///    Rozpady liczy [`crate::sm::decays`], i to tam ten mechanizm robi cokolwiek
///    obserwowalnego.
fn weak_field(state: &State, cutoff: f64) -> PairField {
    let n = state.n();
    let mut force = vec![ZERO; n];
    let mut energy = 0.0;
    let coupling = units::ALPHA_WEAK * units::HBAR_C;
    let lambda = units::WEAK_RANGE;
    let cutoff2 = cutoff * cutoff;

    // Oddziaływaniu słabemu podlegają wszystkie fermiony; bozony pomijamy.
    let fermions: Vec<usize> = (0..n)
        .filter(|i| state.kinds[*i].data().spin_doubled % 2 == 1)
        .collect();
    for (position, &i) in fermions.iter().enumerate() {
        for &j in &fermions[position + 1..] {
            let d = state.positions[i] - state.positions[j];
            let r2 = d.norm_squared();
            if r2 > cutoff2 || r2 == 0.0 {
                continue;
            }
            let r = r2.sqrt();
            let decay = (-r / lambda).exp();
            energy += -coupling * decay / r;
            // −dV/dr = −α_w ħc e^(−r/λ)(1/r² + 1/(λr)) — przyciąganie.
            let magnitude = coupling * decay * (1.0 / (r * r) + 1.0 / (lambda * r));
            let pull = d * (-magnitude / r);
            force[i] += pull;
            force[j] -= pull;
        }
    }
    PairField { force, energy }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::particles::Species;
    use crate::vec3::vec3;

    fn pair(a: Particle, b: Particle, separation: f64) -> State {
        State::new(
            vec![ZERO, vec3(separation, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![a, b],
        )
        .unwrap()
    }

    fn only(interaction: Interaction) -> ForceConfig {
        ForceConfig {
            coulomb: interaction == Interaction::Electromagnetic,
            gravity: interaction == Interaction::Gravitational,
            strong: interaction == Interaction::Strong,
            weak: interaction == Interaction::Weak,
            softening: 0.0,
            backend: BackendKind::Exact,
            ..ForceConfig::default()
        }
    }

    /// Ładunki jednoimienne muszą się ODPYCHAĆ. To jest jedyny test, który
    /// odróżnia poprawne użycie solvera grawitacji od użycia go ze złym znakiem —
    /// a zły znak dałby plazmę, która się zapada zamiast rozbiegać.
    #[test]
    fn like_charges_repel_and_opposite_charges_attract() {
        let e = Particle::of(Species::Electron);
        let p = Particle::of(Species::Proton);

        let mut forces = Forces::new(only(Interaction::Electromagnetic), 2);
        let repelling = forces.compute(&pair(e, e, 1.0));
        assert!(repelling.force[0].x < 0.0, "elektrony się przyciągnęły");
        assert!(repelling.force[1].x > 0.0);

        let attracting = forces.compute(&pair(e, p, 1.0));
        assert!(attracting.force[0].x > 0.0, "elektron i proton się odepchnęły");
        assert!(attracting.force[1].x < 0.0);
    }

    /// Wartość bezwzględna względem wzoru, który da się policzyć w pamięci:
    /// dwa ładunki elementarne w odległości femtometra to `1,44 MeV/fm`.
    #[test]
    fn coulomb_magnitude_matches_the_textbook_value() {
        let p = Particle::of(Species::Proton);
        let mut forces = Forces::new(only(Interaction::Electromagnetic), 2);
        let field = forces.compute(&pair(p, p, 1.0));
        assert!(
            (field.force[1].x - units::COULOMB).abs() / units::COULOMB < 1e-12,
            "F = {} MeV/fm zamiast {}",
            field.force[1].x,
            units::COULOMB
        );
    }

    /// Energia potencjalna pary jednoimiennej jest DODATNIA, przeciwnych — ujemna.
    /// Znak energii decyduje o tym, czy układ jest związany, więc pomyłka tutaj
    /// przechodzi przez cały bilans energii.
    #[test]
    fn coulomb_energy_has_the_right_sign_and_size() {
        let e = Particle::of(Species::Electron);
        let p = Particle::of(Species::Proton);
        let mut forces = Forces::new(only(Interaction::Electromagnetic), 2);

        let same = forces.compute(&pair(e, e, 2.0));
        let opposite = forces.compute(&pair(e, p, 2.0));
        let expected = units::COULOMB / 2.0;

        assert!((same.energy_of(Interaction::Electromagnetic) - expected).abs() < 1e-12);
        assert!((opposite.energy_of(Interaction::Electromagnetic) + expected).abs() < 1e-12);
    }

    /// Sedno tego, po co grawitacja jest w tym modelu włączalna: żeby dało się
    /// ZOBACZYĆ, jak bardzo jest nieistotna. Iloraz `~10⁻³⁶` jest tu wynikiem
    /// pomiaru na dwóch protonach, a nie liczbą przepisaną z podręcznika.
    #[test]
    fn gravity_is_thirty_six_orders_below_the_electric_force() {
        let p = Particle::of(Species::Proton);
        let state = pair(p, p, 1.0);
        let electric = Forces::new(only(Interaction::Electromagnetic), 2)
            .compute(&state)
            .force[1]
            .norm();
        let gravity = Forces::new(only(Interaction::Gravitational), 2)
            .compute(&state)
            .force[1]
            .norm();
        let ratio = gravity / electric;
        assert!(
            (1e-37..1e-35).contains(&ratio),
            "grawitacja/elektryczność = {ratio:e}"
        );
    }

    /// Grawitacja przyciąga zawsze — także cząstki o jednakowym ładunku, dla których
    /// siła elektryczna odpycha. To rozróżnia dwa wywołania tego samego solvera.
    #[test]
    fn gravity_always_attracts() {
        let p = Particle::of(Species::Proton);
        let field = Forces::new(only(Interaction::Gravitational), 2).compute(&pair(p, p, 1.0));
        assert!(field.force[0].x > 0.0, "grawitacja odepchnęła protony");
        assert!(field.energy_of(Interaction::Gravitational) < 0.0, "dodatnia energia wiązania");
    }

    /// Uwięzienie: siła między kwarkiem a antykwarkiem NIE maleje do zera
    /// z odległością, tylko dąży do napięcia struny. To jest cała treść
    /// oddziaływania silnego w tym modelu.
    #[test]
    fn the_strong_force_does_not_fall_off_with_distance() {
        let q = Particle::of(Species::Up);
        let anti = Particle::anti_of(Species::Up);
        let mut forces = Forces::new(only(Interaction::Strong), 2);

        let near = forces.compute(&pair(q, anti, 1.0)).force[0].norm();
        let far = forces.compute(&pair(q, anti, 100.0)).force[0].norm();

        assert!(near > far, "człon Coulombowski nie działa z bliska");
        assert!(
            (far - units::STRING_TENSION).abs() / units::STRING_TENSION < 1e-3,
            "na 100 fm siła to {far} MeV/fm, a napięcie struny to {}",
            units::STRING_TENSION
        );
    }

    /// Energia rozdzielenia pary rośnie liniowo — rozerwanie kwarków kosztuje
    /// nieskończenie wiele, i to jest właśnie uwięzienie.
    #[test]
    fn separating_quarks_costs_energy_without_limit() {
        let q = Particle::of(Species::Up);
        let anti = Particle::anti_of(Species::Up);
        let mut forces = Forces::new(only(Interaction::Strong), 2);
        let energy_at = |r: f64, f: &mut Forces| f.compute(&pair(q, anti, r)).energy_of(Interaction::Strong);

        let (a, b, c) = (
            energy_at(10.0, &mut forces),
            energy_at(20.0, &mut forces),
            energy_at(30.0, &mut forces),
        );
        assert!(a < b && b < c, "energia nie rośnie: {a}, {b}, {c}");
        // Przyrosty muszą być równe — to jest definicja członu liniowego.
        assert!(((b - a) - (c - b)).abs() / (b - a) < 1e-3);
    }

    /// Kwark z antykwarkiem wiąże się mocniej niż dwa kwarki — kanał singletowy
    /// ma współczynnik `4/3`, antytrypletowy `2/3`.
    #[test]
    fn the_singlet_channel_binds_twice_as_hard_as_the_antitriplet() {
        let q = Particle::of(Species::Up);
        let anti = Particle::anti_of(Species::Up);
        let mut forces = Forces::new(only(Interaction::Strong), 2);
        // Blisko dominuje człon jednogluonowy, więc iloraz sił odsłania współczynnik.
        let singlet = forces.compute(&pair(q, anti, 0.01)).force[0].norm();
        let triplet = forces.compute(&pair(q, q, 0.01)).force[0].norm();
        assert!((singlet / triplet - 2.0).abs() < 0.01, "iloraz {}", singlet / triplet);
    }

    /// Cząstka bez koloru nie czuje oddziaływania silnego. Bez tego testu proton
    /// (który jest biały) zachowywałby się jak swobodny kwark.
    #[test]
    fn colorless_particles_ignore_the_strong_force() {
        let p = Particle::of(Species::Proton);
        let field = Forces::new(only(Interaction::Strong), 2).compute(&pair(p, p, 1.0));
        assert_eq!(field.force[0], ZERO);
        assert_eq!(field.energy_of(Interaction::Strong), 0.0);
    }

    /// Sedno oddziaływania słabego: na odległości, którą ta symulacja rozdziela,
    /// jego udział jako siły jest znikomy. Test mierzy to na parze elektronów
    /// oddalonych o promień protonu.
    #[test]
    fn the_weak_force_is_negligible_at_any_resolvable_distance() {
        let e = Particle::of(Species::Electron);
        let state = pair(e, e, 0.8);
        let weak = Forces::new(only(Interaction::Weak), 2).compute(&state).force[0].norm();
        let electric = Forces::new(only(Interaction::Electromagnetic), 2)
            .compute(&state)
            .force[0]
            .norm();
        assert!(
            weak / electric < 1e-100,
            "słabe/elektryczne = {:e} — za dużo jak na zasięg 0,0025 fm",
            weak / electric
        );
    }

    /// Poniżej zasięgu oddziaływanie słabe jednak istnieje — inaczej test wyżej
    /// przechodziłby także dla implementacji zwracającej stałe zero.
    #[test]
    fn the_weak_force_exists_below_its_range() {
        let e = Particle::of(Species::Electron);
        let close = units::WEAK_RANGE / 2.0;
        let field = Forces::new(only(Interaction::Weak), 2).compute(&pair(e, e, close));
        assert!(field.force[0].norm() > 0.0, "słabe zniknęło w swoim zasięgu");
        assert!(field.energy_of(Interaction::Weak) < 0.0, "przyciąganie ma ujemną energię");
    }

    /// Trzecie prawo Newtona dla każdego oddziaływania z osobna. Najczulszy test na
    /// pomylony indeks w pętli po parach.
    #[test]
    fn every_interaction_conserves_total_force() {
        let state = State::new(
            vec![
                vec3(0.0, 0.0, 0.0),
                vec3(0.7, 0.2, -0.1),
                vec3(-0.4, 0.9, 0.3),
                vec3(0.1, -0.6, 0.8),
            ],
            vec![ZERO; 4],
            vec![
                Particle::of(Species::Up),
                Particle::anti_of(Species::Down),
                Particle::of(Species::Electron),
                Particle::anti_of(Species::Electron),
            ],
        )
        .unwrap();

        for interaction in Interaction::ALL {
            let field = Forces::new(only(interaction), state.n()).compute(&state);
            let total: Vec3 = field.force.iter().copied().sum();
            let scale: f64 = field.force.iter().map(|f| f.norm()).sum::<f64>().max(1e-300);
            assert!(
                total.norm() / scale < 1e-10,
                "{}: residuum {:e}",
                interaction.label(),
                total.norm() / scale
            );
        }
    }

    /// Włączenie dwóch oddziaływań ma dawać sumę, a nie jedno z nich.
    #[test]
    fn enabled_interactions_add_up() {
        let p = Particle::of(Species::Proton);
        let state = pair(p, p, 1.0);
        let mut both = Forces::new(
            ForceConfig {
                coulomb: true,
                gravity: true,
                softening: 0.0,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
            2,
        );
        let field = both.compute(&state);
        let electric = Forces::new(only(Interaction::Electromagnetic), 2)
            .compute(&state)
            .force[1];
        let gravity = Forces::new(only(Interaction::Gravitational), 2)
            .compute(&state)
            .force[1];
        assert!((field.force[1] - (electric + gravity)).norm() < 1e-30);
        assert!(field.rms_of(Interaction::Electromagnetic) > 0.0);
        assert!(field.rms_of(Interaction::Gravitational) > 0.0);
        assert_eq!(field.rms_of(Interaction::Strong), 0.0);
    }

    #[test]
    fn a_lone_particle_and_an_empty_state_feel_nothing() {
        let mut forces = Forces::new(ForceConfig::default(), 1);
        let lone = State::new(
            vec![ZERO],
            vec![ZERO],
            vec![Particle::of(Species::Electron)],
        )
        .unwrap();
        assert_eq!(forces.compute(&lone).force, vec![ZERO]);
        assert!(forces.compute(&State::empty()).force.is_empty());
    }

    /// Siatka musi być PRZYBLIŻENIEM solvera dokładnego także dla ładunków —
    /// czyli dla „gęstości" o obu znakach, której jądro FFT nigdy wcześniej nie
    /// widziało w tym projekcie.
    #[test]
    fn the_mesh_solver_agrees_with_exact_for_signed_charges() {
        let mut rng = crate::rng::Rng::seeded(4);
        let n = 400;
        let positions: Vec<Vec3> = (0..n).map(|_| rng.normal_vec(0.0, 10.0)).collect();
        let kinds: Vec<Particle> = (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    Particle::of(Species::Electron)
                } else {
                    Particle::anti_of(Species::Electron)
                }
            })
            .collect();
        let state = State::new(positions, vec![ZERO; n], kinds).unwrap();

        let base = ForceConfig {
            coulomb: true,
            gravity: false,
            softening: 1.0,
            grid: 32,
            ..ForceConfig::default()
        };
        let mesh = Forces::new(
            ForceConfig {
                backend: BackendKind::Mesh,
                ..base
            },
            n,
        )
        .compute(&state);
        let exact = Forces::new(
            ForceConfig {
                backend: BackendKind::Exact,
                softening: mesh.effective_softening,
                ..base
            },
            n,
        )
        .compute(&state);

        let typical = rms(&exact.force);
        let error = (mesh
            .force
            .iter()
            .zip(exact.force.iter())
            .map(|(a, b)| (*a - *b).norm_squared())
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(
            error / typical < 0.2,
            "siatka rozjechała się z dokładnym o {:.1}%",
            100.0 * error / typical
        );
    }

    #[test]
    fn describe_names_the_enabled_interactions() {
        let text = Forces::new(
            ForceConfig {
                coulomb: true,
                strong: true,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
            10,
        )
        .describe();
        assert!(text.contains("EM") && text.contains("silne"), "{text}");
        assert!(!text.contains("graw."), "{text}");

        let none = Forces::new(
            ForceConfig {
                coulomb: false,
                ..ForceConfig::default()
            },
            10,
        )
        .describe();
        assert!(none.contains("brak oddziaływań"), "{none}");
    }

    /// Włączenie oddziaływania w trakcie biegu musi zbudować solver, a nie
    /// przemilczeć prośbę.
    #[test]
    fn runtime_switch_builds_the_missing_solver() {
        let mut forces = Forces::new(only(Interaction::Electromagnetic), 2);
        let p = Particle::of(Species::Proton);
        let state = pair(p, p, 1.0);
        assert_eq!(forces.compute(&state).rms_of(Interaction::Gravitational), 0.0);

        let live = ForceConfig {
            gravity: true,
            ..*forces.config()
        };
        forces.apply_runtime(&live, 2);
        assert!(forces.compute(&state).rms_of(Interaction::Gravitational) > 0.0);
    }
}
