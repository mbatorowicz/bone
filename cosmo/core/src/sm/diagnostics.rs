//! Co jest mierzone w trakcie biegu i co z tego wynika.
//!
//! Model cząstek ma dwie klasy wielkości zachowanych i traktuje je inaczej,
//! bo są innego rodzaju:
//!
//! - **całkowite** — ładunek, liczba barionowa, liczby leptonowe. Muszą się zgadzać
//!   **co do bitu**. Odchylenie o jeden nie jest błędem zaokrąglenia, tylko błędem
//!   w tablicy kanałów, i jest raportowane jako takie;
//! - **zmiennoprzecinkowe** — energia i pęd. Tu mierzymy dryf, bo całkowanie jest
//!   przybliżone, a solver siatkowy dodatkowo przybliżony przestrzennie.
//!
//! # Skok energii przy rozpadzie
//!
//! Rozpad jest zdarzeniem punktowym: produkty powstają w miejscu matki. Jeśli są
//! naładowane, ich klasyczna energia oddziaływania w chwili narodzin jest ogromna —
//! ograniczona wyłącznie zmiękczeniem. `π⁰ → e⁺e⁻γ` przy `ε = 0,01 fm` daje skok
//! rzędu 100 MeV, czyli tyle, ile waży sam pion.
//!
//! **To nie jest usterka całkowania i nie da się tego wygładzić.** Klasyczna energia
//! Coulomba pary punktowej w zerowej odległości jest nieskończona, a prawdziwa
//! poprawka QED do rozpadu jest rzędu `α`. Model klasyczny nie ma jak trafić między
//! te dwie liczby.
//!
//! Dlatego skok jest **księgowany osobno** ([`Snapshot::transmutation_energy`])
//! i odejmowany od dryfu. Dzięki temu dryf mierzy jakość całkowania, a nie sumę
//! całkowania i tego artefaktu — a sam artefakt jest widoczny jako osobna liczba,
//! po której wielkości od razu widać, czy bieg jest wiarygodny.

use crate::sm::forces::{FieldSet, Interaction};
use crate::sm::particles::Flavour;
use crate::sm::state::State;
use crate::vec3::Vec3;

/// Ile cząstek próbkować przy szukaniu najbliższego sąsiada.
const NEIGHBOUR_SAMPLE: usize = 512;

/// Poniżej ilu zmiękczeń para uchodzi za „stykającą się".
const CONTACT_SOFTENINGS: f64 = 3.0;

/// Wynik pomiaru w jednej chwili.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub time: f64,
    pub step: u64,
    pub n: usize,

    /// Suma energii kinetycznych, MeV.
    pub kinetic: f64,
    /// Suma mas spoczynkowych, MeV. Maleje, gdy rozpad zamienia masę w ruch.
    pub rest_mass: f64,
    /// Energia potencjalna wszystkich oddziaływań, MeV.
    pub potential: f64,
    /// `Σ E + U` — wielkość zachowana.
    pub total_energy: f64,
    /// Względny dryf energii z odjętym skokiem z rozpadów.
    pub energy_drift: f64,
    /// Sumaryczny skok energii potencjalnej z narodzin cząstek, MeV.
    pub transmutation_energy: f64,
    /// `|Σp| / Σ|p|`.
    pub momentum_residual: f64,

    pub charge_thirds: i64,
    pub baryon_thirds: i64,
    pub lepton: i64,
    pub lepton_electron: i64,
    pub lepton_muon: i64,
    pub lepton_tau: i64,
    /// Odchylenie liczb całkowitych od wartości początkowych. Zero albo usterka.
    pub integer_drift: IntegerDrift,

    pub energy_of: [f64; 4],
    pub rms_of: [f64; 4],

    pub beta_mean: f64,
    pub beta_max: f64,
    /// `γ` największe wśród cząstek MASYWNYCH; `None`, gdy wszystkie są bezmasowe.
    pub gamma_max: Option<f64>,
    /// Promień, w którym mieści się połowa cząstek.
    pub half_count_radius: f64,

    pub decayed: u64,
    pub annihilated: u64,

    /// Najmniejsza znaleziona odległość między cząstkami, w jednostkach `ε`.
    pub closest_pair_over_softening: f64,
    /// Ułamek próbki, której najbliższy sąsiad leży bliżej niż `3ε`.
    pub contact_fraction: f64,
    pub softening: f64,
}

/// O ile liczby całkowite odjechały od wartości początkowych.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntegerDrift {
    pub charge_thirds: i64,
    pub baryon_thirds: i64,
    pub lepton: i64,
}

impl IntegerDrift {
    pub fn is_zero(self) -> bool {
        self == Self::default()
    }
}

/// Wartości zapamiętane na starcie, do których porównujemy.
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub total_energy: f64,
    /// Energia, która w ogóle **może** się zmienić: kinetyczna plus potencjalna.
    pub dynamic_scale: f64,
    pub charge_thirds: i64,
    pub baryon_thirds: i64,
    pub lepton: i64,
}

impl Reference {
    pub fn capture(state: &State, field: &FieldSet) -> Self {
        Self {
            total_energy: total_energy(state, field),
            dynamic_scale: dynamic_scale(state, field),
            charge_thirds: state.total_charge_thirds(),
            baryon_thirds: state.total_baryon_thirds(),
            lepton: state.total_lepton(),
        }
    }
}

fn total_energy(state: &State, field: &FieldSet) -> f64 {
    let relativistic: f64 = (0..state.n()).map(|i| state.energy(i)).sum();
    relativistic + field.total_potential_energy()
}

/// Budżet energii, którym całkowanie faktycznie obraca.
///
/// Odniesienie do energii **całkowitej** działa w modelach SR i ΛCDM, bo tam
/// energia spoczynkowa nie występuje. Tutaj występuje i przytłacza wszystko inne:
/// para proton-elektron waży 939 MeV, a jej energia oddziaływania na dystansie
/// femtometrów to ułamek MeV. Dryf odniesiony do 939 MeV pokazywałby `10⁻⁹` również
/// wtedy, gdy całkowanie zgubiło **całą** fizykę oddziaływania — a więc dokładnie
/// wtedy, gdy miał ostrzec.
///
/// Mianownikiem jest więc suma energii kinetycznej i modułu potencjalnej: to jedyne
/// składniki, które krok całkowania może zepsuć. Masa spoczynkowa zmienia się
/// wyłącznie w rozpadzie, a rozpad zachowuje czteropęd dokładnie.
fn dynamic_scale(state: &State, field: &FieldSet) -> f64 {
    let kinetic: f64 = (0..state.n()).map(|i| state.kinetic_energy(i)).sum();
    kinetic + field.total_potential_energy().abs()
}

/// Zmierz wszystko, co jest do zmierzenia w tej chwili.
#[allow(clippy::too_many_arguments)]
pub fn collect(
    state: &State,
    field: &FieldSet,
    reference: Reference,
    transmutation_energy: f64,
    decayed: u64,
    annihilated: u64,
    seed: u64,
) -> Snapshot {
    let n = state.n();
    let total = total_energy(state, field);

    // Dryf liczymy po odjęciu skoku z narodzin, bo ten skok nie jest błędem
    // całkowania — jest ceną klasycznego opisu zdarzenia punktowego.
    //
    // Mianownik jest większym z dwóch budżetów: początkowego i bieżącego. Sam
    // początkowy zawyżałby dryf w biegu, który startuje ze spoczynku i dopiero
    // nabiera energii; sam bieżący zaniżałby go w biegu, który energię traci.
    let scale = reference
        .dynamic_scale
        .max(dynamic_scale(state, field))
        .max(1e-30);
    let energy_drift = (total - transmutation_energy - reference.total_energy) / scale;

    let mut kinetic = 0.0;
    let mut rest_mass = 0.0;
    let mut beta_sum = 0.0;
    let mut beta_max: f64 = 0.0;
    let mut gamma_max: Option<f64> = None;
    let mut momentum_sum = crate::vec3::ZERO;
    let mut momentum_scale = 0.0;

    for i in 0..n {
        kinetic += state.kinetic_energy(i);
        rest_mass += state.kinds[i].mass();
        let beta = state.beta(i);
        beta_sum += beta;
        beta_max = beta_max.max(beta);
        if let Some(gamma) =
            crate::sm::kinematics::gamma(state.kinds[i].mass(), state.momenta[i])
        {
            gamma_max = Some(gamma_max.map_or(gamma, |best: f64| best.max(gamma)));
        }
        momentum_sum += state.momenta[i];
        momentum_scale += state.momenta[i].norm();
    }

    let (closest, contact) = neighbour_stats(state, field.effective_softening, seed);

    Snapshot {
        time: state.time,
        step: state.step,
        n,
        kinetic,
        rest_mass,
        potential: field.total_potential_energy(),
        total_energy: total,
        energy_drift,
        transmutation_energy,
        momentum_residual: if momentum_scale > 0.0 {
            momentum_sum.norm() / momentum_scale
        } else {
            0.0
        },
        charge_thirds: state.total_charge_thirds(),
        baryon_thirds: state.total_baryon_thirds(),
        lepton: state.total_lepton(),
        lepton_electron: state.lepton_of(Flavour::Electron),
        lepton_muon: state.lepton_of(Flavour::Muon),
        lepton_tau: state.lepton_of(Flavour::Tau),
        integer_drift: IntegerDrift {
            charge_thirds: state.total_charge_thirds() - reference.charge_thirds,
            baryon_thirds: state.total_baryon_thirds() - reference.baryon_thirds,
            lepton: state.total_lepton() - reference.lepton,
        },
        energy_of: field.energy,
        rms_of: field.rms,
        beta_mean: if n > 0 { beta_sum / n as f64 } else { 0.0 },
        beta_max,
        gamma_max,
        half_count_radius: half_count_radius(state),
        decayed,
        annihilated,
        closest_pair_over_softening: closest,
        contact_fraction: contact,
        softening: field.effective_softening,
    }
}

/// Promień wokół środka geometrycznego, w którym mieści się połowa cząstek.
///
/// Liczony po **liczbie**, a nie po masie — inaczej jeden proton przeważyłby dwa
/// tysiące elektronów i wielkość mówiłaby o czymś innym, niż nazwa obiecuje.
fn half_count_radius(state: &State) -> f64 {
    let n = state.n();
    if n == 0 {
        return 0.0;
    }
    let centre: Vec3 = state.positions.iter().copied().sum::<Vec3>() / n as f64;
    let mut distances: Vec<f64> = state
        .positions
        .iter()
        .map(|p| (*p - centre).norm())
        .collect();
    distances.sort_by(f64::total_cmp);
    distances[n / 2]
}

/// Jak blisko siebie są cząstki, w jednostkach zmiękczenia.
///
/// Wielkość mierzy, na ile wynik zależy od `ε` — czyli od protezy za mechanikę
/// kwantową. Gdy `closest ≫ 1`, zmiękczenie nie robi nic i wynik jest własnością
/// oddziaływań. Gdy `closest < 1`, o sile decyduje `ε`, a nie fizyka, i trzeba to
/// wiedzieć **w trakcie** biegu, a nie po obejrzeniu wyniku.
///
/// Próbka, a nie wszystkie pary: koszt `O(512·N)` zamiast `O(N²)` przy tej samej
/// odpowiedzi co do rzędu wielkości, bo szukamy skrajności, a nie średniej.
fn neighbour_stats(state: &State, softening: f64, seed: u64) -> (f64, f64) {
    let n = state.n();
    if n < 2 || softening <= 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let mut rng = crate::rng::Rng::seeded(seed);
    let rows = rng.sample_indices(n, NEIGHBOUR_SAMPLE.min(n));

    let mut closest = f64::INFINITY;
    let mut touching = 0usize;
    for &i in &rows {
        let mut nearest = f64::INFINITY;
        for j in 0..n {
            if j == i {
                continue;
            }
            let d2 = (state.positions[i] - state.positions[j]).norm_squared();
            if d2 < nearest {
                nearest = d2;
            }
        }
        let nearest = nearest.sqrt();
        closest = closest.min(nearest);
        if nearest < CONTACT_SOFTENINGS * softening {
            touching += 1;
        }
    }
    (closest / softening, touching as f64 / rows.len() as f64)
}

impl Snapshot {
    pub fn energy_from(&self, interaction: Interaction) -> f64 {
        self.energy_of[interaction.index()]
    }

    pub fn rms_from(&self, interaction: Interaction) -> f64 {
        self.rms_of[interaction.index()]
    }

    /// Zastrzeżenie do wypisania w panelu i w biegu wsadowym.
    ///
    /// Kolejność jest kolejnością wagi: naruszenie liczby całkowitej unieważnia
    /// bieg, zależność od zmiękczenia unieważnia wynik ilościowy, a duży dryf
    /// energii znaczy tylko, że krok jest za duży.
    pub fn accuracy_hint(&self) -> Option<String> {
        if !self.integer_drift.is_zero() {
            return Some(format!(
                "naruszona wielkość zachowana: ładunek {:+}/3, liczba barionowa {:+}/3, \
                 liczba leptonowa {:+} — to usterka, nie przybliżenie",
                self.integer_drift.charge_thirds,
                self.integer_drift.baryon_thirds,
                self.integer_drift.lepton
            ));
        }
        if self.contact_fraction > 0.1 {
            return Some(format!(
                "{:.0}% cząstek stykało się bliżej niż 3ε — o sile decyduje zmiękczenie, \
                 a nie oddziaływanie; zmniejsz ε albo nie czytaj tego biegu ilościowo",
                100.0 * self.contact_fraction
            ));
        }
        if self.energy_drift.abs() > 1e-2 {
            return Some(format!(
                "dryf energii {:.1}% — skróć krok",
                100.0 * self.energy_drift
            ));
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::config::{ForceConfig, Ingredient, SpawnConfig};
    use crate::sm::forces::Forces;
    use crate::sm::particles::{Particle, Species};
    use crate::sm::{spawn, Config};
    use crate::sr::config::BackendKind;
    use crate::vec3::{vec3, ZERO};

    fn measured(state: &State, cfg: ForceConfig) -> Snapshot {
        let mut forces = Forces::new(cfg, state.n());
        let field = forces.compute(state);
        let reference = Reference::capture(state, &field);
        collect(state, &field, reference, 0.0, 0, 0, 1)
    }

    fn hydrogen_like() -> State {
        State::new(
            vec![ZERO, vec3(50_000.0, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![
                Particle::of(Species::Proton),
                Particle::of(Species::Electron),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_fresh_snapshot_has_no_drift_at_all() {
        let snapshot = measured(&hydrogen_like(), ForceConfig::default());
        assert_eq!(snapshot.energy_drift, 0.0);
        assert!(snapshot.integer_drift.is_zero());
        assert_eq!(snapshot.n, 2);
    }

    /// Energia całkowita to suma energii relatywistycznych i potencjału — czyli
    /// masa spoczynkowa też się w niej liczy. Bez tego rozpad wyglądałby na
    /// źródło energii.
    #[test]
    fn total_energy_includes_the_rest_mass() {
        let snapshot = measured(&hydrogen_like(), ForceConfig::default());
        let masses = Species::Proton.mass() + Species::Electron.mass();
        assert!((snapshot.rest_mass - masses).abs() < 1e-9);
        assert!(snapshot.total_energy > masses * 0.99);
        assert!(snapshot.kinetic.abs() < 1e-9, "cząstki spoczywają");
    }

    /// Liczby całkowite muszą być liczone dokładnie i pokazywane osobno dla
    /// każdego zapachu leptonowego.
    #[test]
    fn integer_charges_are_reported_per_flavour() {
        let state = State::new(
            vec![ZERO; 3],
            vec![ZERO; 3],
            vec![
                Particle::of(Species::Electron),
                Particle::of(Species::Muon),
                Particle::anti_of(Species::NeutrinoMu),
            ],
        )
        .unwrap();
        let snapshot = measured(&state, ForceConfig::default());
        assert_eq!(snapshot.charge_thirds, -6);
        assert_eq!(snapshot.lepton_electron, 1);
        assert_eq!(snapshot.lepton_muon, 0, "mion i antyneutrino się znoszą");
        assert_eq!(snapshot.lepton_tau, 0);
        assert_eq!(snapshot.lepton, 1);
    }

    /// Dryf odniesiony do energii CAŁKOWITEJ byłby w tym modelu bezużyteczny:
    /// masa spoczynkowa pary proton-elektron jest tysiące razy większa od energii
    /// ich oddziaływania, więc utrata całej fizyki pokazałaby się jako `10⁻⁹`.
    #[test]
    fn the_drift_is_measured_against_the_energy_that_can_change() {
        let state = hydrogen_like();
        let mut forces = Forces::new(ForceConfig::default(), state.n());
        let field = forces.compute(&state);
        let reference = Reference::capture(&state, &field);

        // Energia oddziaływania jest tu o rzędy wielkości mniejsza od mas.
        assert!(reference.dynamic_scale > 0.0);
        assert!(reference.dynamic_scale < reference.total_energy * 1e-6);

        // Zgubienie połowy energii oddziaływania to dryf rzędu 0,5 — a nie 10⁻⁹,
        // które wyszłoby z odniesienia do 939 MeV.
        let mut broken = reference;
        broken.total_energy += reference.dynamic_scale * 0.5;
        let snapshot = collect(&state, &field, broken, 0.0, 0, 0, 1);
        assert!(
            (snapshot.energy_drift.abs() - 0.5).abs() < 1e-9,
            "dryf {} zamiast 0,5",
            snapshot.energy_drift
        );
        assert!(snapshot.accuracy_hint().is_some());
    }

    /// Naruszenie wielkości zachowanej ma być nazwane **usterką**, a nie
    /// przybliżeniem. To jedyne zastrzeżenie, po którym wyniku nie da się uratować
    /// zmianą nastaw.
    #[test]
    fn a_broken_conservation_law_is_called_a_fault() {
        let state = hydrogen_like();
        let mut forces = Forces::new(ForceConfig::default(), state.n());
        let field = forces.compute(&state);
        let wrong = Reference {
            charge_thirds: 99,
            ..Reference::capture(&state, &field)
        };
        let snapshot = collect(&state, &field, wrong, 0.0, 0, 0, 1);

        assert!(!snapshot.integer_drift.is_zero());
        let hint = snapshot.accuracy_hint().expect("usterka ma być zgłoszona");
        assert!(hint.contains("usterka"), "{hint}");
    }

    /// Skok energii z narodzin jest ODEJMOWANY od dryfu. Bez tego jeden rozpad
    /// pary naładowanej zagłuszyłby pomiar jakości całkowania na resztę biegu.
    #[test]
    fn the_birth_jump_is_accounted_separately_from_the_drift() {
        let state = hydrogen_like();
        let mut forces = Forces::new(ForceConfig::default(), state.n());
        let field = forces.compute(&state);
        let mut reference = Reference::capture(&state, &field);
        // Udajemy, że energia całkowita podskoczyła o 100 MeV wskutek narodzin:
        // punkt odniesienia jest o tyle niższy od bieżącej energii.
        reference.total_energy -= 100.0;

        let blind = collect(&state, &field, reference, 0.0, 0, 0, 1);
        let accounted = collect(&state, &field, reference, 100.0, 0, 0, 1);

        assert!(blind.energy_drift.abs() > 0.05, "{}", blind.energy_drift);
        assert!(
            accounted.energy_drift.abs() < 1e-12,
            "skok nie został odjęty: {}",
            accounted.energy_drift
        );
    }

    /// Zależność od zmiękczenia musi być widoczna jako liczba. Cząstki wciśnięte
    /// w jeden punkt dają ostrzeżenie, rozrzucone — nie.
    #[test]
    fn softening_dependence_is_measured_and_announced() {
        let cfg = ForceConfig {
            softening: 1.0,
            backend: BackendKind::Exact,
            ..ForceConfig::default()
        };
        let electron = Particle::of(Species::Electron);

        let crowded = State::new(
            (0..20).map(|i| vec3(i as f64 * 0.05, 0.0, 0.0)).collect(),
            vec![ZERO; 20],
            vec![electron; 20],
        )
        .unwrap();
        let snapshot = measured(&crowded, cfg);
        assert!(snapshot.closest_pair_over_softening < 1.0);
        assert!(snapshot.contact_fraction > 0.9);
        let hint = snapshot.accuracy_hint().expect("stłoczenie ma być zgłoszone");
        assert!(hint.contains("zmiękczenie"), "{hint}");

        let spread = State::new(
            (0..20).map(|i| vec3(i as f64 * 100.0, 0.0, 0.0)).collect(),
            vec![ZERO; 20],
            vec![electron; 20],
        )
        .unwrap();
        let snapshot = measured(&spread, cfg);
        assert!(snapshot.closest_pair_over_softening > 50.0);
        assert_eq!(snapshot.contact_fraction, 0.0);
        assert!(snapshot.accuracy_hint().is_none());
    }

    /// `γ` istnieje tylko dla cząstek masywnych. Chmura samych fotonów nie może
    /// zgłosić „γ = ∞" ani „γ = 1" — obie odpowiedzi byłyby nieprawdziwe.
    #[test]
    fn gamma_is_absent_when_every_particle_is_massless() {
        let photons = State::new(
            vec![ZERO, vec3(10.0, 0.0, 0.0)],
            vec![vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0)],
            vec![Particle::of(Species::Photon); 2],
        )
        .unwrap();
        let snapshot = measured(&photons, ForceConfig::default());
        assert!(snapshot.gamma_max.is_none());
        assert!((snapshot.beta_mean - 1.0).abs() < 1e-12);
        assert!((snapshot.beta_max - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gamma_is_reported_when_a_massive_particle_is_present() {
        let mixed = State::new(
            vec![ZERO, vec3(10.0, 0.0, 0.0)],
            vec![vec3(1.0, 0.0, 0.0), vec3(1_000.0, 0.0, 0.0)],
            vec![
                Particle::of(Species::Photon),
                Particle::of(Species::Electron),
            ],
        )
        .unwrap();
        let snapshot = measured(&mixed, ForceConfig::default());
        let gamma = snapshot.gamma_max.expect("elektron ma γ");
        assert!((gamma - 1_000.0 / Species::Electron.mass()).abs() / gamma < 1e-3);
    }

    /// Promień połowy liczy CZĄSTKI, nie masę — inaczej jeden proton przeważyłby
    /// tysiąc elektronów i wielkość mówiłaby o czym innym niż jej nazwa.
    #[test]
    fn the_half_radius_counts_particles_not_mass() {
        let mut positions = vec![ZERO; 999];
        positions.push(vec3(1e6, 0.0, 0.0));
        let mut kinds = vec![Particle::of(Species::Electron); 999];
        kinds.push(Particle::of(Species::Proton));
        let state = State::new(positions, vec![ZERO; 1000], kinds).unwrap();

        let snapshot = measured(&state, ForceConfig::default());
        // Środek geometryczny jest o `1e6/1000 = 1000` fm od gromady — to cena
        // liczenia po liczbie, nie po masie. Po masie środek wylądowałby przy
        // protonie (`~6·10⁵ fm`) i „promień połowy" opisywałby jego odsunięcie.
        assert!(
            snapshot.half_count_radius <= 1e3,
            "{}",
            snapshot.half_count_radius
        );
    }

    #[test]
    fn an_empty_state_does_not_panic() {
        let snapshot = measured(&State::empty(), ForceConfig::default());
        assert_eq!(snapshot.n, 0);
        assert_eq!(snapshot.half_count_radius, 0.0);
        assert_eq!(snapshot.momentum_residual, 0.0);
        assert_eq!(snapshot.beta_mean, 0.0);
    }

    /// Udziały poszczególnych oddziaływań muszą trafiać do właściwych kolumn.
    #[test]
    fn per_interaction_columns_line_up() {
        let mut cfg = Config::default();
        cfg.spawn = SpawnConfig {
            mixture: vec![
                Ingredient::new(Particle::of(Species::Up), 4),
                Ingredient::new(Particle::anti_of(Species::Up), 4),
            ],
            radius: 2.0,
            ..cfg.spawn
        };
        let state = spawn::make_state(&cfg).state;
        let snapshot = measured(
            &state,
            ForceConfig {
                coulomb: true,
                strong: true,
                gravity: true,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
        );

        assert!(snapshot.rms_from(Interaction::Strong) > 0.0);
        assert!(snapshot.rms_from(Interaction::Electromagnetic) > 0.0);
        assert!(snapshot.rms_from(Interaction::Gravitational) > 0.0);
        assert_eq!(snapshot.rms_from(Interaction::Weak), 0.0);
        // Silne musi tu przeważać nad elektromagnetycznym — to jest cała treść
        // słowa „silne".
        assert!(
            snapshot.rms_from(Interaction::Strong)
                > snapshot.rms_from(Interaction::Electromagnetic)
        );
    }
}
