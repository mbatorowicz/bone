//! Stan gazu cząstek: położenia, pędy i **tożsamość** każdej cząstki.
//!
//! Różnica wobec [`crate::sr::state::State`] jest jedna, ale zasadnicza: tam liczba
//! cząstek jest ustalona na starcie i nigdy się nie zmienia, tu rozpad usuwa jedną
//! cząstkę i wstawia dwie albo trzy. Dlatego stan musi umieć się przebudować,
//! a wszystko, co go opisuje, musi przeżyć tę przebudowę.
//!
//! # Masa i ładunek są w pamięci podręcznej
//!
//! `masses` i `charges` da się w każdej chwili odtworzyć z `kinds`, więc trzymanie
//! ich obok wygląda na zbędne powielenie. Nie jest: solver sił bierze `&[f64]`
//! i wywołuje się raz na krok, więc bez tych tablic każdy krok alokowałby dwie nowe.
//! Spójność jest pilnowana jednym miejscem — [`State::rebuild_cache`] wołanym po
//! każdej zmianie składu — i sprawdzana testem, a nie umową.

use crate::sm::kinematics::{self, FourMomentum};
use crate::sm::particles::{Flavour, Particle};
use crate::vec3::Vec3;

/// Cząstka wyprodukowana w rozpadzie albo anihilacji.
#[derive(Clone, Copy, Debug)]
pub struct Birth {
    pub particle: Particle,
    pub position: Vec3,
    pub momentum: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    ShapeMismatch {
        positions: usize,
        momenta: usize,
        kinds: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch {
                positions,
                momenta,
                kinds,
            } => write!(
                f,
                "położenia ({positions}), pędy ({momenta}) i rodzaje ({kinds}) \
                 muszą mieć tę samą długość"
            ),
        }
    }
}

impl std::error::Error for StateError {}

#[derive(Clone, Debug)]
pub struct State {
    pub positions: Vec<Vec3>,
    /// Pęd w MeV/c. Zmienną stanu jest pęd, nie prędkość — z tego samego powodu
    /// co w modelu SR i dodatkowo dlatego, że cząstka bezmasowa prędkości jako
    /// zmiennej stanu mieć nie może (ma zawsze `c`, niezależnie od energii).
    pub momenta: Vec<Vec3>,
    pub kinds: Vec<Particle>,
    /// Czas w fm/c.
    pub time: f64,
    pub step: u64,
    masses: Vec<f64>,
    charges: Vec<f64>,
}

impl State {
    /// # Errors
    /// Gdy trzy tablice mają różne długości.
    pub fn new(
        positions: Vec<Vec3>,
        momenta: Vec<Vec3>,
        kinds: Vec<Particle>,
    ) -> Result<Self, StateError> {
        if positions.len() != momenta.len() || positions.len() != kinds.len() {
            return Err(StateError::ShapeMismatch {
                positions: positions.len(),
                momenta: momenta.len(),
                kinds: kinds.len(),
            });
        }
        let mut state = Self {
            positions,
            momenta,
            kinds,
            time: 0.0,
            step: 0,
            masses: Vec::new(),
            charges: Vec::new(),
        };
        state.rebuild_cache();
        Ok(state)
    }

    pub fn empty() -> Self {
        Self::new(Vec::new(), Vec::new(), Vec::new()).expect("trzy puste tablice pasują")
    }

    pub fn n(&self) -> usize {
        self.kinds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kinds.is_empty()
    }

    /// Masy spoczynkowe w MeV — gotowa tablica dla kinematyki.
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Ładunki w jednostkach `e` — gotowa tablica dla solvera Coulomba.
    pub fn charges(&self) -> &[f64] {
        &self.charges
    }

    pub fn energy(&self, index: usize) -> f64 {
        kinematics::energy(self.masses[index], self.momenta[index])
    }

    pub fn kinetic_energy(&self, index: usize) -> f64 {
        kinematics::kinetic_energy(self.masses[index], self.momenta[index])
    }

    pub fn velocity(&self, index: usize) -> Vec3 {
        kinematics::velocity(self.masses[index], self.momenta[index])
    }

    pub fn beta(&self, index: usize) -> f64 {
        kinematics::beta(self.masses[index], self.momenta[index])
    }

    pub fn four_momentum(&self, index: usize) -> FourMomentum {
        FourMomentum::new(self.energy(index), self.momenta[index])
    }

    /// Sumaryczny czteropęd. Zachowywany dokładnie przez rozpady i przybliżenie
    /// przez całkowanie — dlatego jest mierzony, a nie zakładany.
    pub fn total_four_momentum(&self) -> FourMomentum {
        (0..self.n()).map(|i| self.four_momentum(i)).sum()
    }

    /// Ładunek całkowity w trzecich `e`. Liczba **całkowita**: po poprawnym rozpadzie
    /// musi się nie zmienić co do bitu, a nie „w granicach błędu".
    pub fn total_charge_thirds(&self) -> i64 {
        self.kinds.iter().map(|p| p.charge_thirds() as i64).sum()
    }

    pub fn total_baryon_thirds(&self) -> i64 {
        self.kinds.iter().map(|p| p.baryon_thirds() as i64).sum()
    }

    pub fn total_lepton(&self) -> i64 {
        self.kinds.iter().map(|p| p.lepton() as i64).sum()
    }

    /// Liczba leptonowa jednego zapachu. Model Standardowy (bez mas neutrin)
    /// zachowuje każdą z trzech osobno, więc osobno są sprawdzane.
    pub fn lepton_of(&self, flavour: Flavour) -> i64 {
        self.kinds.iter().map(|p| p.lepton_of(flavour) as i64).sum()
    }

    /// Ile cząstek danego rodzaju.
    pub fn count_of(&self, particle: Particle) -> usize {
        self.kinds.iter().filter(|p| **p == particle).count()
    }

    /// Skład posortowany malejąco po liczebności — do panelu i do wydruku.
    pub fn census(&self) -> Vec<(Particle, usize)> {
        let mut tally: Vec<(Particle, usize)> = Vec::new();
        for kind in &self.kinds {
            match tally.iter_mut().find(|(p, _)| p == kind) {
                Some((_, count)) => *count += 1,
                None => tally.push((*kind, 1)),
            }
        }
        tally.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.id().cmp(&b.0.id())));
        tally
    }

    /// Dopisz cząstkę na koniec.
    pub fn push(&mut self, birth: Birth) {
        self.positions.push(birth.position);
        self.momenta.push(birth.momentum);
        self.kinds.push(birth.particle);
        self.masses.push(birth.particle.mass());
        self.charges.push(birth.particle.charge());
    }

    /// Usuń cząstki spod podanych indeksów i dopisz nowe — jedną operacją.
    ///
    /// Rozdzielenie tego na „usuń" i „dodaj" byłoby wygodniejsze w zapisie
    /// i błędne w użyciu: usunięcie przesuwa indeksy, więc druga operacja
    /// działałaby na innych cząstkach niż zamierzone. Tutaj `removed` odnosi się
    /// zawsze do stanu sprzed wywołania.
    ///
    /// Zwraca liczbę usuniętych cząstek.
    pub fn replace(&mut self, removed: &[bool], born: &[Birth]) -> usize {
        debug_assert_eq!(removed.len(), self.n(), "maska nie pasuje do stanu");
        let gone = removed.iter().filter(|r| **r).count();
        if gone > 0 {
            // `Vec::retain` woła domknięcie dokładnie raz na element i w kolejności,
            // więc licznik odtwarza indeks w tablicy sprzed usunięcia. Osobny licznik
            // dla każdej tablicy, bo każda przechodzi własny przebieg.
            let mut index = 0;
            self.positions.retain(|_| {
                index += 1;
                !removed[index - 1]
            });
            let mut index = 0;
            self.momenta.retain(|_| {
                index += 1;
                !removed[index - 1]
            });
            let mut index = 0;
            self.kinds.retain(|_| {
                index += 1;
                !removed[index - 1]
            });
        }
        for birth in born {
            self.positions.push(birth.position);
            self.momenta.push(birth.momentum);
            self.kinds.push(birth.particle);
        }
        if gone > 0 || !born.is_empty() {
            self.rebuild_cache();
        }
        gone
    }

    /// Odtwórz tablice mas i ładunków z rodzajów cząstek.
    pub fn rebuild_cache(&mut self) {
        self.masses.clear();
        self.charges.clear();
        self.masses.extend(self.kinds.iter().map(|p| p.mass()));
        self.charges.extend(self.kinds.iter().map(|p| p.charge()));
    }

    /// Czy stan jest liczbowo zdrowy. Wywoływane raz na krok — `NaN` w położeniu
    /// rozlewa się na resztę chmury przez solver siatkowy, więc im wcześniej
    /// zostanie zauważony, tym mniej niesie ze sobą pytań.
    pub fn is_finite(&self) -> bool {
        self.positions.iter().all(|p| p.is_finite()) && self.momenta.iter().all(|p| p.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::particles::Species;
    use crate::vec3::{vec3, ZERO};

    fn sample() -> State {
        State::new(
            vec![vec3(1.0, 0.0, 0.0), vec3(-1.0, 0.0, 0.0), ZERO],
            vec![vec3(0.0, 2.0, 0.0), vec3(0.0, -2.0, 0.0), vec3(5.0, 0.0, 0.0)],
            vec![
                Particle::of(Species::Electron),
                Particle::anti_of(Species::Electron),
                Particle::of(Species::Photon),
            ],
        )
        .unwrap()
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        let err = State::new(vec![ZERO], Vec::new(), vec![Particle::of(Species::Photon)]);
        assert!(matches!(err, Err(StateError::ShapeMismatch { .. })));
    }

    /// Tablice podręczne muszą zgadzać się z rodzajami cząstek — po każdej zmianie
    /// składu. To jedyne miejsce, w którym powielenie danych może się rozjechać.
    fn assert_cache_is_consistent(state: &State) {
        assert_eq!(state.masses().len(), state.n());
        assert_eq!(state.charges().len(), state.n());
        for (i, kind) in state.kinds.iter().enumerate() {
            assert_eq!(state.masses()[i], kind.mass(), "masa cząstki {i}");
            assert_eq!(state.charges()[i], kind.charge(), "ładunek cząstki {i}");
        }
    }

    #[test]
    fn cache_matches_the_particle_kinds() {
        assert_cache_is_consistent(&sample());
    }

    #[test]
    fn conserved_numbers_are_exact_integers() {
        let state = sample();
        // e⁻ i e⁺ znoszą się w ładunku i w liczbie leptonowej, foton nie wnosi nic.
        assert_eq!(state.total_charge_thirds(), 0);
        assert_eq!(state.total_lepton(), 0);
        assert_eq!(state.lepton_of(Flavour::Electron), 0);
        assert_eq!(state.total_baryon_thirds(), 0);
    }

    #[test]
    fn replacing_particles_keeps_the_cache_and_the_bookkeeping() {
        let mut state = sample();
        // Anihilacja: znika para e⁺e⁻, pojawiają się dwa fotony.
        let removed = vec![true, true, false];
        let born = vec![
            Birth {
                particle: Particle::of(Species::Photon),
                position: ZERO,
                momentum: vec3(1.0, 0.0, 0.0),
            },
            Birth {
                particle: Particle::of(Species::Photon),
                position: ZERO,
                momentum: vec3(-1.0, 0.0, 0.0),
            },
        ];
        let gone = state.replace(&removed, &born);

        assert_eq!(gone, 2);
        assert_eq!(state.n(), 3);
        assert_cache_is_consistent(&state);
        assert_eq!(state.count_of(Particle::of(Species::Photon)), 3);
        assert_eq!(state.total_charge_thirds(), 0);
        assert_eq!(state.total_lepton(), 0);
    }

    /// Maska odnosi się do stanu SPRZED wywołania. Gdyby usuwanie i dodawanie
    /// odbywało się osobno, druga operacja trafiałaby w przesunięte indeksy —
    /// ten test pilnuje, że cząstka, która miała zostać, faktycznie została.
    #[test]
    fn the_mask_refers_to_the_state_before_the_call() {
        let mut state = sample();
        let survivor = state.kinds[2];
        let survivor_position = state.positions[2];
        state.replace(
            &[true, false, false],
            &[Birth {
                particle: Particle::of(Species::Muon),
                position: vec3(9.0, 9.0, 9.0),
                momentum: ZERO,
            }],
        );
        assert_eq!(state.kinds[1], survivor);
        assert_eq!(state.positions[1], survivor_position);
        assert_eq!(state.kinds[2], Particle::of(Species::Muon));
        assert_cache_is_consistent(&state);
    }

    #[test]
    fn replacing_nothing_changes_nothing() {
        let mut state = sample();
        let before = state.clone();
        assert_eq!(state.replace(&[false, false, false], &[]), 0);
        assert_eq!(state.n(), before.n());
        assert_eq!(state.kinds, before.kinds);
        assert_cache_is_consistent(&state);
    }

    #[test]
    fn removing_everything_leaves_an_empty_but_valid_state() {
        let mut state = sample();
        state.replace(&[true, true, true], &[]);
        assert!(state.is_empty());
        assert_eq!(state.total_charge_thirds(), 0);
        assert_cache_is_consistent(&state);
        assert!(state.is_finite());
    }

    #[test]
    fn pushing_keeps_the_cache_consistent() {
        let mut state = State::empty();
        state.push(Birth {
            particle: Particle::of(Species::Proton),
            position: ZERO,
            momentum: vec3(0.0, 0.0, 100.0),
        });
        assert_eq!(state.n(), 1);
        assert_eq!(state.total_charge_thirds(), 3);
        assert_cache_is_consistent(&state);
    }

    /// Foton porusza się dokładnie z `c`, elektron wolniej — mimo że oba mają tu
    /// pęd tego samego rzędu.
    #[test]
    fn massless_and_massive_particles_move_differently() {
        let state = sample();
        assert!((state.beta(2) - 1.0).abs() < 1e-15, "foton nie leci z c");
        assert!(state.beta(0) < 1.0, "elektron osiągnął c");
        assert_eq!(state.energy(2), 5.0, "energia fotonu to |p|");
    }

    #[test]
    fn census_counts_every_kind_once() {
        let state = sample();
        let census = state.census();
        assert_eq!(census.len(), 3);
        assert_eq!(census.iter().map(|(_, n)| n).sum::<usize>(), state.n());

        let mut many = state;
        for _ in 0..5 {
            many.push(Birth {
                particle: Particle::of(Species::Photon),
                position: ZERO,
                momentum: vec3(1.0, 0.0, 0.0),
            });
        }
        // Najliczniejszy rodzaj musi trafić na początek listy.
        assert_eq!(many.census()[0], (Particle::of(Species::Photon), 6));
    }

    #[test]
    fn total_four_momentum_adds_up() {
        let state = sample();
        let total = state.total_four_momentum();
        let by_hand: f64 = (0..state.n()).map(|i| state.energy(i)).sum();
        assert!((total.energy - by_hand).abs() < 1e-12);
        // Pędy elektronu i pozytonu znoszą się, zostaje foton.
        assert!((total.momentum - vec3(5.0, 0.0, 0.0)).norm() < 1e-12);
    }

    #[test]
    fn non_finite_state_is_detected() {
        let mut state = sample();
        assert!(state.is_finite());
        state.momenta[1] = vec3(f64::NAN, 0.0, 0.0);
        assert!(!state.is_finite());
    }
}
