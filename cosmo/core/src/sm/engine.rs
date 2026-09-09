//! Silnik: całkowanie po pędzie plus warstwa stochastyczna nad nim.
//!
//! # Krok
//!
//! Leapfrog KDK na **pędzie**, tak samo jak w modelu SR i z tego samego powodu:
//! `dp/dt = F` jest równaniem, które faktycznie całkujemy, a prędkość wynika z pędu.
//! Tu dochodzi drugi powód — cząstka bezmasowa nie ma prędkości jako zmiennej
//! stanu, bo jej prędkość jest zawsze `c` niezależnie od energii. Foton wyprodukowany
//! w anihilacji musiałby wtedy dostać „prędkość", która nie niesie żadnej informacji.
//!
//! ```text
//! p ← p + F·dt/2        x ← x + v(p)·dt        p ← p + F·dt/2
//! ```
//!
//! Po pełnym kroku dochodzi losowanie rozpadów. Kolejność nie jest dowolna: rozpad
//! zmienia skład, więc pole sił policzone przed nim przestaje obowiązywać i jest
//! przeliczane — patrz [`Engine::step`].
//!
//! # Dobór kroku
//!
//! Model SR ogranicza krok przez `η√(ε/a_max)`, gdzie `a` jest przyspieszeniem.
//! Tutaj to nie działa, bo `a = F/m` nie istnieje dla cząstki bezmasowej. Oba
//! warunki są więc wyrażone tak, żeby przejść granicę `m → 0` bez dzielenia przez
//! zero — patrz [`Engine::choose_dt`].

use crate::sm::config::Config;
use crate::sm::decays::{Decays, Tally};
use crate::sm::diagnostics::{self, Reference, Snapshot};
use crate::sm::forces::{FieldSet, Forces};
use crate::sm::kinematics;
use crate::sm::spawn;
use crate::sm::state::State;

/// Ułamek skali, o który wolno posunąć się w jednym kroku.
const COURANT: f64 = 0.1;

#[derive(Clone, Copy, Debug)]
pub struct Diverged {
    pub step: u64,
}

impl std::fmt::Display for Diverged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stan przestał być skończony na kroku {}", self.step)
    }
}

impl std::error::Error for Diverged {}

pub struct Engine {
    pub cfg: Config,
    pub state: State,
    forces: Forces,
    decays: Decays,
    field: FieldSet,
    reference: Reference,
    /// Suma skoków energii potencjalnej z narodzin cząstek — patrz
    /// [`crate::sm::diagnostics`].
    transmutation_energy: f64,
    warnings: Vec<String>,
    last_dt: f64,
}

impl Engine {
    pub fn new(cfg: Config) -> Self {
        let spawned = spawn::make_state(&cfg);
        let mut warnings = cfg.warnings();
        warnings.extend(spawned.warnings);
        Self::assemble(cfg, spawned.state, warnings)
    }

    /// Silnik na gotowym stanie — po wznowieniu z checkpointu.
    ///
    /// Punkt odniesienia dla dryfu jest zapamiętywany **na nowo**, a nie wczytywany.
    /// Dryf mierzy jakość całkowania od chwili, w której zaczęliśmy liczyć; udawanie,
    /// że pamiętamy energię sprzed przerwy, przypisywałoby wznowionemu biegowi błąd
    /// biegu poprzedniego. Liczby całkowite są odtwarzane ze stanu, więc ich
    /// zachowanie jest pilnowane dalej — bez przerwy.
    pub fn with_state(cfg: Config, state: State, warnings: Vec<String>) -> Self {
        Self::assemble(cfg, state, warnings)
    }

    fn assemble(cfg: Config, state: State, warnings: Vec<String>) -> Self {
        let mut forces = Forces::new(cfg.forces, state.n());
        let field = forces.compute(&state);
        let reference = Reference::capture(&state, &field);
        Self {
            decays: Decays::new(cfg.decay),
            forces,
            field,
            reference,
            transmutation_energy: 0.0,
            warnings,
            last_dt: cfg.run.dt,
            cfg,
            state,
        }
    }

    pub fn n(&self) -> usize {
        self.state.n()
    }

    pub fn tally(&self) -> Tally {
        self.decays.tally()
    }

    pub fn last_dt(&self) -> f64 {
        self.last_dt
    }

    /// Jeden krok. Zwraca użyty `dt` w fm/c.
    pub fn step(&mut self) -> f64 {
        let dt = self.choose_dt();
        self.last_dt = dt;

        if self.state.is_empty() {
            self.state.time += dt;
            self.state.step += 1;
            return dt;
        }

        self.kick(dt / 2.0);
        self.drift(dt);
        self.field = self.forces.compute(&self.state);
        self.kick(dt / 2.0);

        self.state.time += dt;
        self.state.step += 1;

        // Rozpady dopiero po pełnym kroku: gdyby zaszły w środku, druga połowa
        // kopnięcia działałaby na cząstki, których w chwili liczenia siły nie było.
        let potential_before = self.field.total_potential_energy();
        let outcome = self
            .decays
            .step(&mut self.state, dt, self.cfg.run.max_particles);
        if outcome.happened() {
            self.field = self.forces.compute(&self.state);
            self.transmutation_energy += self.field.total_potential_energy() - potential_before;
            if outcome.capped {
                self.warnings.push(format!(
                    "limit {} cząstek powstrzymał rozpady — zwiększ limit albo skróć bieg",
                    self.cfg.run.max_particles
                ));
            }
        }

        dt
    }

    /// # Errors
    /// Gdy stan przestanie być skończony.
    pub fn advance(&mut self, steps: u32) -> Result<f64, Diverged> {
        let mut elapsed = 0.0;
        for _ in 0..steps.max(1) {
            elapsed += self.step();
            if !self.state.is_finite() {
                return Err(Diverged {
                    step: self.state.step,
                });
            }
        }
        Ok(elapsed)
    }

    fn kick(&mut self, half: f64) {
        for (momentum, force) in self.state.momenta.iter_mut().zip(self.field.force.iter()) {
            *momentum += *force * half;
        }
    }

    fn drift(&mut self, dt: f64) {
        for i in 0..self.state.n() {
            let velocity = kinematics::velocity(self.state.masses()[i], self.state.momenta[i]);
            self.state.positions[i] += velocity * dt;
        }
    }

    /// Największy krok, przy którym oba warunki dokładności są spełnione.
    ///
    /// # Warunek przesunięcia
    ///
    /// Żadna cząstka nie może przebyć w jednym kroku więcej niż `η·ε`, bo `ε` jest
    /// skalą, na której zmienia się siła — przeskoczenie jej znaczy minięcie
    /// najsilniejszego odcinka oddziaływania bez poczucia go.
    ///
    /// Naiwne `dt ≤ η·ε/β` zawodzi w obie strony: dla cząstki spoczywającej dzieli
    /// przez zero, a dla zimnej chmury daje krok tysiące razy za duży, bo pomija
    /// prędkość, którą cząstka **nabierze** w tym kroku. Bierzemy więc pęd po
    /// kopnięciu, `|p| + |F|·dt`, i korzystamy z tego, że `β ≤ min(1, p/m)`:
    ///
    /// ```text
    /// (|p| + |F|·dt)/m · dt ≤ η·ε      →  |F|dt² + |p|dt − η·ε·m ≤ 0
    /// ```
    ///
    /// Kwadratowa nierówność ma rozwiązanie w postaci zamkniętej, więc nie ma tu
    /// żadnej iteracji ani zgadywania. Drugie ograniczenie, `β ≤ 1`, daje `dt ≤ η·ε`
    /// i obowiązuje zawsze — w szczególności jest **jedynym** warunkiem dla cząstki
    /// bezmasowej. Ponieważ wystarczy, by spełniony był którykolwiek z dwóch
    /// kresów, dla każdej cząstki bierzemy ten **łagodniejszy**.
    ///
    /// # Warunek zmiany pędu
    ///
    /// `|F|·dt ≤ η·max(|p|, m)`: pęd nie może w jednym kroku zmienić się o więcej
    /// niż ułamek własnej skali cząstki. Dla cząstki szybkiej skalą jest `|p|`
    /// (chodzi o kąt, o jaki skręci tor), dla spoczywającej `m` (chodzi o to, żeby
    /// nie stała się relatywistyczna w jednym kroku). **Skala jest liczona osobno
    /// dla każdej cząstki**, a nie ze średniej: średnia w gazie proton-elektron
    /// jest zdominowana przez masę protonu i pozwoliłaby elektronowi na krok tysiąc
    /// razy za duży.
    ///
    /// # Wyłączenie
    ///
    /// Adaptację da się wyłączyć i przy biegach nastawionych na rozpady **trzeba** —
    /// krok rzędu czasu życia mionu jest siedemnaście rzędów wielkości większy od
    /// skali sił. To nie jest obejście: gdy bada się rozpad swobodnej wiązki, sił
    /// po prostu nie ma i nie ma czego rozdzielać.
    fn choose_dt(&self) -> f64 {
        let requested = self.cfg.run.dt.max(f64::MIN_POSITIVE);
        if !self.cfg.run.adaptive {
            return requested;
        }

        let softening = self.field.effective_softening;
        let mut dt = requested;
        for i in 0..self.state.n() {
            let force = self.field.force[i].norm();
            let momentum = self.state.momenta[i].norm();
            let mass = self.state.masses()[i];

            if force > 0.0 {
                dt = dt.min(COURANT * momentum.max(mass) / force);
            }
            if softening > 0.0 {
                dt = dt.min(displacement_limit(softening, mass, momentum, force));
            }
        }
        dt
    }

    pub fn collect_diagnostics(&self) -> Snapshot {
        let tally = self.decays.tally();
        diagnostics::collect(
            &self.state,
            &self.field,
            self.reference,
            self.transmutation_energy,
            tally.decayed,
            tally.annihilated,
            self.cfg.decay.seed,
        )
    }

    pub fn accuracy_hint(&self) -> Option<String> {
        self.collect_diagnostics().accuracy_hint()
    }

    pub fn describe(&self) -> String {
        let tally = self.decays.tally();
        format!(
            "N={}  {}  rozpadów {}  anihilacji {}",
            self.state.n(),
            self.forces.describe(),
            tally.decayed,
            tally.annihilated
        )
    }

    /// Skład układu, od najliczniejszego rodzaju.
    pub fn census_text(&self, limit: usize) -> String {
        let census = self.state.census();
        let shown: Vec<String> = census
            .iter()
            .take(limit)
            .map(|(particle, count)| format!("{particle}×{count}"))
            .collect();
        if census.len() > limit {
            return format!("{}  (+{} rodzajów)", shown.join("  "), census.len() - limit);
        }
        shown.join("  ")
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Odcień punktu dla renderera: `β = v/c`.
    ///
    /// Ten sam wybór co w modelu SR i z tym samym skutkiem — tyle że tutaj rozdziela
    /// dodatkowo rodzaje cząstek. Elektron i proton o tej samej energii termicznej
    /// mają prędkości różniące się o czynnik `√(m_p/m_e) ≈ 43`, a foton z anihilacji
    /// leci dokładnie z `c` i jest najjaśniejszym punktem w kadrze.
    pub fn shade(&self, index: usize) -> f32 {
        self.state.beta(index) as f32
    }

    /// Podmień nastawy, które wolno zmieniać w trakcie biegu.
    ///
    /// Skład początkowy i zarodek losowy są celowo pominięte: jedno i drugie
    /// obowiązuje w chwili startu, a podmiana ich w połowie biegu znaczyłaby, że
    /// zapisany `config.json` opisuje coś innego niż to, co zostało policzone.
    pub fn apply_runtime_config(&mut self, live: &Config) {
        self.cfg.run = live.run;
        self.cfg.forces = live.forces;
        self.cfg.decay.enabled = live.decay.enabled;
        self.cfg.decay.annihilation = live.decay.annihilation;
        self.cfg.decay.pair_radius = live.decay.pair_radius;

        self.forces.apply_runtime(&live.forces, self.state.n());
        self.decays.apply_runtime(&live.decay);
        self.field = self.forces.compute(&self.state);
    }
}

/// Krok, po którym cząstka na pewno nie przebędzie więcej niż `η·ε`.
///
/// Zwraca łagodniejszy z dwóch kresów opisanych przy [`Engine::choose_dt`]: tego
/// z `β ≤ 1` i tego z `β ≤ p/m`. Dla cząstki bezmasowej drugi nie istnieje
/// i zostaje pierwszy.
fn displacement_limit(softening: f64, mass: f64, momentum: f64, force: f64) -> f64 {
    let budget = COURANT * softening;
    if mass <= 0.0 {
        return budget;
    }
    // |F|dt² + |p|dt − η·ε·m = 0
    let from_mass = if force > 0.0 {
        let discriminant = momentum * momentum + 4.0 * force * budget * mass;
        (discriminant.sqrt() - momentum) / (2.0 * force)
    } else if momentum > 0.0 {
        budget * mass / momentum
    } else {
        // Ani pędu, ani siły: cząstka stoi i nic jej nie ruszy w tym kroku.
        f64::INFINITY
    };
    budget.max(from_mass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::config::{DecayConfig, ForceConfig, Ingredient, RunConfig, Shape, SpawnConfig};
    use crate::sm::particles::{Particle, Species};
    use crate::sr::config::BackendKind;

    fn two_body(a: Particle, b: Particle, separation: f64) -> Config {
        Config {
            spawn: SpawnConfig {
                shape: Shape::Pair,
                mixture: vec![Ingredient::new(a, 1), Ingredient::new(b, 1)],
                radius: separation / 2.0,
                temperature: 0.0,
                beam_energy: 0.0,
                seed: 5,
            },
            forces: ForceConfig {
                coulomb: true,
                softening: separation / 100.0,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
            decay: DecayConfig {
                enabled: false,
                annihilation: false,
                ..DecayConfig::default()
            },
            run: RunConfig {
                dt: separation / 100.0,
                adaptive: true,
                ..RunConfig::default()
            },
        }
    }

    /// Sedno całkowania: energia ma oscylować wokół zera, a nie odpływać.
    /// Mierzone na parze proton-elektron, czyli najostrzejszym przypadku
    /// dostępnym w tym modelu (stosunek mas 1836).
    #[test]
    fn energy_drift_stays_small_and_does_not_grow() {
        let cfg = two_body(
            Particle::of(Species::Proton),
            Particle::of(Species::Electron),
            5.0e4,
        );
        let mut engine = Engine::new(cfg);
        let mut worst: f64 = 0.0;
        for _ in 0..400 {
            engine.step();
            worst = worst.max(engine.collect_diagnostics().energy_drift.abs());
        }
        assert!(worst < 1e-6, "dryf energii sięgnął {worst:e}");
    }

    /// Dwa ładunki jednoimienne muszą się rozjechać, a przeciwne — zbliżyć.
    /// Najprostszy sprawdzian, że siła w ogóle trafia do pędu z właściwym znakiem.
    #[test]
    fn charges_move_the_way_the_force_points() {
        let separation = 1.0e4;
        let apart = |a: Particle, b: Particle| {
            let mut engine = Engine::new(two_body(a, b, separation));
            let before = (engine.state.positions[0] - engine.state.positions[1]).norm();
            engine.advance(300).unwrap();
            (engine.state.positions[0] - engine.state.positions[1]).norm() - before
        };

        let electron = Particle::of(Species::Electron);
        assert!(apart(electron, electron) > 0.0, "elektrony się zbliżyły");
        assert!(
            apart(electron, Particle::of(Species::Proton)) < 0.0,
            "elektron i proton się rozjechali"
        );
    }

    /// Foton porusza się dokładnie z `c` i żadna siła tego nie zmieni — bo foton
    /// nie ma ładunku ani masy. Test pilnuje, że całkowanie nie zepsuło
    /// tożsamości `|v| = c` przez podstawienie masy zastępczej.
    #[test]
    fn photons_travel_at_c_through_the_integrator() {
        let cfg = Config {
            spawn: SpawnConfig {
                shape: Shape::Beam,
                mixture: vec![Ingredient::new(Particle::of(Species::Photon), 20)],
                radius: 10.0,
                temperature: 0.0,
                beam_energy: 500.0,
                seed: 3,
            },
            run: RunConfig {
                dt: 1.0,
                adaptive: false,
                ..RunConfig::default()
            },
            ..Config::default()
        };
        let mut engine = Engine::new(cfg);
        let before: Vec<f64> = (0..engine.n()).map(|i| engine.state.positions[i].z).collect();
        let elapsed = engine.advance(50).unwrap();
        for (i, start) in before.iter().enumerate() {
            let travelled = engine.state.positions[i].z - start;
            assert!(
                (travelled - elapsed).abs() < 1e-9,
                "foton {i} przebył {travelled} fm w czasie {elapsed} fm/c"
            );
        }
    }

    /// Kaskada w silniku: liczby całkowite muszą zostać nietknięte, a diagnostyka
    /// musi to potwierdzić własnym pomiarem.
    #[test]
    fn a_decaying_beam_conserves_the_integer_charges() {
        let muon = Particle::of(Species::Muon);
        let cfg = Config {
            spawn: SpawnConfig {
                shape: Shape::Beam,
                mixture: vec![Ingredient::new(muon, 500)],
                radius: 1.0e3,
                temperature: 0.0,
                beam_energy: 0.0,
                seed: 8,
            },
            // Bez sił: swobodna wiązka w próżni. To nie jest uproszczenie dla
            // wygody — na skali czasu życia mionu cząstki są od siebie o kilometry.
            forces: ForceConfig {
                coulomb: false,
                ..ForceConfig::default()
            },
            decay: DecayConfig::default(),
            run: RunConfig {
                dt: muon.lifetime_fm().unwrap() / 5.0,
                adaptive: false,
                max_particles: 100_000,
                ..RunConfig::default()
            },
        };
        let mut engine = Engine::new(cfg);
        let start = engine.collect_diagnostics();
        engine.advance(20).unwrap();
        let end = engine.collect_diagnostics();

        assert!(engine.tally().decayed > 400, "mionów ubyło za mało");
        assert!(end.integer_drift.is_zero(), "{:?}", end.integer_drift);
        assert_eq!(end.charge_thirds, start.charge_thirds);
        // Bez oddziaływań energia jest zachowana dokładnie, także przez rozpady.
        assert!(end.energy_drift.abs() < 1e-12, "{}", end.energy_drift);
        // Masa spoczynkowa zamieniła się w ruch — to jest treść rozpadu.
        assert!(end.rest_mass < start.rest_mass * 0.2);
        assert!(end.kinetic > start.kinetic);
    }

    /// Skok energii z narodzin ma być **zaksięgowany**, a nie zamieciony do dryfu.
    /// Bez tego jeden rozpad w plazmie zniszczyłby pomiar jakości całkowania.
    #[test]
    fn the_birth_jump_is_booked_and_the_drift_stays_clean() {
        let pion = Particle::of(Species::PionNeutral);
        let cfg = Config {
            spawn: SpawnConfig {
                shape: Shape::Ball,
                mixture: vec![Ingredient::new(pion, 60)],
                radius: 200.0,
                temperature: 0.0,
                beam_energy: 0.0,
                seed: 2,
            },
            forces: ForceConfig {
                coulomb: true,
                softening: 0.5,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
            decay: DecayConfig {
                annihilation: false,
                ..DecayConfig::default()
            },
            run: RunConfig {
                dt: pion.lifetime_fm().unwrap(),
                adaptive: false,
                max_particles: 10_000,
                ..RunConfig::default()
            },
        };
        let mut engine = Engine::new(cfg);
        engine.advance(6).unwrap();
        let snapshot = engine.collect_diagnostics();

        assert!(engine.tally().decayed > 0, "piony się nie rozpadły");
        // Kanał e⁺e⁻γ tworzy parę naładowaną w jednym punkcie — skok MUSI być
        // niezerowy, inaczej test nie sprawdzałby tego, o co chodzi.
        assert!(
            snapshot.transmutation_energy.abs() > 0.0,
            "skok z narodzin nie został zauważony"
        );
        // Krok równy czasowi życia, bez adaptacji: całkowanie ma prawo się
        // odezwać. Skok z narodzin jest rzędu 100 MeV — dryf poniżej 10⁻³
        // znaczy, że ten skok nie wyleciał do pomiaru jakości.
        assert!(
            snapshot.energy_drift.abs() < 1e-3,
            "skok wyciekł do dryfu: {}",
            snapshot.energy_drift
        );
    }

    /// Adaptacja ma przycinać krok, a jej wyłączenie — nie. Bieg nastawiony na
    /// rozpady musi móc iść krokiem o siedemnaście rzędów większym od skali sił.
    #[test]
    fn the_adaptive_step_can_be_switched_off() {
        let huge = 1.0e17;
        let mut cfg = two_body(
            Particle::of(Species::Electron),
            Particle::of(Species::Proton),
            1.0e4,
        );
        cfg.run.dt = huge;

        cfg.run.adaptive = true;
        let mut adaptive = Engine::new(cfg.clone());
        adaptive.step();
        assert!(adaptive.last_dt() < huge * 1e-10, "{}", adaptive.last_dt());

        cfg.run.adaptive = false;
        let mut fixed = Engine::new(cfg);
        fixed.step();
        assert_eq!(fixed.last_dt(), huge);
    }

    /// Adaptacja nie może przepuścić kroku, w którym cząstka przeskakuje
    /// zmiękczenie — bo wtedy mija najsilniejszy odcinek oddziaływania, nie czując go.
    #[test]
    fn the_adaptive_step_never_jumps_over_the_softening() {
        let mut cfg = two_body(
            Particle::of(Species::Electron),
            Particle::of(Species::Proton),
            1.0e3,
        );
        cfg.run.dt = 1.0e9;
        let mut engine = Engine::new(cfg);
        let softening = engine.cfg.forces.softening;
        for _ in 0..50 {
            let before = engine.state.positions.clone();
            engine.step();
            // Krok może być większy niż `ε` — zimna cząstka nie przebędzie `ε`
            // nawet w długim `dt`. Warunek dotyczy drogi, nie zegara.
            for (i, (now, was)) in engine
                .state
                .positions
                .iter()
                .zip(before.iter())
                .enumerate()
            {
                let moved = (*now - *was).norm();
                assert!(
                    moved <= softening,
                    "cząstka {i} przebyła {moved} fm przy zmiękczeniu {softening}"
                );
            }
        }
    }

    /// Uwięzienie w praktyce: para kwark-antykwark nie ucieka, choćby dostała pęd.
    /// To jest jedyny test, w którym oddziaływanie silne robi coś, czego żadne inne
    /// nie potrafi.
    #[test]
    fn quarks_cannot_escape_each_other() {
        let cfg = Config {
            spawn: SpawnConfig {
                shape: Shape::Pair,
                mixture: vec![
                    Ingredient::new(Particle::of(Species::Up), 1),
                    Ingredient::new(Particle::anti_of(Species::Up), 1),
                ],
                radius: 0.5,
                temperature: 0.0,
                // Pęd skierowany DO ŚRODKA, więc para przelatuje obok siebie
                // i próbuje uciec na drugą stronę.
                beam_energy: 300.0,
                seed: 4,
            },
            forces: ForceConfig {
                coulomb: false,
                strong: true,
                softening: 0.05,
                backend: BackendKind::Exact,
                ..ForceConfig::default()
            },
            decay: DecayConfig {
                enabled: false,
                annihilation: false,
                ..DecayConfig::default()
            },
            run: RunConfig {
                dt: 1.0e-3,
                adaptive: true,
                ..RunConfig::default()
            },
        };
        let mut engine = Engine::new(cfg);
        let mut furthest: f64 = 0.0;
        for _ in 0..4_000 {
            engine.step();
            let separation = (engine.state.positions[0] - engine.state.positions[1]).norm();
            furthest = furthest.max(separation);
        }
        // 300 MeV energii kinetycznej przy napięciu ~0,9 GeV/fm wystarcza na
        // rozciągnięcie struny o ułamek femtometra i ani femtometra więcej.
        assert!(furthest < 2.0, "kwarki uciekły na {furthest} fm");
    }

    /// Rozbiegany stan ma dać błąd, a nie ciszę i `NaN` w wyniku.
    #[test]
    fn a_diverged_state_is_reported() {
        let mut engine = Engine::new(two_body(
            Particle::of(Species::Electron),
            Particle::of(Species::Electron),
            100.0,
        ));
        engine.state.momenta[0] = crate::vec3::vec3(f64::NAN, 0.0, 0.0);
        assert!(engine.advance(1).is_err());
    }

    /// Pusty układ nie ma prawa panikować — mieszanka bez cząstek jest legalnym,
    /// choć bezcelowym, wejściem.
    #[test]
    fn an_empty_run_advances_without_panicking() {
        let cfg = Config {
            spawn: SpawnConfig {
                mixture: Vec::new(),
                ..SpawnConfig::default()
            },
            ..Config::default()
        };
        let mut engine = Engine::new(cfg);
        engine.advance(5).unwrap();
        assert_eq!(engine.n(), 0);
        assert!(engine.state.time > 0.0, "czas ma płynąć nawet w pustce");
        assert!(!engine.take_warnings().is_empty(), "pustka ma być zapowiedziana");
    }

    #[test]
    fn describe_and_census_say_what_is_inside() {
        let mut engine = Engine::new(Config::default());
        let text = engine.describe();
        assert!(text.contains("N=800"), "{text}");
        assert!(text.contains("EM"), "{text}");

        let census = engine.census_text(4);
        assert!(census.contains("e⁻×400"), "{census}");
        assert!(census.contains("p×400"), "{census}");

        engine.advance(2).unwrap();
        assert!(engine.collect_diagnostics().n == 800);
    }

    /// Ten sam bieg dwa razy musi dać ten sam wynik — z rozpadami włącznie.
    #[test]
    fn runs_are_reproducible() {
        let cfg = two_body(
            Particle::of(Species::Electron),
            Particle::of(Species::Proton),
            1.0e4,
        );
        let run = || {
            let mut engine = Engine::new(cfg.clone());
            engine.advance(80).unwrap();
            engine.state.positions.clone()
        };
        assert_eq!(run(), run());
    }
}
