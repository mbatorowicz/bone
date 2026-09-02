//! Silnik SR: stan + solver + integrator + dyssypacja + diagnostyka.
//!
//! Pętlę biegu pisze WYWOŁUJĄCY — bieg wsadowy albo panel — a silnik udostępnia tylko
//! [`Engine::step`] i [`Engine::advance`]. Odwrotny podział, w którym silnik trzyma
//! pętlę i woła wstrzyknięte domknięcia (`on_frame`, `on_diagnostics`, `should_stop`),
//! wymagałby domknięć pożyczających silnik, żeby zapisać klatkę, i jednocześnie
//! wołanych przez ten silnik — czyli komórek współdzielonych i sprawdzania pożyczek
//! w czasie działania. Przy okazji ten podział usuwa pola konfiguracji, które służyłyby
//! wyłącznie do ustalania, jak często silnik ma wołać cudzy kod.

use crate::rng::Rng;
use crate::sr::backends::{make_backend, Backend};
use crate::sr::config::Config;
use crate::sr::cooling::Cooling;
use crate::sr::diagnostics::{measure_force_error, Context, Diagnostics, ForceError, Snapshot};
use crate::sr::integrator;
use crate::sr::spawn;
use crate::sr::state::State;

pub struct Engine {
    pub cfg: Config,
    pub state: State,
    pub diagnostics: Diagnostics,
    backend: Box<dyn Backend>,
    cooling: Cooling,
    rng: Rng,
    last_dt: f64,
    /// Skumulowana energia odprowadzona przez dyssypację.
    ///
    /// Wielkością zachowaną w modelu z chłodzeniem jest `E_tot + ta suma`, i to jej
    /// dryf mierzy diagnostyka.
    energy_removed: f64,
    pending_force_error: Option<ForceError>,
    last_force_error: Option<f64>,
    warnings: Vec<String>,
}

impl Engine {
    pub fn new(cfg: Config) -> Self {
        let spawned = spawn::make_state(&cfg);
        Self::with_state(cfg, spawned.state, spawned.warnings)
    }

    /// Zbuduj silnik na gotowym stanie — używane przy wznawianiu z checkpointu.
    pub fn with_state(cfg: Config, state: State, warnings: Vec<String>) -> Self {
        let mut backend = make_backend(&cfg, state.n());
        let cooling = Cooling::new(cfg.physics.cooling_grid, cfg.solver.box_margin)
            .unwrap_or_else(|_| Cooling::new(0, cfg.solver.box_margin).expect("automat"));
        let mut state = state;
        integrator::ensure_field(backend.as_mut(), &mut state, &cfg.physics);
        Self {
            last_dt: cfg.physics.dt_max,
            rng: Rng::seeded(cfg.spawn.seed.wrapping_add(1)),
            cfg,
            state,
            diagnostics: Diagnostics::new(),
            backend,
            cooling,
            energy_removed: 0.0,
            pending_force_error: None,
            last_force_error: None,
            warnings,
        }
    }

    /// Wykonaj jeden krok. Zwraca użyty `Δt`.
    pub fn step(&mut self) -> f64 {
        self.last_dt = integrator::step(self.backend.as_mut(), &mut self.state, &self.cfg.physics);
        // Rozdzielenie operatorów: całkowanie zachowawcze, potem dyssypacja.
        // Chłodzenie zmienia tylko pędy, więc siły policzone w kroku pozostają
        // aktualne i nie ma potrzeby ich unieważniać.
        self.energy_removed += self
            .cooling
            .apply(&mut self.state, &self.cfg.physics, self.last_dt);
        if let Some(warning) = self.cooling.take_warning() {
            self.warnings.push(warning);
        }
        self.last_dt
    }

    /// Wykonaj `n` kroków. Zwraca sumaryczny upływ czasu symulacji.
    ///
    /// # Errors
    /// Gdy stan przestanie być skończony — dalsze liczenie nie miałoby sensu, a ciche
    /// kontynuowanie dawałoby wykresy pełne `NaN` bez wskazania, kiedy to się stało.
    pub fn advance(&mut self, n: u32) -> Result<f64, Diverged> {
        let mut elapsed = 0.0;
        for _ in 0..n.max(1) {
            elapsed += self.step();
        }
        if !self.state.is_finite() {
            return Err(Diverged {
                step: self.state.step,
            });
        }
        Ok(elapsed)
    }

    pub fn collect_diagnostics(&mut self) -> Snapshot {
        let ctx = Context {
            dt: self.last_dt,
            energy_removed: self.energy_removed,
            effective_softening: self.effective_softening(),
            approximate: self.backend.approximate(),
            force_error: self.pending_force_error.take(),
            cooling_particles_per_cell: if self.cfg.physics.cooling_rate > 0.0 {
                Some(self.cooling.particles_per_cell())
            } else {
                None
            },
        };
        self.diagnostics.observe(&self.state, self.cfg.physics.c, ctx)
    }

    pub fn effective_softening(&self) -> f64 {
        self.backend.effective_softening(self.cfg.physics.softening)
    }

    /// Zmierz błąd przybliżenia względem dokładnego O(N²).
    ///
    /// Wzorzec liczymy z softeningiem, którym solver NAPRAWDĘ pracuje — inaczej
    /// mierzylibyśmy nie błąd metody, tylko różnicę dwóch modeli.
    pub fn check_backend_error(&mut self) -> Option<ForceError> {
        let forces = self.state.forces.clone()?;
        let error = measure_force_error(
            &self.state,
            &forces,
            self.cfg.physics.g,
            self.effective_softening(),
            self.cfg.solver.error_check_sample,
            &mut self.rng,
        );
        self.pending_force_error = Some(error);
        self.last_force_error = Some(error.rms);
        Some(error)
    }

    /// Czy na tym kroku wypada pomiar błędu.
    pub fn should_check_error(&self, iteration: u64) -> bool {
        let every = self.cfg.solver.error_check_every as u64;
        every > 0 && iteration.is_multiple_of(every)
    }

    /// Podmień konfigurację; przebuduj solver, jeśli zmieniły się jego parametry.
    ///
    /// Tylko pola runtime'owe są brane z `updated` — o tym, które to, decyduje
    /// [`Config::with_runtime_from`].
    pub fn apply_runtime_config(&mut self, updated: &Config) {
        let next = self.cfg.with_runtime_from(updated);
        if next == self.cfg {
            return;
        }
        let solver_changed = self.cfg.solver_changed(&next);
        let field_invalid = self.cfg.field_invalidated(&next);
        let cooling_changed = self.cfg.physics.cooling_grid != next.physics.cooling_grid
            || self.cfg.solver.box_margin != next.solver.box_margin;
        self.cfg = next;

        if cooling_changed {
            if let Ok(fresh) = Cooling::new(self.cfg.physics.cooling_grid, self.cfg.solver.box_margin)
            {
                self.cooling = fresh;
            }
        }
        if solver_changed {
            self.backend = make_backend(&self.cfg, self.state.n());
        }
        if solver_changed || field_invalid {
            self.state.invalidate_field();
            integrator::ensure_field(self.backend.as_mut(), &mut self.state, &self.cfg.physics);
            self.diagnostics.reset_reference();
        }
    }

    pub fn describe(&self) -> String {
        if self.cfg.physics.cooling_rate > 0.0 {
            format!("{} · {}", self.backend.describe(), self.cooling.describe())
        } else {
            self.backend.describe()
        }
    }

    /// Ostrzeżenia zebrane od ostatniego odczytu.
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    pub fn energy_removed(&self) -> f64 {
        self.energy_removed
    }

    /// Co zrobić z wysokim błędem siły — sama liczba nie mówi nic o działaniu.
    ///
    /// Błąd rośnie w trakcie biegu, kiedy układ wytworzy strukturę mniejszą od oczka
    /// siatki: PM rozdziela grawitację tylko do rozmiaru komórki.
    pub fn accuracy_hint(&self) -> Option<String> {
        let error = self.last_force_error?;
        if !self.backend.approximate() || error < 0.02 {
            return None;
        }
        let finer = (2 * self.cfg.solver.grid).min(384);
        let remedy = if finer > self.cfg.solver.grid {
            format!("zagęść siatkę do {finer}³")
        } else {
            "użyj solvera „exact”".to_string()
        };
        Some(format!(
            "błąd siły {:.1}% — układ ma strukturę drobniejszą od oczka siatki; {remedy}",
            100.0 * error
        ))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Diverged {
    pub step: u64,
}

impl std::fmt::Display for Diverged {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stan przestał być skończony na kroku {}", self.step)
    }
}

impl std::error::Error for Diverged {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::config::{BackendKind, Geometry, PhysicsConfig, SolverConfig, SpawnConfig};
    use crate::sr::presets;

    fn small(n: usize) -> Config {
        Config {
            spawn: SpawnConfig {
                geometry: Geometry::Plummer,
                n_particles: n,
                ..SpawnConfig::default()
            },
            solver: SolverConfig {
                backend: BackendKind::Exact,
                ..SolverConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn engine_starts_with_a_field_and_advances() {
        let mut eng = Engine::new(small(200));
        assert!(eng.state.forces.is_some());
        assert!(eng.state.potential.is_some());
        let elapsed = eng.advance(10).expect("nie rozbiegło się");
        assert!(elapsed > 0.0);
        assert_eq!(eng.state.step, 10);
    }

    /// Bez chłodzenia energia musi być zachowana z dryfem ograniczonym — to główny
    /// sprawdzian, czy silnik składa integrator, solver i diagnostykę spójnie.
    #[test]
    fn conservative_run_keeps_its_energy() {
        let mut cfg = presets::precision();
        cfg.spawn.n_particles = 400;
        let mut eng = Engine::new(cfg);
        eng.collect_diagnostics();
        for _ in 0..300 {
            eng.step();
        }
        let snap = eng.collect_diagnostics();
        assert!(
            snap.energy_drift.abs() < 0.02,
            "dryf energii {}",
            snap.energy_drift
        );
        assert_eq!(snap.energy_removed, 0.0);
    }

    /// Z chłodzeniem energia całkowita MA spadać, ale suma
    /// `E_tot + E_odprowadzona` nadal musi być zachowana.
    #[test]
    fn dissipative_run_balances_the_removed_energy() {
        let mut cfg = small(600);
        cfg.physics.cooling_rate = 2.0;
        cfg.spawn.temperature = 0.1;
        cfg.spawn.rotation = 0.0;
        let mut eng = Engine::new(cfg);
        let first = eng.collect_diagnostics();
        for _ in 0..200 {
            eng.step();
        }
        let snap = eng.collect_diagnostics();
        assert!(eng.energy_removed() > 0.0, "nic nie wypromieniowano");
        assert!(
            snap.total_energy < first.total_energy,
            "energia nie spadła: {} → {}",
            first.total_energy,
            snap.total_energy
        );
        assert!(
            snap.energy_drift.abs() < 0.05,
            "bilans z dyssypacją nie domyka się: {}",
            snap.energy_drift
        );
    }

    #[test]
    fn runtime_config_changes_g_but_not_particle_count() {
        let mut eng = Engine::new(small(300));
        let n_before = eng.state.n();
        let mut wanted = eng.cfg.clone();
        wanted.physics.g *= 2.0;
        wanted.spawn.n_particles = 999_999;
        wanted.spawn.seed = 12_345;
        eng.apply_runtime_config(&wanted);

        assert_eq!(eng.state.n(), n_before, "liczba cząstek zmieniona w locie");
        assert_eq!(eng.cfg.spawn.seed, small(300).spawn.seed, "ziarno zmienione");
        assert!((eng.cfg.physics.g - 2.0 * small(300).physics.g).abs() < 1e-12);
    }

    /// Zmiana G unieważnia siły. Gdyby zostały stare, pierwszy krok po zmianie
    /// policzyłby kopnięcie z poprzedniej fizyki.
    #[test]
    fn changing_gravity_recomputes_the_field() {
        let mut eng = Engine::new(small(200));
        let before = eng.state.forces.clone().unwrap();
        let mut wanted = eng.cfg.clone();
        wanted.physics.g *= 3.0;
        eng.apply_runtime_config(&wanted);
        let after = eng.state.forces.clone().unwrap();
        let changed = before
            .iter()
            .zip(after.iter())
            .any(|(a, b)| (*a - *b).norm() > 1e-12);
        assert!(changed, "siły nie zostały przeliczone");
    }

    #[test]
    fn switching_backend_rebuilds_the_solver() {
        let mut cfg = small(600);
        cfg.solver.backend = BackendKind::Exact;
        let mut eng = Engine::new(cfg);
        assert!(!eng.backend.approximate());

        // backend jest polem startowym, więc runtime go nie zmienia
        let mut wanted = eng.cfg.clone();
        wanted.solver.backend = BackendKind::Mesh;
        eng.apply_runtime_config(&wanted);
        assert!(!eng.backend.approximate(), "backend zmieniony w locie");
    }

    #[test]
    fn error_check_schedule_respects_zero_as_never() {
        let mut cfg = small(100);
        cfg.solver.error_check_every = 0;
        let eng = Engine::new(cfg);
        assert!(!eng.should_check_error(1));
        assert!(!eng.should_check_error(1_000));

        let mut cfg = small(100);
        cfg.solver.error_check_every = 10;
        let eng = Engine::new(cfg);
        assert!(eng.should_check_error(10));
        assert!(!eng.should_check_error(11));
    }

    #[test]
    fn exact_backend_reports_no_accuracy_hint() {
        let mut eng = Engine::new(small(200));
        eng.check_backend_error();
        assert!(eng.accuracy_hint().is_none());
    }

    #[test]
    fn mesh_backend_reports_measured_error() {
        let mut cfg = small(3_000);
        cfg.solver.backend = BackendKind::Mesh;
        cfg.solver.grid = 32;
        let mut eng = Engine::new(cfg);
        let err = eng.check_backend_error().expect("siły są policzone");
        assert!(err.rms > 0.0, "siatka nie ma błędu?");
        let snap = eng.collect_diagnostics();
        assert!(snap.approximate);
        assert!(snap.force_error.is_some());
        // pomiar jest jednorazowy: drugi odczyt nie powtarza tej samej liczby
        assert!(eng.collect_diagnostics().force_error.is_none());
    }

    #[test]
    fn describe_mentions_cooling_only_when_enabled() {
        let eng = Engine::new(small(100));
        assert!(!eng.describe().contains("chłodzenie"));

        let mut cfg = small(100);
        cfg.physics.cooling_rate = 1.0;
        let eng = Engine::new(cfg);
        assert!(eng.describe().contains("chłodzenie"));
    }

    #[test]
    fn unreachable_initial_condition_surfaces_as_warning() {
        let cfg = Config {
            spawn: SpawnConfig {
                n_particles: 200,
                geometry: Geometry::Ball,
                rotation: 1.0,
                total_mass: 1e6,
                radius: 1.0,
                ..SpawnConfig::default()
            },
            physics: PhysicsConfig {
                g: 10.0,
                c: 1.0,
                ..PhysicsConfig::default()
            },
            solver: SolverConfig {
                backend: BackendKind::Exact,
                ..SolverConfig::default()
            },
            ..Config::default()
        };
        let mut eng = Engine::new(cfg);
        assert!(!eng.take_warnings().is_empty());
        assert!(eng.take_warnings().is_empty(), "ostrzeżenia się powtarzają");
    }
}
