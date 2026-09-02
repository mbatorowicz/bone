//! Jedna pętla biegu dla CLI i panelu: krok, diagnostyka, zapis.
//!
//! Silnik nie wie o plikach. Panel nie wie, jak często pisać klatkę. Tu schodzi
//! to, co wcześniej było skopiowane w `cli.rs` i nieistniejące w oknie.

use std::io;
use std::path::{Path, PathBuf};

use crate::grid::center_span;
use crate::io::{checkpoint, trajectory};
use crate::lcdm;
use crate::sr;
use crate::sr::diagnostics::Snapshot;
use crate::vec3::Vec3;

/// Co ile kroków ΛCDM zapisuje klatkę — ten sam rytm co dawny bieg wsadowy.
const LCDM_TRAJECTORY_EVERY: u64 = 20;

pub struct Session {
    run: Run,
    out_dir: PathBuf,
    recorder: Option<trajectory::Writer>,
}

pub enum Run {
    Relativistic(Box<Relativistic>),
    Cosmological(Box<Cosmological>),
}

pub struct Relativistic {
    pub engine: sr::Engine,
    pub latest: Option<Snapshot>,
    iteration: u64,
}

pub struct Cosmological {
    pub engine: lcdm::Engine,
}

pub struct Report {
    pub headline: String,
    pub warnings: Vec<String>,
    pub finished: bool,
}

impl Relativistic {
    fn start(cfg: sr::Config) -> Self {
        let mut run = Self {
            engine: sr::Engine::new(cfg),
            latest: None,
            iteration: 0,
        };
        run.latest = Some(run.engine.collect_diagnostics());
        run
    }

    fn from_engine(engine: sr::Engine) -> Self {
        let mut run = Self {
            engine,
            latest: None,
            iteration: 0,
        };
        run.latest = Some(run.engine.collect_diagnostics());
        run
    }

    fn advance(&mut self, steps: u32) -> Result<(), String> {
        self.engine.advance(steps).map_err(|e| e.to_string())?;
        self.iteration += 1;
        if self.engine.should_check_error(self.iteration) {
            self.engine.check_backend_error();
        }
        let every = self.engine.cfg.run.diagnostics_every.max(1) as u64;
        if self.iteration.is_multiple_of(every) || self.latest.is_none() {
            self.latest = Some(self.engine.collect_diagnostics());
        }
        Ok(())
    }

    fn rows(&self) -> Vec<(&'static str, String)> {
        let Some(s) = self.latest else {
            return Vec::new();
        };
        let mut rows = vec![
            ("czas", format!("{:>10.3}", s.time)),
            ("krok", format!("{:>10}", s.step)),
            ("N", format!("{:>10}", s.n)),
            ("K", format!("{:>10.3e}", s.kinetic)),
            ("U", format!("{:>10.3e}", s.potential)),
            ("E", format!("{:>10.3e}", s.total_energy)),
            ("dryf E", format!("{:>+10.2e}", s.energy_drift)),
            ("2K/|U|", format!("{:>10.3}", s.virial)),
            ("β średnie", format!("{:>10.3}", s.beta_mean)),
            ("β maks.", format!("{:>10.3}", s.beta_max)),
            ("γ maks.", format!("{:>10.3}", s.gamma_max)),
            ("R połowy masy", format!("{:>10.3}", s.half_mass_radius)),
            ("dryf pędu", format!("{:>10.2e}", s.momentum_residual)),
        ];
        if s.energy_removed != 0.0 {
            rows.push(("E odprowadzona", format!("{:>10.3e}", s.energy_removed)));
        }
        if let Some(err) = s.force_error {
            rows.push(("błąd siły", format!("{:>9.2}%", 100.0 * err.rms)));
        }
        rows
    }

    fn headline(&self) -> String {
        let mut text = format!(
            "t={:.3}  krok={}  {}",
            self.engine.state.time,
            self.engine.state.step,
            self.engine.describe()
        );
        if let Some(hint) = self.engine.accuracy_hint() {
            text.push_str(&format!("  ⚠ {hint}"));
        }
        text
    }
}

impl Cosmological {
    fn start(cfg: lcdm::RunConfig) -> Self {
        Self {
            engine: lcdm::Engine::new(lcdm::Cosmology::planck18(), cfg),
        }
    }

    fn from_engine(engine: lcdm::Engine) -> Self {
        Self { engine }
    }

    fn advance(&mut self, steps: u32) -> Result<(), String> {
        for _ in 0..steps.max(1) {
            if self.engine.finished() {
                break;
            }
            self.engine.advance();
        }
        if self.engine.positions.iter().any(|p| !p.is_finite()) {
            return Err(format!(
                "stan przestał być skończony na kroku {}",
                self.engine.step
            ));
        }
        Ok(())
    }

    fn rows(&self) -> Vec<(&'static str, String)> {
        let e = self.engine.energies();
        let c = &self.engine.cosmology;
        vec![
            ("z", format!("{:>10.3}", self.engine.redshift())),
            ("a", format!("{:>10.4}", self.engine.a)),
            ("wiek [Gyr]", format!("{:>10.2}", self.engine.age_gyr())),
            ("krok", format!("{:>10}", self.engine.step)),
            ("N", format!("{:>10}", self.engine.n())),
            ("T", format!("{:>10.3e}", e.kinetic)),
            ("W", format!("{:>10.3e}", e.potential)),
            (
                "residuum LI",
                format!("{:>+10.2e}", self.engine.layzer_irvine()),
            ),
            (
                "σ(δ) start",
                format!("{:>10.3}", self.engine.initial_contrast),
            ),
            ("Ω_m", format!("{:>10.3}", c.omega_m)),
            ("h", format!("{:>10.4}", c.h)),
            ("σ₈", format!("{:>10.3}", c.sigma8)),
        ]
    }

    fn headline(&self) -> String {
        format!(
            "z={:.3}  a={:.4}  wiek={:.2} Gyr  krok={}  {}",
            self.engine.redshift(),
            self.engine.a,
            self.engine.age_gyr(),
            self.engine.step,
            self.engine.describe_solver()
        )
    }
}

impl Session {
    pub fn start_sr(cfg: sr::Config, out_dir: PathBuf, record: bool) -> Result<Self, String> {
        let stride = cfg.run.point_stride;
        let run = Run::Relativistic(Box::new(Relativistic::start(cfg)));
        Self::assemble(run, out_dir, record, stride)
    }

    pub fn start_lcdm(cfg: lcdm::RunConfig, out_dir: PathBuf, record: bool) -> Result<Self, String> {
        let run = Run::Cosmological(Box::new(Cosmological::start(cfg)));
        Self::assemble(run, out_dir, record, 1)
    }

    pub fn resume(out_dir: PathBuf, record: bool) -> Result<Self, String> {
        let kind = checkpoint::kind(&out_dir).ok_or_else(|| {
            format!("brak checkpointu w {}", out_dir.display())
        })?;
        match kind {
            checkpoint::Kind::Relativistic => {
                let (state, cfg) = checkpoint::load(&out_dir)
                    .map_err(|e| format!("wznowienie z {}: {e}", out_dir.display()))?;
                let stride = cfg.run.point_stride;
                let run = Run::Relativistic(Box::new(Relativistic::from_engine(
                    sr::Engine::with_state(cfg, state, Vec::new()),
                )));
                Self::assemble(run, out_dir, record, stride)
            }
            checkpoint::Kind::Cosmological => {
                let engine = checkpoint::load_lcdm(&out_dir)
                    .map_err(|e| format!("wznowienie z {}: {e}", out_dir.display()))?;
                let run = Run::Cosmological(Box::new(Cosmological::from_engine(engine)));
                Self::assemble(run, out_dir, record, 1)
            }
        }
    }

    fn assemble(
        run: Run,
        out_dir: PathBuf,
        record: bool,
        stride: usize,
    ) -> Result<Self, String> {
        let recorder = if record {
            Some(
                trajectory::Writer::new(&out_dir, stride)
                    .map_err(|e| format!("nie mogę pisać trajektorii do {}: {e}", out_dir.display()))?,
            )
        } else {
            None
        };
        Ok(Self {
            run,
            out_dir,
            recorder,
        })
    }

    pub fn advance(&mut self, steps: u32) -> Result<Report, String> {
        match &mut self.run {
            Run::Relativistic(run) => run.advance(steps)?,
            Run::Cosmological(run) => run.advance(steps)?,
        }
        self.maybe_record()?;
        Ok(Report {
            headline: self.headline(),
            warnings: self.take_warnings(),
            finished: self.finished(),
        })
    }

    fn maybe_record(&mut self) -> Result<(), String> {
        let Some(recorder) = self.recorder.as_mut() else {
            return Ok(());
        };
        match &self.run {
            Run::Relativistic(run) => {
                let every = run.engine.cfg.run.trajectory_every.max(1) as u64;
                let step = run.engine.state.step;
                if step == 0 || !step.is_multiple_of(every) {
                    return Ok(());
                }
                let state = &run.engine.state;
                let c = run.engine.cfg.physics.c;
                recorder
                    .push(state.time, &state.positions, |i| {
                        sr::relativity::speed_over_c(state.masses[i], state.momenta[i], c) as f32
                    })
                    .map_err(|e| format!("zapis klatki: {e}"))
            }
            Run::Cosmological(run) => {
                let step = run.engine.step;
                if step == 0 || !step.is_multiple_of(LCDM_TRAJECTORY_EVERY) {
                    return Ok(());
                }
                recorder
                    .push(run.engine.a, &run.engine.positions, |i| run.engine.shade(i))
                    .map_err(|e| format!("zapis klatki: {e}"))
            }
        }
    }

    pub fn save_checkpoint(&self) -> io::Result<PathBuf> {
        match &self.run {
            Run::Relativistic(run) => {
                checkpoint::save(&run.engine.state, &run.engine.cfg, &self.out_dir)
            }
            Run::Cosmological(run) => checkpoint::save_lcdm(&run.engine, &self.out_dir),
        }
    }

    pub fn flush_recording(&mut self) -> Result<usize, String> {
        let Some(recorder) = self.recorder.as_mut() else {
            return Ok(0);
        };
        recorder
            .flush()
            .map_err(|e| format!("zapis indeksu: {e}"))?;
        Ok(recorder.n_frames())
    }

    pub fn finished(&self) -> bool {
        match &self.run {
            Run::Relativistic(_) => false,
            Run::Cosmological(run) => run.engine.finished(),
        }
    }

    pub fn rows(&self) -> Vec<(&'static str, String)> {
        match &self.run {
            Run::Relativistic(run) => run.rows(),
            Run::Cosmological(run) => run.rows(),
        }
    }

    pub fn headline(&self) -> String {
        match &self.run {
            Run::Relativistic(run) => run.headline(),
            Run::Cosmological(run) => run.headline(),
        }
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        match &mut self.run {
            Run::Relativistic(run) => run.engine.take_warnings(),
            Run::Cosmological(_) => Vec::new(),
        }
    }

    pub fn apply_runtime_sr(&mut self, live: &sr::Config) {
        if let Run::Relativistic(run) = &mut self.run {
            run.engine.apply_runtime_config(live);
        }
    }

    pub fn apply_runtime_lcdm(&mut self, dlna: f64) {
        if let Run::Cosmological(run) = &mut self.run {
            run.engine.cfg.dlna = dlna;
        }
    }

    pub fn sr_snapshot(&self) -> Option<&Snapshot> {
        match &self.run {
            Run::Relativistic(run) => run.latest.as_ref(),
            Run::Cosmological(_) => None,
        }
    }

    pub fn accuracy_hint(&self) -> Option<String> {
        match &self.run {
            Run::Relativistic(run) => run.engine.accuracy_hint(),
            Run::Cosmological(_) => None,
        }
    }

    pub fn should_check_error(&self) -> bool {
        match &self.run {
            Run::Relativistic(run) => run.engine.should_check_error(run.iteration),
            Run::Cosmological(_) => false,
        }
    }

    pub fn is_relativistic(&self) -> bool {
        matches!(self.run, Run::Relativistic(_))
    }

    pub fn lcdm_log_values(&self) -> Option<(u64, f64, f64, f64, f64, f64)> {
        match &self.run {
            Run::Cosmological(run) => {
                let e = run.engine.energies();
                Some((
                    run.engine.step,
                    run.engine.redshift(),
                    run.engine.age_gyr(),
                    e.kinetic,
                    e.potential,
                    run.engine.layzer_irvine(),
                ))
            }
            Run::Relativistic(_) => None,
        }
    }

    pub fn n(&self) -> usize {
        match &self.run {
            Run::Relativistic(run) => run.engine.state.n(),
            Run::Cosmological(run) => run.engine.n(),
        }
    }

    pub fn position(&self, index: usize) -> Vec3 {
        match &self.run {
            Run::Relativistic(run) => run.engine.state.positions[index],
            Run::Cosmological(run) => run.engine.positions[index],
        }
    }

    pub fn shade(&self, index: usize) -> f32 {
        match &self.run {
            Run::Relativistic(run) => run
                .engine
                .state
                .speed_over_c(index, run.engine.cfg.physics.c) as f32,
            Run::Cosmological(run) => run.engine.shade(index),
        }
    }

    pub fn center_span(&self) -> (Vec3, f64) {
        match &self.run {
            Run::Relativistic(run) => center_span(&run.engine.state.positions),
            Run::Cosmological(run) => run.engine.center_span(),
        }
    }

    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    pub fn n_frames(&self) -> usize {
        self.recorder.as_ref().map_or(0, trajectory::Writer::n_frames)
    }

    pub fn diagnostics_every(&self) -> u64 {
        match &self.run {
            Run::Relativistic(run) => run.engine.cfg.run.diagnostics_every.max(1) as u64,
            Run::Cosmological(_) => LCDM_TRAJECTORY_EVERY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_sr() -> sr::Config {
        let mut cfg = sr::presets::galaxy();
        cfg.spawn.n_particles = 80;
        cfg.solver.backend = sr::config::BackendKind::Exact;
        cfg.run.trajectory_every = 2;
        cfg.run.diagnostics_every = 1;
        cfg
    }

    fn small_lcdm() -> lcdm::RunConfig {
        lcdm::RunConfig {
            n_grid: 8,
            pm_grid: 16,
            ..lcdm::RunConfig::linear_growth()
        }
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "bone-session-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn session_advances_and_reports_both_models() {
        let mut sr = Session::start_sr(small_sr(), temp_dir("sr-adv"), false).unwrap();
        sr.advance(4).unwrap();
        assert!(!sr.rows().is_empty());
        assert!(sr.headline().contains("krok"));
        assert_eq!(sr.n(), 80);

        let mut lcdm = Session::start_lcdm(small_lcdm(), temp_dir("lcdm-adv"), false).unwrap();
        lcdm.advance(3).unwrap();
        assert!(!lcdm.rows().is_empty());
        assert!(lcdm.headline().contains("z="));
        assert_eq!(lcdm.n(), 8usize.pow(3));
    }

    #[test]
    fn session_writes_frames_and_checkpoint() {
        let dir = temp_dir("record");
        let mut session = Session::start_sr(small_sr(), dir.clone(), true).unwrap();
        session.advance(4).unwrap();
        let frames = session.flush_recording().unwrap();
        assert!(frames >= 1, "brak klatek: {frames}");
        session.save_checkpoint().unwrap();
        assert!(checkpoint::exists(&dir));
        assert_eq!(checkpoint::kind(&dir), Some(checkpoint::Kind::Relativistic));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lcdm_session_round_trips_through_checkpoint() {
        let dir = temp_dir("lcdm-resume");
        let mut session = Session::start_lcdm(small_lcdm(), dir.clone(), false).unwrap();
        session.advance(5).unwrap();
        let step = session.lcdm_log_values().unwrap().0;
        session.save_checkpoint().unwrap();
        let resumed = Session::resume(dir.clone(), false).unwrap();
        assert!(!resumed.is_relativistic());
        assert_eq!(resumed.lcdm_log_values().unwrap().0, step);
        std::fs::remove_dir_all(&dir).ok();
    }
}
