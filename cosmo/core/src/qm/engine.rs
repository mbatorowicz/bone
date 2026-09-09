//! Silnik atomowy: faza w czasie i realizacja `|ψ|²` jako chmury punktów.
//!
//! Nie ma tu całkowania sił. Stan stacjonarny ma stałą gęstość; superpozycja
//! stanów o różnych energiach bije z okresem `2π ħ / |ΔE|`. Krok to przesunięcie
//! fazy i ewentualne przesample’owanie chmury, nie leapfrog.

use crate::qm::config::{Config, Scene, Term};
use crate::qm::elements::{self, occupied_orbitals, Occupied};
use crate::qm::hydrogen::{self, density, energy_hartree, mean_radius, wavefunction};
use crate::qm::sample;
use crate::qm::state::State;
use crate::vec3::{vec3, Vec3};

pub struct Engine {
    pub cfg: Config,
    pub state: State,
    catalog: Catalog,
    warnings: Vec<String>,
}

enum Catalog {
    Orbital {
        z: f64,
        n: u32,
        l: u32,
        m: i32,
        mu: f64,
    },
    Superposition {
        z: f64,
        mu: f64,
        terms: Vec<LiveTerm>,
    },
    Atom {
        electrons: Vec<Occupied>,
        mu: f64,
    },
    Helium {
        zeta: f64,
        mu: f64,
    },
}

#[derive(Clone, Copy)]
struct LiveTerm {
    n: u32,
    l: u32,
    m: i32,
    re: f64,
    im: f64,
    energy: f64,
}

impl Catalog {
    fn compile(scene: &Scene) -> Result<Self, String> {
        match scene {
            Scene::Orbital { z, n, l, m } => {
                if !hydrogen::quantum_ok(*n, *l, *m) {
                    return Err(format!("niedozwolone liczby kwantowe n={n} l={l} m={m}"));
                }
                let el = elements::nearest(*z);
                Ok(Self::Orbital {
                    z: *z as f64,
                    n: *n,
                    l: *l,
                    m: *m,
                    mu: el.reduced_mass(),
                })
            }
            Scene::Superposition { z, terms } => {
                let cleaned: Vec<&Term> = terms
                    .iter()
                    .filter(|t| hydrogen::quantum_ok(t.n, t.l, t.m) && t.amplitude_sq() > 0.0)
                    .collect();
                if cleaned.is_empty() {
                    return Err("superpozycja nie ma żadnego dozwolonego składnika".into());
                }
                let norm = cleaned.iter().map(|t| t.amplitude_sq()).sum::<f64>().sqrt();
                let el = elements::nearest(*z);
                let mu = el.reduced_mass();
                let terms = cleaned
                    .into_iter()
                    .map(|t| LiveTerm {
                        n: t.n,
                        l: t.l,
                        m: t.m,
                        re: t.re / norm,
                        im: t.im / norm,
                        energy: energy_hartree(*z as f64, t.n, mu),
                    })
                    .collect();
                Ok(Self::Superposition {
                    z: *z as f64,
                    mu,
                    terms,
                })
            }
            Scene::Atom { z } => {
                let el = elements::nearest(*z);
                Ok(Self::Atom {
                    electrons: occupied_orbitals(el),
                    mu: el.reduced_mass(),
                })
            }
            Scene::Helium => {
                let el = elements::nearest(2);
                Ok(Self::Helium {
                    zeta: crate::qm::helium::ZETA,
                    mu: el.reduced_mass(),
                })
            }
        }
    }

    fn density(&self, p: Vec3, time: f64) -> f64 {
        match self {
            Self::Orbital { z, n, l, m, .. } => density(*z, *n, *l, *m, p.x, p.y, p.z),
            Self::Superposition { z, terms, .. } => superposition_density(*z, terms, p, time),
            Self::Helium { zeta, .. } => density(*zeta, 1, 0, 0, p.x, p.y, p.z),
            Self::Atom { .. } => 0.0,
        }
    }

    fn typical_radius(&self) -> f64 {
        match self {
            Self::Orbital { z, n, l, .. } => mean_radius(*z, *n, *l),
            Self::Superposition { z, terms, .. } => terms
                .iter()
                .map(|t| mean_radius(*z, t.n, t.l))
                .fold(1.0, f64::max),
            Self::Atom { electrons, .. } => electrons
                .iter()
                .map(|e| mean_radius(e.z_eff, e.n, e.l))
                .fold(1.0, f64::max),
            Self::Helium { zeta, .. } => mean_radius(*zeta, 1, 0),
        }
    }

    fn mu(&self) -> f64 {
        match self {
            Self::Orbital { mu, .. }
            | Self::Superposition { mu, .. }
            | Self::Atom { mu, .. }
            | Self::Helium { mu, .. } => *mu,
        }
    }
}

fn superposition_density(z: f64, terms: &[LiveTerm], p: Vec3, time: f64) -> f64 {
    let mut re = 0.0;
    let mut im = 0.0;
    for t in terms {
        let psi = wavefunction(z, t.n, t.l, t.m, p.x, p.y, p.z);
        let phase = t.energy * time;
        let (c, s) = (phase.cos(), phase.sin());
        // (a + ib) · (cos − i sin) · ψ, ψ rzeczywiste.
        re += psi * (t.re * c + t.im * s);
        im += psi * (t.im * c - t.re * s);
    }
    re * re + im * im
}

impl Engine {
    pub fn new(cfg: Config) -> Self {
        let cfg = cfg.sanitized();
        let warnings = cfg.warnings();
        let catalog = Catalog::compile(&cfg.scene).expect("sanitized scene compiles");
        let (positions, shades) = sample_cloud(&catalog, &cfg, 0.0);
        let state = State::new(positions, shades).expect("próbka ma odcień na punkt");
        Self {
            cfg,
            state,
            catalog,
            warnings,
        }
    }

    pub fn with_state(cfg: Config, mut state: State) -> Self {
        let cfg = cfg.sanitized();
        let warnings = cfg.warnings();
        let catalog = Catalog::compile(&cfg.scene).expect("sanitized scene compiles");
        if state.n() != cfg.run.n_samples {
            let (positions, shades) = sample_cloud(&catalog, &cfg, state.time);
            state.positions = positions;
            state.shades = shades;
        } else {
            state.shades = shades_for(&catalog, &state.positions, state.time);
        }
        Self {
            cfg,
            state,
            catalog,
            warnings,
        }
    }

    pub fn advance(&mut self, steps: u32) -> Result<(), String> {
        let dt = self.cfg.run.dt;
        for _ in 0..steps.max(1) {
            self.state.time += dt;
            self.state.step += 1;
            if self.cfg.scene.is_time_dependent() {
                let step = 0.35 * self.catalog.typical_radius();
                let seed = self.cfg.run.seed.wrapping_add(self.state.step);
                let time = self.state.time;
                sample::relax(
                    &mut self.state.positions,
                    |p| self.catalog.density(p, time),
                    seed,
                    step,
                    2,
                );
            } else if self.cfg.run.sparkle {
                let step = 0.25 * self.catalog.typical_radius();
                let seed = self.cfg.run.seed.wrapping_add(self.state.step.wrapping_mul(17));
                sample::relax(
                    &mut self.state.positions,
                    |p| self.catalog.density(p, 0.0),
                    seed,
                    step,
                    1,
                );
            }
        }
        self.state.shades = shades_for(&self.catalog, &self.state.positions, self.state.time);
        if !self.state.is_finite() {
            return Err(format!(
                "stan przestał być skończony na kroku {}",
                self.state.step
            ));
        }
        Ok(())
    }

    pub fn apply_runtime_config(&mut self, live: &Config) {
        let live = live.sanitized();
        let scene_changed = live.scene != self.cfg.scene;
        let n_changed = live.run.n_samples != self.cfg.run.n_samples;
        self.cfg.run = live.run;
        if scene_changed {
            self.cfg.scene = live.scene;
            self.warnings = self.cfg.warnings();
            self.catalog = Catalog::compile(&self.cfg.scene).expect("sanitized scene compiles");
        }
        if scene_changed || n_changed {
            let (positions, shades) = sample_cloud(&self.catalog, &self.cfg, self.state.time);
            self.state.positions = positions;
            self.state.shades = shades;
        }
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    pub fn describe(&self) -> String {
        let exact = match &self.cfg.scene {
            Scene::Orbital { .. } | Scene::Superposition { .. } => "dokładny Schrödinger",
            Scene::Helium => "wariacja ζ=27/16",
            Scene::Atom { z } => {
                let el = elements::nearest(*z);
                if elements::reports_ionization(el) {
                    "Slater Z_eff"
                } else {
                    "Aufbau (konfiguracja)"
                }
            }
        };
        format!("{} · {}", self.cfg.scene.label(), exact)
    }

    pub fn accuracy_hint(&self) -> Option<String> {
        match &self.cfg.scene {
            Scene::Helium => {
                let err = crate::qm::helium::energy_error();
                Some(format!(
                    "wariacja He {:+.1}% vs −2.904 Ha",
                    100.0 * err
                ))
            }
            Scene::Atom { z } => {
                let el = elements::nearest(*z);
                if !elements::reports_ionization(el) {
                    return Some(format!(
                        "{}: konfiguracja Aufbau, bez IE",
                        el.symbol
                    ));
                }
                elements::ionization_error(el).and_then(|err| {
                    if err.abs() > 0.15 {
                        Some(format!(
                            "IE modelu {:+.0}% vs NIST na {}",
                            100.0 * err,
                            el.symbol
                        ))
                    } else {
                        None
                    }
                })
            }
            _ => None,
        }
    }

    pub fn shade(&self, index: usize) -> f32 {
        self.state.shades.get(index).copied().unwrap_or(0.5)
    }

    pub fn catalog_mu(&self) -> f64 {
        self.catalog.mu()
    }

    pub fn typical_radius(&self) -> f64 {
        self.catalog.typical_radius()
    }

    pub fn collect_diagnostics(&self) -> crate::qm::diagnostics::Snapshot {
        crate::qm::diagnostics::collect(
            &self.cfg,
            self.state.time,
            self.state.step,
            &self.state.positions,
            self.catalog.mu(),
        )
    }
}

fn sample_cloud(catalog: &Catalog, cfg: &Config, time: f64) -> (Vec<Vec3>, Vec<f32>) {
    let n = cfg.run.n_samples;
    let seed = cfg.run.seed;
    let positions = match catalog {
        Catalog::Orbital { z, n: qn, l, m, .. } => {
            sample::sample_orbital(*z, *qn, *l, *m, n, seed)
        }
        Catalog::Superposition { .. } => {
            let scale = catalog.typical_radius();
            sample::metropolis(
                |p| catalog.density(p, time),
                n,
                seed,
                0.4 * scale,
                vec3(0.0, 0.0, 0.5 * scale),
            )
        }
        Catalog::Atom { electrons, .. } => sample_atom(electrons, n, seed),
        Catalog::Helium { zeta, .. } => sample_helium(*zeta, n, seed),
    };
    let shades = shades_for(catalog, &positions, time);
    (positions, shades)
}

fn sample_atom(electrons: &[Occupied], n: usize, seed: u64) -> Vec<Vec3> {
    if electrons.is_empty() {
        return sample::sample_orbital(1.0, 1, 0, 0, n, seed);
    }
    let mut out = Vec::with_capacity(n);
    let base = n / electrons.len();
    let extra = n - base * electrons.len();
    for (i, e) in electrons.iter().enumerate() {
        let count = base + usize::from(i < extra);
        if count == 0 {
            continue;
        }
        let pts = sample::sample_orbital(e.z_eff, e.n, e.l, e.m, count, seed.wrapping_add(i as u64 + 1));
        out.extend(pts);
    }
    out
}

fn sample_helium(zeta: f64, n: usize, seed: u64) -> Vec<Vec3> {
    // Iloczyn φ(r₁)φ(r₂): chmura to dwa niezależne elektrony w 1s(ζ).
    let electrons = [
        Occupied {
            n: 1,
            l: 0,
            m: 0,
            z_eff: zeta,
        },
        Occupied {
            n: 1,
            l: 0,
            m: 0,
            z_eff: zeta,
        },
    ];
    sample_atom(&electrons, n, seed)
}

fn shades_for(catalog: &Catalog, positions: &[Vec3], time: f64) -> Vec<f32> {
    match catalog {
        Catalog::Orbital { z, n, l, m, .. } => positions
            .iter()
            .map(|p| {
                let psi = wavefunction(*z, *n, *l, *m, p.x, p.y, p.z);
                lobe_shade(psi)
            })
            .collect(),
        Catalog::Superposition { z, terms, .. } => positions
            .iter()
            .map(|p| {
                let d = superposition_density(*z, terms, *p, time);
                let r = p.norm().max(1e-9);
                let axis = (0.5 + 0.5 * (p.z / r)).clamp(0.0, 1.0);
                (0.25 + 0.55 * axis + 0.15 * (d / (d + 1e-4)).min(1.0)) as f32
            })
            .collect(),
        Catalog::Atom { electrons, .. } => atom_shades(electrons, positions.len()),
        Catalog::Helium { .. } => vec![0.55; positions.len()],
    }
}

fn atom_shades(electrons: &[Occupied], n: usize) -> Vec<f32> {
    let n_max = electrons.iter().map(|e| e.n).max().unwrap_or(1);
    let mut shades = Vec::with_capacity(n);
    if electrons.is_empty() {
        return vec![0.5; n];
    }
    let chunk = (n / electrons.len()).max(1);
    for (i, e) in electrons.iter().enumerate() {
        let start = i * chunk;
        let end = if i + 1 == electrons.len() {
            n
        } else {
            ((i + 1) * chunk).min(n)
        };
        let t = (e.n - 1) as f32 / n_max.max(1) as f32;
        for _ in start..end {
            shades.push(0.22 + 0.72 * t);
        }
    }
    shades.resize(n, 0.5);
    shades
}

fn lobe_shade(psi: f64) -> f32 {
    if psi >= 0.0 {
        0.82
    } else {
        0.28
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm::config::{RunConfig, Term};
    use crate::qm::presets;

    fn tiny(mut cfg: Config) -> Config {
        cfg.run.n_samples = 800;
        cfg.run.sparkle = false;
        cfg
    }

    #[test]
    fn hydrogen_engine_takes_a_step() {
        let mut eng = Engine::new(tiny(Config::default()));
        eng.advance(3).unwrap();
        assert_eq!(eng.state.step, 3);
        assert!(eng.state.is_finite());
        assert_eq!(eng.state.n(), 800);
    }

    #[test]
    fn superposition_density_moves_along_z() {
        let cfg = Config {
            scene: Scene::Superposition {
                z: 1,
                terms: vec![Term::real(1, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
            },
            run: RunConfig {
                n_samples: 1_200,
                sparkle: false,
                dt: 0.4,
                ..RunConfig::default()
            },
        };
        let mut eng = Engine::new(cfg);
        let z0: f64 = eng.state.positions.iter().map(|p| p.z).sum::<f64>() / eng.state.n() as f64;
        // Pół okresu bicia 1s–2p: T = 2π / 0,375 ≈ 16,76; dt=0,4 → ~21 kroków.
        eng.advance(21).unwrap();
        let z1: f64 = eng.state.positions.iter().map(|p| p.z).sum::<f64>() / eng.state.n() as f64;
        assert!(
            (z1 - z0).abs() > 0.15,
            "chmura 1s+2p_z powinna się przesunąć wzdłuż z: {z0} → {z1}"
        );
    }

    #[test]
    fn changing_n_at_runtime_rebuilds_the_cloud() {
        let mut eng = Engine::new(tiny(Config::default()));
        let mut live = eng.cfg.clone();
        live.scene = Scene::Orbital {
            z: 1,
            n: 2,
            l: 1,
            m: 0,
        };
        eng.apply_runtime_config(&live);
        match &eng.cfg.scene {
            Scene::Orbital { n, l, .. } => assert_eq!((*n, *l), (2, 1)),
            other => panic!("{other:?}"),
        }
        let mean = eng.state.positions.iter().map(|p| p.norm()).sum::<f64>()
            / eng.state.n() as f64;
        assert!(mean > 2.0, "2p powinno być szersze od 1s, ⟨r⟩={mean}");
    }

    #[test]
    fn every_preset_compiles_and_steps() {
        for entry in presets::PRESETS {
            let mut cfg = (entry.build)();
            cfg.run.n_samples = 400;
            cfg.run.sparkle = false;
            let mut eng = Engine::new(cfg);
            eng.advance(1).unwrap();
            assert!(eng.state.is_finite(), "{}", entry.id);
            assert!(eng.state.n() >= 400, "{}", entry.id);
        }
    }
}
