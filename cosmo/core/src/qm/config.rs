//! Nastawy modelu atomowego: co jest na scenie i jak gęsto próbkować `|ψ|²`.

use serde::{Deserialize, Serialize};

use crate::qm::elements::{self, Element};
use crate::qm::hydrogen::{self, quantum_ok};

/// Składnik superpozycji. Amplitudy nie muszą być unormowane — silnik znormuje
/// je przy starcie, a suwak „mieszanka" w panelu ustawia stosunek dwóch pierwszych.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Term {
    pub n: u32,
    pub l: u32,
    pub m: i32,
    pub re: f64,
    pub im: f64,
}

impl Term {
    pub fn new(n: u32, l: u32, m: i32, re: f64, im: f64) -> Self {
        Self { n, l, m, re, im }
    }

    pub fn real(n: u32, l: u32, m: i32, weight: f64) -> Self {
        Self::new(n, l, m, weight, 0.0)
    }

    pub fn amplitude_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn label(&self) -> String {
        hydrogen::orbital_label(self.n, self.l, self.m)
    }
}

/// Co jest liczone. Trzy sceny, bo trzy różne pytania.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Scene {
    /// Jeden orbital wodoropodobny — rozwiązanie dokładne.
    Orbital { z: u32, n: u32, l: u32, m: i32 },
    /// Suma `Σ c_k ψ_k e^{−i E_k t}` przy tym samym `Z`. Gęstość bije w czasie,
    /// gdy energie się różnią — to jedyna scena, w której chmura naprawdę żyje.
    Superposition { z: u32, terms: Vec<Term> },
    /// Neutralny atom: elektrony niezależne, `Z_eff` ze Slatera.
    Atom { z: u32 },
}

impl Default for Scene {
    fn default() -> Self {
        Self::Orbital {
            z: 1,
            n: 1,
            l: 0,
            m: 0,
        }
    }
}

impl Scene {
    pub fn z(&self) -> u32 {
        match *self {
            Self::Orbital { z, .. } | Self::Superposition { z, .. } | Self::Atom { z } => z.max(1),
        }
    }

    pub fn element(&self) -> &'static Element {
        elements::nearest(self.z())
    }

    pub fn is_time_dependent(&self) -> bool {
        match self {
            Self::Superposition { terms, .. } => {
                let energies: Vec<u32> = terms.iter().map(|t| t.n).collect();
                energies.windows(2).any(|w| w[0] != w[1])
            }
            Self::Orbital { .. } | Self::Atom { .. } => false,
        }
    }

    pub fn primary_orbital(&self) -> (u32, u32, u32, i32) {
        match *self {
            Self::Orbital { z, n, l, m } => (z, n, l, m),
            Self::Superposition { z, ref terms } => terms
                .first()
                .map(|t| (z, t.n, t.l, t.m))
                .unwrap_or((z, 1, 0, 0)),
            Self::Atom { z } => {
                let el = elements::nearest(z);
                el.subshells
                    .last()
                    .map(|s| (z, s.n, s.l, 0))
                    .unwrap_or((z, 1, 0, 0))
            }
        }
    }

    pub fn label(&self) -> String {
        match self {
            Self::Orbital { z, n, l, m } => {
                let el = elements::nearest(*z);
                format!(
                    "{}{}  {}",
                    el.symbol,
                    if *z == el.z && *z > 1 {
                        format!("⁺{}", z.saturating_sub(1))
                    } else if *z != 1 {
                        format!(" Z={z}")
                    } else {
                        String::new()
                    },
                    hydrogen::orbital_label(*n, *l, *m)
                )
            }
            Self::Superposition { terms, .. } => {
                let parts: Vec<String> = terms.iter().map(Term::label).collect();
                format!("superpozycja {}", parts.join(" + "))
            }
            Self::Atom { z } => {
                let el = elements::nearest(*z);
                format!("{}  {}", el.symbol, el.configuration_text())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    /// Ile punktów losowanych z `|ψ|²`. To rozdzielczość obrazu, nie fizyka.
    pub n_samples: usize,
    pub seed: u64,
    /// Krok czasu w jednostkach atomowych. Ma znaczenie tylko w superpozycji.
    pub dt: f64,
    pub steps: u64,
    pub diagnostics_every: u32,
    pub trajectory_every: u32,
    pub point_stride: usize,
    /// Czy punkty stacjonarnego orbitalu mają delikatnie błądzić (Metropolis).
    /// Gęstość się nie zmienia — zmienia się tylko realizacja tej samej miary.
    pub sparkle: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            n_samples: 12_000,
            seed: 1_927_627,
            dt: 0.25,
            steps: 800,
            diagnostics_every: 4,
            trajectory_every: 4,
            point_stride: 1,
            sparkle: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub scene: Scene,
    pub run: RunConfig,
}

impl Config {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("konfiguracja jest serializowalna")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Popraw niedozwolone liczby kwantowe zamiast liczyć zero. Panel i CLI
    /// mogą podać `l ≥ n` przy ruszaniu suwakiem; milczenie dawałoby pustą chmurę.
    pub fn sanitized(&self) -> Self {
        let mut out = self.clone();
        out.scene = match out.scene {
            Scene::Orbital { z, n, l, m } => {
                let n = n.max(1);
                let l = l.min(n - 1);
                let cap = l as i32;
                let m = m.clamp(-cap, cap);
                Scene::Orbital { z: z.max(1), n, l, m }
            }
            Scene::Superposition { z, terms } => {
                let mut terms: Vec<Term> = terms
                    .into_iter()
                    .filter(|t| quantum_ok(t.n, t.l, t.m) && t.amplitude_sq() > 0.0)
                    .collect();
                if terms.is_empty() {
                    terms.push(Term::real(1, 0, 0, 1.0));
                }
                Scene::Superposition { z: z.max(1), terms }
            }
            Scene::Atom { z } => Scene::Atom {
                z: elements::nearest(z.max(1)).z,
            },
        };
        out.run.n_samples = out.run.n_samples.clamp(64, 200_000);
        out.run.dt = out.run.dt.abs().max(1e-6);
        out
    }

    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();
        match &self.scene {
            Scene::Orbital { n, l, m, .. } => {
                if !quantum_ok(*n, *l, *m) {
                    out.push(format!(
                        "({n},{l},{m}) nie jest dozwolone — l < n i |m| ≤ l; liczby zostaną przycięte"
                    ));
                }
            }
            Scene::Superposition { terms, .. } => {
                if terms.len() < 2 {
                    out.push("superpozycja ma mniej niż dwa stany — gęstość nie będzie bić".into());
                }
                if terms.iter().any(|t| !quantum_ok(t.n, t.l, t.m)) {
                    out.push("któryś składnik ma niedozwolone liczby kwantowe".into());
                }
                if !self.scene.is_time_dependent() && terms.len() >= 2 {
                    out.push(
                        "składniki mają to samo n, więc tę samą energię — chmura jest nieruchoma"
                            .into(),
                    );
                }
            }
            Scene::Atom { z } => {
                if elements::by_z(*z).is_none() {
                    out.push(format!(
                        "brak tablicy dla Z={z} — użyty zostanie najbliższy pierwiastek {}",
                        elements::nearest(*z).symbol
                    ));
                }
                let el = elements::nearest(*z);
                if let Some(err) = elements::ionization_error(el) {
                    if err.abs() > 0.15 {
                        out.push(format!(
                            "Slater na {} myli pierwszą jonizację o {:+.0}% — to granica modelu, nie usterka silnika",
                            el.symbol,
                            100.0 * err
                        ));
                    }
                }
                if *z >= 26 {
                    out.push(
                        "ciężki atom: brak korelacji, wymienności i struktury subtelnej; \
                         obraz pokazuje powłoki, nie chemię"
                            .into(),
                    );
                }
            }
        }
        out
    }

    /// `mix = 0` to sam pierwszy składnik, `1` — sam drugi. Używane przez suwak
    /// w panelu; reszta amplitud zostaje.
    pub fn set_mix(&mut self, mix: f64) {
        let mix = mix.clamp(0.0, 1.0);
        if let Scene::Superposition { terms, .. } = &mut self.scene {
            if terms.len() >= 2 {
                terms[0].re = (1.0 - mix).sqrt();
                terms[0].im = 0.0;
                terms[1].re = mix.sqrt();
                terms[1].im = 0.0;
            }
        }
    }

    pub fn mix(&self) -> f64 {
        match &self.scene {
            Scene::Superposition { terms, .. } if terms.len() >= 2 => {
                let a = terms[0].amplitude_sq();
                let b = terms[1].amplitude_sq();
                let s = a + b;
                if s <= 0.0 {
                    0.5
                } else {
                    b / s
                }
            }
            _ => 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_round_trip_keeps_a_superposition() {
        let cfg = Config {
            scene: Scene::Superposition {
                z: 1,
                terms: vec![Term::real(1, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
            },
            ..Config::default()
        };
        let back = Config::from_json(&cfg.to_json()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn partial_file_fills_defaults() {
        let cfg = Config::from_json(r#"{"run":{"n_samples":512}}"#).unwrap();
        assert_eq!(cfg.run.n_samples, 512);
        assert!(matches!(cfg.scene, Scene::Orbital { n: 1, .. }));
    }

    #[test]
    fn sanitizer_clips_l_and_m() {
        let cfg = Config {
            scene: Scene::Orbital {
                z: 1,
                n: 2,
                l: 5,
                m: 9,
            },
            ..Config::default()
        };
        let s = cfg.sanitized();
        match s.scene {
            Scene::Orbital { n, l, m, .. } => {
                assert_eq!((n, l, m), (2, 1, 1));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn superposition_of_n1_and_n2_is_time_dependent() {
        let scene = Scene::Superposition {
            z: 1,
            terms: vec![Term::real(1, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
        };
        assert!(scene.is_time_dependent());
        let same = Scene::Superposition {
            z: 1,
            terms: vec![Term::real(2, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
        };
        assert!(!same.is_time_dependent());
    }

    #[test]
    fn atom_without_a_table_entry_is_announced() {
        let cfg = Config {
            scene: Scene::Atom { z: 40 },
            ..Config::default()
        };
        assert!(cfg.warnings().iter().any(|w| w.contains("Z=40")));
    }
}
