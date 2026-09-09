//! Nazwane zestawy nastaw modelu atomowego — jedno źródło dla CLI i panelu.
//!
//! Każdy preset jest hipotezą albo demonstracją, nie galerią ładnych chmur.
//! `superpozycja` istnieje po to, żeby było widać bicie 1s+2p; `hel` — po to,
//! żeby wariacja dała ~2% błędu; `ekranowanie` — po to, żeby Slaterowski błąd
//! IE stał obok obrazka powłok.

use crate::qm::config::{Config, RunConfig, Scene, Term};

pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub build: fn() -> Config,
}

pub const PRESETS: &[Preset] = &[
    Preset { id: "wodor", label: "Wodór 1s", build: wodor },
    Preset { id: "orbital_2p", label: "2p_z", build: orbital_2p },
    Preset { id: "orbital_3d", label: "3d_z²", build: orbital_3d },
    Preset { id: "rydberg", label: "Rydberg", build: rydberg },
    Preset { id: "superpozycja", label: "1s+2p", build: superpozycja },
    Preset { id: "hel_plus", label: "He⁺", build: hel_plus },
    Preset { id: "hel", label: "Hel (wariacja)", build: hel },
    Preset { id: "ekranowanie", label: "Ekranowanie", build: ekranowanie },
];

pub fn preset(id: &str) -> Option<Config> {
    PRESETS.iter().find(|p| p.id == id).map(|p| (p.build)())
}

pub fn ids() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.id).collect()
}

fn run_cloud() -> RunConfig {
    RunConfig {
        n_samples: 14_000,
        sparkle: true,
        dt: 0.25,
        steps: 400,
        ..RunConfig::default()
    }
}

/// Wodór w stanie podstawowym — jedyny atom, którego `1s` jest dokładny.
pub fn wodor() -> Config {
    Config {
        scene: Scene::Orbital {
            z: 1,
            n: 1,
            l: 0,
            m: 0,
        },
        run: run_cloud(),
    }
}

/// `2p_z`: dwa płaty, węzeł na równiku, znak funkcji falowej w odcieniu.
pub fn orbital_2p() -> Config {
    Config {
        scene: Scene::Orbital {
            z: 1,
            n: 2,
            l: 1,
            m: 0,
        },
        run: run_cloud(),
    }
}

pub fn orbital_3d() -> Config {
    Config {
        scene: Scene::Orbital {
            z: 1,
            n: 3,
            l: 2,
            m: 0,
        },
        run: run_cloud(),
    }
}

/// `n = 8`, `l = 1` — elektron daleko od jądra, skala ~ n².
pub fn rydberg() -> Config {
    Config {
        scene: Scene::Orbital {
            z: 1,
            n: 8,
            l: 1,
            m: 0,
        },
        run: RunConfig {
            n_samples: 16_000,
            ..run_cloud()
        },
    }
}

/// Równa mieszanka `1s` i `2p_z`. Gęstość oscyluje wzdłuż osi z — to nie jest
/// klasyczna orbita, tylko interferencja dwóch stacjonarnych stanów.
pub fn superpozycja() -> Config {
    Config {
        scene: Scene::Superposition {
            z: 1,
            terms: vec![Term::real(1, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
        },
        run: RunConfig {
            dt: 0.35,
            n_samples: 10_000,
            sparkle: false,
            steps: 200,
            diagnostics_every: 2,
            trajectory_every: 2,
            ..run_cloud()
        },
    }
}

/// He⁺ — nadal jeden elektron, więc nadal rozwiązanie dokładne, tylko `Z = 2`.
pub fn hel_plus() -> Config {
    Config {
        scene: Scene::Orbital {
            z: 2,
            n: 1,
            l: 0,
            m: 0,
        },
        run: run_cloud(),
    }
}

/// Hel: wariacja `ζ = 27/16`, nie Koopmans+Slater.
pub fn hel() -> Config {
    Config {
        scene: Scene::Helium,
        run: RunConfig {
            n_samples: 16_000,
            sparkle: false,
            ..run_cloud()
        },
    }
}

/// Slater na węglu: błąd IE jest tematem karty.
pub fn ekranowanie() -> Config {
    Config {
        scene: Scene::Atom { z: 6 },
        run: RunConfig {
            n_samples: 16_000,
            sparkle: false,
            ..run_cloud()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_has_id_label_and_builds() {
        for entry in PRESETS {
            assert!(!entry.id.is_empty());
            assert!(!entry.label.is_empty());
            let cfg = (entry.build)();
            assert!(cfg.run.n_samples >= 64, "{}", entry.id);
            assert_eq!(preset(entry.id).unwrap().scene, cfg.scene);
        }
        assert!(preset("nie-ma").is_none());
        assert_eq!(ids()[0], "wodor");
        assert!(!ids().iter().any(|id| *id == "zelazo" || *id == "wegiel"));
    }

    #[test]
    fn superposition_is_the_only_time_dependent_preset() {
        for entry in PRESETS {
            let cfg = (entry.build)();
            assert_eq!(
                cfg.scene.is_time_dependent(),
                entry.id == "superpozycja",
                "{}",
                entry.id
            );
        }
    }

    #[test]
    fn helium_is_variational_and_hydrogen_is_exact() {
        assert!(matches!(hel().scene, Scene::Helium));
        assert!(matches!(ekranowanie().scene, Scene::Atom { z: 6 }));
        assert!(matches!(wodor().scene, Scene::Orbital { z: 1, n: 1, .. }));
        assert!(matches!(hel_plus().scene, Scene::Orbital { z: 2, .. }));
    }

    #[test]
    fn no_preset_advertises_iron_ionization() {
        for entry in PRESETS {
            let cfg = (entry.build)();
            if let Scene::Atom { z } = cfg.scene {
                let el = crate::qm::elements::nearest(z);
                assert!(
                    crate::qm::elements::reports_ionization(el),
                    "{} reklamuje ciężki atom {}",
                    entry.id,
                    el.symbol
                );
            }
        }
    }
}
