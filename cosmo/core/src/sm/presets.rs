//! Nazwane zestawy nastaw modelu cząstek — jedno źródło dla CLI i panelu.
//!
//! Każdy preset to hipoteza fizyczna, nie zestaw ładnych domyślnych. Komentarze
//! podają, skąd wzięła się każda liczba: zmiana jednej z nich potrafi unieważnić
//! całe zjawisko, po które zestaw został złożony.

use crate::sm::config::{
    Config, DecayConfig, ForceConfig, Ingredient, RunConfig, Shape, SpawnConfig,
};
use crate::sm::particles::{Particle, Species};
use crate::sr::config::BackendKind;

/// Nazwany zestaw parametrów.
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub build: fn() -> Config,
}

/// Presety w kolejności wyświetlania.
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "plazma",
        label: "Plazma",
        build: plazma,
    },
    Preset {
        id: "para",
        label: "Para",
        build: para,
    },
    Preset {
        id: "miony",
        label: "Miony",
        build: miony,
    },
    Preset {
        id: "anihilacja",
        label: "Anihilacja",
        build: anihilacja,
    },
    Preset {
        id: "piony",
        label: "Piony",
        build: piony,
    },
    Preset {
        id: "uwiezienie",
        label: "Uwięzienie",
        build: uwiezienie,
    },
];

pub fn preset(id: &str) -> Option<Config> {
    PRESETS.iter().find(|p| p.id == id).map(|p| (p.build)())
}

pub fn ids() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.id).collect()
}

/// Neutralna plazma elektron–proton: ekranowanie Debye'a, nie rozlot.
///
/// Promień `5·10⁴ fm` (50 pm) jest duży wobec zmiękczenia i mały wobec skali
/// atomowej — klasyczny gaz Coulombowski, nie atom. Temperatura `1 keV` jest
/// dużo poniżej masy elektronu, więc rozkład Maxwella jest tu uprawniony.
pub fn plazma() -> Config {
    Config::default()
}

/// Elektron i proton naprzeciw siebie — najprostszy układ, w którym widać siłę.
///
/// Odległość `5·10⁴ fm` i solver dokładny: dryf energii ma prawo być mały,
/// a znak siły — oczywisty gołym okiem.
pub fn para() -> Config {
    let separation = 5.0e4;
    Config {
        spawn: SpawnConfig {
            shape: Shape::Pair,
            mixture: vec![
                Ingredient::new(Particle::of(Species::Electron), 1),
                Ingredient::new(Particle::of(Species::Proton), 1),
            ],
            radius: separation / 2.0,
            temperature: 0.0,
            beam_energy: 0.0,
            seed: 1,
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
            steps: 2_000,
            ..RunConfig::default()
        },
    }
}

/// Swobodna wiązka mionów — rozpad z dylatacją czasu.
///
/// Sił nie ma: na skali czasu życia mionu (`6,6·10¹⁷ fm/c`) cząstki są od siebie
/// o kilometry. Adaptacja musi być wyłączona, bo przycięłaby krok do skali sił,
/// których tu nie ma. Energia wiązki `10 GeV` daje `γ ≈ 95`, więc mion żyje
/// stokrotnie dłużej niż w spoczynku — i to nie jest dołożone ręcznie.
pub fn miony() -> Config {
    let muon = Particle::of(Species::Muon);
    let lifetime = muon.lifetime_fm().expect("mion ma zmierzony czas życia");
    Config {
        spawn: SpawnConfig {
            shape: Shape::Beam,
            mixture: vec![Ingredient::new(muon, 400)],
            radius: 1.0e3,
            temperature: 0.0,
            beam_energy: 10_000.0,
            seed: 8,
        },
        forces: ForceConfig {
            coulomb: false,
            ..ForceConfig::default()
        },
        decay: DecayConfig::default(),
        run: RunConfig {
            dt: lifetime / 5.0,
            adaptive: false,
            steps: 40,
            max_particles: 100_000,
            ..RunConfig::default()
        },
    }
}

/// Chmura par elektron–pozyton: anihilacja do fotonów.
///
/// Promień `200 fm` jest większy od `pair_radius`, więc anihilacja nie zjada
/// wszystkiego w pierwszym kroku — widać ubywanie par, a nie błysk.
pub fn anihilacja() -> Config {
    Config {
        spawn: SpawnConfig {
            shape: Shape::Ball,
            mixture: vec![
                Ingredient::new(Particle::of(Species::Electron), 200),
                Ingredient::new(Particle::anti_of(Species::Electron), 200),
            ],
            radius: 200.0,
            temperature: 1.0e-3,
            beam_energy: 0.0,
            seed: 11,
        },
        forces: ForceConfig {
            coulomb: true,
            softening: 0.5,
            backend: BackendKind::Exact,
            ..ForceConfig::default()
        },
        decay: DecayConfig {
            enabled: false,
            annihilation: true,
            pair_radius: 50.0,
            ..DecayConfig::default()
        },
        run: RunConfig {
            dt: 1.0,
            adaptive: true,
            steps: 400,
            ..RunConfig::default()
        },
    }
}

/// Piony obojętne: masa spoczynkowa zamienia się w ruch fotonów.
///
/// Krok równy czasowi życia — w kilku krokach widać kaskadę. Adaptacja jest
/// wyłączona z tego samego powodu co przy mionach: badamy rozpad, nie siły.
pub fn piony() -> Config {
    let pion = Particle::of(Species::PionNeutral);
    let lifetime = pion.lifetime_fm().expect("π⁰ ma zmierzony czas życia");
    Config {
        spawn: SpawnConfig {
            shape: Shape::Ball,
            mixture: vec![Ingredient::new(pion, 80)],
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
            dt: lifetime,
            adaptive: false,
            steps: 20,
            max_particles: 10_000,
            ..RunConfig::default()
        },
    }
}

/// Para kwark–antykwark: liniowy człon potencjału Cornella nie puszcza.
///
/// `300 MeV` energii kinetycznej przy napięciu `~0,9 GeV/fm` wystarcza na
/// ułamek femtometra rozciągnięcia i ani femtometra więcej. To jedyny preset,
/// w którym oddziaływanie silne robi coś, czego żadne inne nie potrafi.
pub fn uwiezienie() -> Config {
    Config {
        spawn: SpawnConfig {
            shape: Shape::Pair,
            mixture: vec![
                Ingredient::new(Particle::of(Species::Up), 1),
                Ingredient::new(Particle::anti_of(Species::Up), 1),
            ],
            radius: 0.5,
            temperature: 0.0,
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
            steps: 4_000,
            ..RunConfig::default()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::engine::Engine;

    #[test]
    fn every_preset_has_id_label_and_builds() {
        for entry in PRESETS {
            assert!(!entry.id.is_empty());
            assert!(!entry.label.is_empty());
            let cfg = (entry.build)();
            assert!(cfg.spawn.total_count() > 0, "{}", entry.id);
            assert_eq!(preset(entry.id).unwrap(), cfg);
        }
    }

    #[test]
    fn ids_match_the_table() {
        assert_eq!(
            ids(),
            ["plazma", "para", "miony", "anihilacja", "piony", "uwiezienie"]
        );
        assert!(preset("nie-ma").is_none());
    }

    /// Każdy zestaw ma dać się złożyć i zrobić krok — inaczej panel startowałby
    /// w błąd, o którym nikt by nie wiedział przed kliknięciem.
    #[test]
    fn every_preset_takes_a_step() {
        for entry in PRESETS {
            let mut engine = Engine::new((entry.build)());
            engine.advance(1).unwrap();
            assert!(
                engine.state.is_finite(),
                "{} rozbiegł się w pierwszym kroku",
                entry.id
            );
        }
    }

    #[test]
    fn plazma_is_neutral_and_uses_coulomb() {
        let cfg = plazma();
        assert_eq!(cfg.spawn.net_charge_thirds(), 0);
        assert!(cfg.forces.coulomb);
        assert!(!cfg.forces.strong);
    }

    #[test]
    fn muon_run_is_on_the_lifetime_scale() {
        let cfg = miony();
        assert!(!cfg.forces.coulomb);
        assert!(!cfg.run.adaptive);
        let horizon = cfg.run.dt * cfg.run.steps as f64;
        let lifetime = Species::Muon.lifetime_fm().unwrap();
        assert!(horizon > lifetime, "bieg krótszy od czasu życia mionu");
    }

    #[test]
    fn confinement_is_the_only_preset_with_color() {
        for entry in PRESETS {
            let cfg = (entry.build)();
            let colored = cfg
                .spawn
                .mixture
                .iter()
                .any(|i| i.particle.colored() && i.count > 0);
            assert_eq!(colored, entry.id == "uwiezienie", "{}", entry.id);
        }
    }
}
