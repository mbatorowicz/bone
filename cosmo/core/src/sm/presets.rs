//! Nazwane zestawy nastaw modelu cząstek — jedno źródło dla CLI i panelu.
//!
//! Każdy preset to jedno laboratorium, nie galeria gatunków. Komentarze podają,
//! skąd wzięła się każda liczba: zmiana jednej z nich potrafi unieważnić całe
//! zjawisko, po które zestaw został złożony.

use crate::sm::config::{
    Config, DecayConfig, ForceConfig, Ingredient, RunConfig, Shape, SpawnConfig,
};
use crate::sm::particles::{Particle, Species};
use crate::sr::config::BackendKind;

/// Cztery linie karty nad suwakami: równanie, zakres, porównanie, czym to nie jest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LabCard {
    pub equation: &'static str,
    pub scope: &'static str,
    pub comparison: &'static str,
    pub not_this: &'static str,
}

/// Nazwany zestaw parametrów.
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub card: LabCard,
    pub build: fn() -> Config,
}

/// Presety w kolejności wyświetlania — jedno zjawisko na laboratorium.
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "plazma",
        label: "Plazma",
        card: LabCard {
            equation: "Coulomb · e⁻p",
            scope: "klasyczny gaz, Debye",
            comparison: "λ_D, quasi-neutralność",
            not_this: "nie QFT, nie atom",
        },
        build: plazma,
    },
    Preset {
        id: "para",
        label: "Para",
        card: LabCard {
            equation: "F ∝ q₁q₂ / r²",
            scope: "e⁻ i p, znak siły",
            comparison: "przyciąganie vs odpychanie",
            not_this: "nie przekrój dσ/dΩ",
        },
        build: para,
    },
    Preset {
        id: "miony",
        label: "Miony",
        card: LabCard {
            equation: "τ_lab = γτ",
            scope: "wiązka, rozpady PDG",
            comparison: "czas życia mionu",
            not_this: "nie amplitudy",
        },
        build: miony,
    },
    Preset {
        id: "piony",
        label: "Piony",
        card: LabCard {
            equation: "π⁰ → γγ · E₁+E₂ = m",
            scope: "rozpad dwuciałowy",
            comparison: "masa spoczynkowa w fotonach",
            not_this: "nie hadronizacja",
        },
        build: piony,
    },
    Preset {
        id: "anihilacja",
        label: "Anihilacja",
        card: LabCard {
            equation: "e⁺e⁻ → γγ, czteropęd",
            scope: "klasyczne tory + stochastyka",
            comparison: "B, L, Q całkowite",
            not_this: "nie QED",
        },
        build: anihilacja,
    },
    Preset {
        id: "uwiezienie",
        label: "Uwięzienie",
        card: LabCard {
            equation: "V = −κ/r + σr",
            scope: "jedna para qq̄, Cornell",
            comparison: "fenomenologia 1977",
            not_this: "struna nie pęka",
        },
        build: uwiezienie,
    },
];

pub fn preset(id: &str) -> Option<Config> {
    PRESETS.iter().find(|p| p.id == id).map(|p| (p.build)())
}

pub fn ids() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.id).collect()
}

/// Karta laboratorium; nieznane id dostaje kartę plazmy, bo to zestaw domyślny.
pub fn lab_card(id: &str) -> LabCard {
    PRESETS
        .iter()
        .find(|p| p.id == id)
        .map(|p| p.card)
        .unwrap_or(PRESETS[0].card)
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
            enabled: true,
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
            coulomb: false,
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
            assert!(!entry.card.equation.is_empty());
            assert!(!entry.card.not_this.is_empty());
            let cfg = (entry.build)();
            assert!(cfg.spawn.total_count() > 0, "{}", entry.id);
            assert_eq!(preset(entry.id).unwrap(), cfg);
        }
    }

    #[test]
    fn ids_match_the_table() {
        assert_eq!(
            ids(),
            ["plazma", "para", "miony", "piony", "anihilacja", "uwiezienie"]
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

    /// Żadne laboratorium nie stawia W, Z, H, t ani gluonu. Zostają w katalogu mas.
    #[test]
    fn no_preset_spawns_heavy_bosons_top_or_gluons() {
        for entry in PRESETS {
            let cfg = (entry.build)();
            for ingredient in &cfg.spawn.mixture {
                assert!(
                    !ingredient.particle.species.catalog_only(),
                    "{} spawnuje {}",
                    entry.id,
                    ingredient.particle
                );
            }
            let spawned = crate::sm::spawn::make_state(&cfg);
            assert!(
                spawned
                    .state
                    .kinds
                    .iter()
                    .all(|p| !p.species.catalog_only()),
                "{}",
                entry.id
            );
        }
    }

    /// Kwarki tylko w uwięzieniu, i to dokładnie jedna para.
    #[test]
    fn quarks_appear_only_as_one_confinement_pair() {
        for entry in PRESETS {
            let cfg = (entry.build)();
            let quarks: usize = cfg
                .spawn
                .mixture
                .iter()
                .filter(|i| i.particle.species.confinement_only())
                .map(|i| i.count)
                .sum();
            if entry.id == "uwiezienie" {
                assert_eq!(quarks, 2, "uwięzienie ma być jedną parą");
                assert_eq!(cfg.spawn.total_count(), 2);
                assert!(cfg.forces.strong);
                assert!(!cfg.forces.coulomb);
            } else {
                assert_eq!(quarks, 0, "{} nie powinno mieć kwarków", entry.id);
                assert!(!cfg.forces.strong);
            }
        }
    }

    /// Karta zależy od laboratorium, nie od „czterech oddziaływań".
    #[test]
    fn each_lab_has_its_own_card() {
        let muon = lab_card("miony");
        assert_eq!(muon.equation, "τ_lab = γτ");
        let pion = lab_card("piony");
        assert!(pion.equation.contains("E₁+E₂ = m"));
        let annihil = lab_card("anihilacja");
        assert!(annihil.not_this.contains("QED"));
        let confine = lab_card("uwiezienie");
        assert!(confine.comparison.contains("1977"));
        assert!(confine.not_this.contains("struna"));
        let pair = lab_card("para");
        assert!(pair.scope.contains("znak siły"));
        assert_eq!(lab_card("nie-ma").equation, lab_card("plazma").equation);
    }

    /// Wiązka mionów: γ z energii wiązki, a krok jest ułamkiem czasu własnego,
    /// więc τ_lab = γτ nie jest wpisane ręcznie — wychodzi z kinematyki.
    #[test]
    fn muon_beam_gamma_sets_the_lab_lifetime() {
        let cfg = miony();
        let mass = Species::Muon.mass();
        let rest = Species::Muon.lifetime_fm().unwrap();
        let gamma = (mass + cfg.spawn.beam_energy) / mass;
        assert!((gamma - 95.6).abs() < 1.0, "γ = {gamma}");
        assert!((cfg.run.dt - rest / 5.0).abs() / rest < 1e-12);

        let spawned = crate::sm::spawn::make_state(&cfg);
        let measured = crate::sm::kinematics::gamma(mass, spawned.state.momenta[0]).unwrap();
        assert!((measured - gamma).abs() / gamma < 1e-9, "γ ze stanu {measured}");
    }

    /// Anihilacja laboratorium: czteropęd pary e⁺e⁻ przechodzi w dwa fotony.
    #[test]
    fn annihilation_lab_conserves_four_momentum() {
        let cfg = anihilacja();
        assert!(cfg.decay.annihilation);
        assert!(cfg.decay.enabled, "warstwa stochastyczna musi być włączona, żeby para zniknęła");
        let electron = Particle::of(Species::Electron);
        let mut state = crate::sm::state::State::new(
            vec![crate::vec3::ZERO, crate::vec3::vec3(0.1, 0.0, 0.0)],
            vec![
                crate::vec3::vec3(0.0, 0.001, 0.0),
                crate::vec3::vec3(0.0, -0.001, 0.0),
            ],
            vec![electron, electron.conjugate()],
        )
        .unwrap();
        let before = state.total_four_momentum();
        let mut decays = crate::sm::decays::Decays::new(DecayConfig {
            pair_radius: 1.0,
            ..cfg.decay
        });
        let outcome = decays.step(&mut state, 1.0e6, 100);
        assert_eq!(outcome.annihilated, 1);
        assert_eq!(state.count_of(Particle::of(Species::Photon)), 2);
        let after = state.total_four_momentum();
        assert!(
            (after.energy - before.energy).abs() / before.energy < 1e-12,
            "energia {} → {}",
            before.energy,
            after.energy
        );
        assert!(
            (after.momentum - before.momentum).norm() / before.energy < 1e-12,
            "pęd {:?} → {:?}",
            before.momentum,
            after.momentum
        );
    }

    #[test]
    fn pion_lab_is_a_decay_not_a_force() {
        let cfg = piony();
        assert!(!cfg.forces.coulomb);
        assert!(!cfg.run.adaptive);
        assert_eq!(
            cfg.spawn.mixture[0].particle.species,
            Species::PionNeutral
        );
    }
}
