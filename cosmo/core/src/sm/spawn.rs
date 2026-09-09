//! Warunek początkowy: gdzie postawić cząstki i jaki dać im pęd.
//!
//! # Rozkład pędów zależy od masy, a nie od jednego wzoru
//!
//! „Temperatura" zadaje skalę energii kinetycznej, ale przełożenie jej na pęd nie
//! jest jedno dla wszystkich cząstek:
//!
//! - **cząstka masywna** dostaje pędy z rozkładu Maxwella-Boltzmanna, czyli trzy
//!   niezależne gaussy o `σ = √(mT)`. Wtedy `⟨T_kin⟩ = 3T/2` dokładnie;
//! - **cząstka bezmasowa** nie ma masy, przez którą można by pomnożyć, więc pęd
//!   losujemy wprost z rozkładu `∝ p² e^{−p/T}` (suma trzech zmiennych
//!   wykładniczych). Wtedy `⟨E⟩ = 3T`, co jest poprawnym wynikiem relatywistycznym.
//!
//! Ograniczenie wypowiedziane wprost: dla `T` porównywalnego z masą właściwym
//! rozkładem jest Maxwell-Jüttner, a nie gauss. Zamiast go po cichu podmieniać,
//! [`make_state`] **ostrzega**, że warunek początkowy jest wtedy przybliżony.

use crate::rng::Rng;
use crate::sm::config::{Config, Shape};
use crate::sm::kinematics;
use crate::sm::particles::Particle;
use crate::sm::state::State;
use crate::vec3::{vec3, Vec3, ZERO};

/// Powyżej tego ilorazu `T/m` rozkład gaussowski przestaje być Maxwellem.
const RELATIVISTIC_TEMPERATURE: f64 = 0.1;

pub struct Spawned {
    pub state: State,
    pub warnings: Vec<String>,
}

/// Zbuduj stan początkowy z konfiguracji.
pub fn make_state(cfg: &Config) -> Spawned {
    let mut rng = Rng::seeded(cfg.spawn.seed);
    let mut warnings = Vec::new();

    let kinds: Vec<Particle> = spawnable_kinds(cfg, &mut warnings);
    let n = kinds.len();
    if n == 0 {
        return Spawned {
            state: State::empty(),
            warnings,
        };
    }

    let radius = cfg.spawn.radius.max(0.0);
    let temperature = cfg.spawn.temperature.max(0.0);

    let mut positions = Vec::with_capacity(n);
    let mut momenta = Vec::with_capacity(n);

    for (index, kind) in kinds.iter().enumerate() {
        let thermal = thermal_momentum(&mut rng, *kind, temperature);
        let (position, directed) = match cfg.spawn.shape {
            Shape::Ball => (in_ball(&mut rng, radius), ZERO),
            Shape::Beam => (
                in_beam(&mut rng, radius),
                kinematics::momentum_from_kinetic(
                    kind.mass(),
                    cfg.spawn.beam_energy,
                    vec3(0.0, 0.0, 1.0),
                ),
            ),
            Shape::Pair => {
                // Naprzemiennie w lewej i prawej gromadzie, pędem do siebie.
                let side = if index % 2 == 0 { -1.0 } else { 1.0 };
                let centre = vec3(side * radius, 0.0, 0.0);
                let spread = in_ball(&mut rng, radius * 0.1);
                (
                    centre + spread,
                    kinematics::momentum_from_kinetic(
                        kind.mass(),
                        cfg.spawn.beam_energy,
                        vec3(-side, 0.0, 0.0),
                    ),
                )
            }
        };
        positions.push(position);
        momenta.push(thermal + directed);
    }

    // Chmura w spoczynku nie ma dryfować przez kadr. Odjęcie średniego pędu jest
    // standardowe i celowo NIE jest robione dla wiązki: tam pęd wypadkowy jest
    // treścią warunku początkowego, a nie skutkiem ubocznym losowania.
    if cfg.spawn.shape == Shape::Ball {
        let drift: Vec3 = momenta.iter().copied().sum::<Vec3>() / n as f64;
        for momentum in &mut momenta {
            *momentum -= drift;
        }
    }

    warn_about_relativistic_temperature(&kinds, temperature, &mut warnings);

    let state = State::new(positions, momenta, kinds).expect("trzy tablice o tej samej długości");
    Spawned { state, warnings }
}

/// W, Z, H, t i gluony zostają w katalogu. Kwarki — tylko przy Cornelli.
fn spawnable_kinds(cfg: &Config, warnings: &mut Vec<String>) -> Vec<Particle> {
    let strong = cfg.forces.strong;
    let mut dropped: Vec<&'static str> = Vec::new();
    let kinds: Vec<Particle> = cfg
        .spawn
        .mixture
        .iter()
        .flat_map(|ingredient| std::iter::repeat_n(ingredient.particle, ingredient.count))
        .filter(|particle| {
            if particle.species.may_spawn(strong) {
                true
            } else {
                dropped.push(particle.data().name);
                false
            }
        })
        .collect();
    dropped.sort_unstable();
    dropped.dedup();
    if !dropped.is_empty() {
        warnings.push(format!(
            "poza gazem, nie postawiono: {} — W, Z, H, t i gluony zostają w katalogu; \
             kwarki tylko w uwięzieniu",
            dropped.join(", ")
        ));
    }
    kinds
}

/// Pęd termiczny właściwy dla masy cząstki.
fn thermal_momentum(rng: &mut Rng, particle: Particle, temperature: f64) -> Vec3 {
    if temperature <= 0.0 {
        return ZERO;
    }
    let mass = particle.mass();
    if mass > 0.0 {
        // Maxwell-Boltzmann: trzy niezależne gaussy o σ = √(mT).
        return rng.normal_vec(0.0, (mass * temperature).sqrt());
    }
    // Bezmasowa: |p| z rozkładu Γ(3, T) = suma trzech zmiennych wykładniczych,
    // czyli dokładnie `∝ p²e^{−p/T}`. Trzy logarytmy zamiast losowania
    // z odrzuceniem — bez pętli i bez zależności wyniku od strumienia losowego.
    let product = (0..3)
        .map(|_| rng.unit().max(f64::MIN_POSITIVE))
        .product::<f64>();
    rng.unit_vector() * (-temperature * product.ln())
}

/// Punkt jednostajnie w kuli.
///
/// Promień skalowany jak `u^{1/3}`, a nie jednostajnie — bez tego objętość przy
/// środku byłaby przereprezentowana i chmura miałaby jądro, którego nie zamówiono.
fn in_ball(rng: &mut Rng, radius: f64) -> Vec3 {
    rng.unit_vector() * (radius * rng.unit().cbrt())
}

/// Punkt w wiązce: przekrój kołowy o promieniu `radius`, rozciągnięty wzdłuż `z`.
fn in_beam(rng: &mut Rng, radius: f64) -> Vec3 {
    let angle = std::f64::consts::TAU * rng.unit();
    let r = radius * rng.unit().sqrt();
    vec3(
        r * angle.cos(),
        r * angle.sin(),
        rng.uniform(-radius, radius) * 4.0,
    )
}

fn warn_about_relativistic_temperature(
    kinds: &[Particle],
    temperature: f64,
    warnings: &mut Vec<String>,
) {
    if temperature <= 0.0 {
        return;
    }
    let mut hot: Vec<&'static str> = kinds
        .iter()
        .filter(|k| k.mass() > 0.0 && temperature > RELATIVISTIC_TEMPERATURE * k.mass())
        .map(|k| k.data().name)
        .collect();
    hot.sort_unstable();
    hot.dedup();
    if hot.is_empty() {
        return;
    }
    warnings.push(format!(
        "temperatura {temperature:.3} MeV jest porównywalna z masą: {} — rozkład startowy \
         jest gaussowski, a powinien być Maxwella-Jüttnera",
        hot.join(", ")
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::config::{Ingredient, SpawnConfig};
    use crate::sm::particles::Species;

    fn config(shape: Shape, mixture: Vec<Ingredient>) -> Config {
        Config {
            spawn: SpawnConfig {
                shape,
                mixture,
                radius: 100.0,
                temperature: 0.0,
                beam_energy: 0.0,
                seed: 99,
            },
            ..Config::default()
        }
    }

    fn electrons(count: usize) -> Vec<Ingredient> {
        vec![Ingredient::new(Particle::of(Species::Electron), count)]
    }

    #[test]
    fn the_mixture_becomes_the_particle_list() {
        let cfg = config(
            Shape::Ball,
            vec![
                Ingredient::new(Particle::of(Species::Electron), 30),
                Ingredient::new(Particle::of(Species::Proton), 20),
            ],
        );
        let state = make_state(&cfg).state;
        assert_eq!(state.n(), 50);
        assert_eq!(state.count_of(Particle::of(Species::Electron)), 30);
        assert_eq!(state.count_of(Particle::of(Species::Proton)), 20);
        // Ładunek mieszanki liczony wprost musi zgadzać się z zapowiedzią z konfiguracji.
        assert_eq!(
            state.total_charge_thirds(),
            cfg.spawn.net_charge_thirds()
        );
    }

    #[test]
    fn an_empty_mixture_gives_an_empty_state() {
        let state = make_state(&config(Shape::Ball, Vec::new())).state;
        assert!(state.is_empty());
    }

    /// Kula musi być wypełniona jednorodnie. Skalowanie `u^{1/3}` jest tym, co
    /// odróżnia jednorodną kulę od chmury z gęstym jądrem — a różnicy nie widać
    /// na obrazie, bo obie wyglądają jak kula.
    #[test]
    fn the_ball_is_filled_uniformly() {
        let mut cfg = config(Shape::Ball, electrons(20_000));
        cfg.spawn.radius = 10.0;
        let state = make_state(&cfg).state;

        // W jednorodnej kuli połowa masy leży wewnątrz promienia `R/∛2 ≈ 0,794R`.
        let half = 10.0 / 2.0f64.cbrt();
        let inside = state
            .positions
            .iter()
            .filter(|p| p.norm() <= half)
            .count() as f64
            / state.n() as f64;
        assert!(
            (inside - 0.5).abs() < 0.02,
            "wewnątrz promienia połowy objętości jest {inside} cząstek"
        );
        assert!(state.positions.iter().all(|p| p.norm() <= 10.0 + 1e-9));
    }

    /// Średnia energia kinetyczna cząstki masywnej to `3T/2`. To sprawdza jednocześnie
    /// dobór `σ` i to, że rozkład jest gaussowski w PĘDZIE, a nie w energii.
    #[test]
    fn massive_particles_get_the_maxwell_boltzmann_energy() {
        let mut cfg = config(Shape::Ball, electrons(40_000));
        cfg.spawn.temperature = 1e-5;
        let state = make_state(&cfg).state;

        let mean: f64 =
            (0..state.n()).map(|i| state.kinetic_energy(i)).sum::<f64>() / state.n() as f64;
        let expected = 1.5 * cfg.spawn.temperature;
        assert!(
            (mean / expected - 1.0).abs() < 0.03,
            "⟨T⟩ = {mean}, a ma być {expected}"
        );
    }

    /// Dla cząstki bezmasowej poprawną odpowiedzią jest `⟨E⟩ = 3T`, a nie `3T/2`.
    /// Użycie jednego wzoru dla obu przypadków dałoby fotony dwa razy za zimne.
    #[test]
    fn massless_particles_get_the_relativistic_energy() {
        let mut cfg = config(
            Shape::Ball,
            vec![Ingredient::new(Particle::of(Species::Photon), 40_000)],
        );
        cfg.spawn.temperature = 2.0;
        let state = make_state(&cfg).state;

        let mean: f64 = (0..state.n()).map(|i| state.energy(i)).sum::<f64>() / state.n() as f64;
        assert!(
            (mean / (3.0 * cfg.spawn.temperature) - 1.0).abs() < 0.03,
            "⟨E⟩ = {mean}, a ma być {}",
            3.0 * cfg.spawn.temperature
        );
        // Każdy foton leci dokładnie z c — niezależnie od tego, ile ma energii.
        for i in 0..state.n() {
            assert!((state.beta(i) - 1.0).abs() < 1e-12);
        }
    }

    /// Chmura w spoczynku nie dryfuje. Bez odjęcia średniej losowanie zostawiłoby
    /// pęd wypadkowy rzędu `√N`, czyli powolny odjazd całego układu z kadru.
    #[test]
    fn a_ball_has_no_net_momentum() {
        let mut cfg = config(Shape::Ball, electrons(5_000));
        cfg.spawn.temperature = 1e-4;
        let state = make_state(&cfg).state;

        let total = state.total_four_momentum().momentum.norm();
        let scale: f64 = state.momenta.iter().map(|p| p.norm()).sum();
        assert!(total / scale < 1e-12, "pęd wypadkowy {total} przy skali {scale}");
    }

    /// Wiązka ma lecieć, więc jej pędu wypadkowego zerować NIE wolno.
    #[test]
    fn a_beam_keeps_its_momentum_and_direction() {
        let mut cfg = config(
            Shape::Beam,
            vec![Ingredient::new(Particle::of(Species::Muon), 500)],
        );
        cfg.spawn.beam_energy = 1_000.0;
        let state = make_state(&cfg).state;

        assert!(state.momenta.iter().all(|p| p.z > 0.0), "nie wszystkie lecą w +z");
        let gamma_expected = (Species::Muon.mass() + 1_000.0) / Species::Muon.mass();
        for i in 0..state.n() {
            let gamma = state.energy(i) / Species::Muon.mass();
            assert!((gamma - gamma_expected).abs() / gamma_expected < 1e-9, "γ = {gamma}");
        }
        assert!(state.total_four_momentum().momentum.z > 0.0);
    }

    /// Dwie gromady mają lecieć na siebie, a nie od siebie.
    #[test]
    fn a_pair_faces_inwards() {
        let cfg = {
            let mut c = config(
                Shape::Pair,
                vec![Ingredient::new(Particle::of(Species::Proton), 100)],
            );
            c.spawn.beam_energy = 50.0;
            c
        };
        let state = make_state(&cfg).state;
        for i in 0..state.n() {
            let towards_centre = -state.positions[i].x.signum();
            assert!(
                state.momenta[i].x * towards_centre > 0.0,
                "cząstka {i} leci na zewnątrz"
            );
        }
        // Zderzacz jest symetryczny, więc pęd wypadkowy znika sam z siebie.
        assert!(state.total_four_momentum().momentum.norm() < 1e-9);
    }

    /// Zerowa temperatura znaczy dokładnie zero, a nie „mało".
    #[test]
    fn zero_temperature_means_zero_momentum() {
        let state = make_state(&config(Shape::Ball, electrons(100))).state;
        assert!(state.momenta.iter().all(|p| *p == ZERO));
    }

    /// Ten sam zarodek musi dać ten sam warunek początkowy, a inny — inny.
    #[test]
    fn seeding_is_reproducible() {
        let mut cfg = config(Shape::Ball, electrons(200));
        cfg.spawn.temperature = 1e-3;
        let first = make_state(&cfg).state;
        let again = make_state(&cfg).state;
        assert_eq!(first.positions, again.positions);
        assert_eq!(first.momenta, again.momenta);

        cfg.spawn.seed += 1;
        let other = make_state(&cfg).state;
        assert_ne!(first.positions, other.positions);
    }

    /// Temperatura porównywalna z masą musi być zapowiedziana. Milczenie dałoby
    /// warunek początkowy z niewłaściwego rozkładu, a wynik wyglądałby normalnie.
    #[test]
    fn a_relativistic_temperature_is_announced() {
        let mut cfg = config(Shape::Ball, electrons(10));
        cfg.spawn.temperature = 1.0; // dwukrotność masy elektronu
        let warnings = make_state(&cfg).warnings;
        assert!(
            warnings.iter().any(|w| w.contains("Maxwella-Jüttnera")),
            "{warnings:?}"
        );

        // Zimna chmura tych samych elektronów nie wywołuje ostrzeżenia.
        cfg.spawn.temperature = 1e-4;
        assert!(make_state(&cfg).warnings.is_empty());
    }

    /// Cząstka bezmasowa nie ma temperatury nierelatywistycznej, więc nie ma o czym
    /// ostrzegać — inaczej każdy bieg z fotonami zaczynałby się od ostrzeżenia.
    #[test]
    fn massless_particles_do_not_trigger_the_temperature_warning() {
        let mut cfg = config(
            Shape::Ball,
            vec![Ingredient::new(Particle::of(Species::Photon), 10)],
        );
        cfg.spawn.temperature = 500.0;
        assert!(make_state(&cfg).warnings.is_empty());
    }

    /// W, Z, H, t i gluony nie wchodzą do gazu nawet jeśli ktoś wpisze je w mieszankę.
    #[test]
    fn catalog_only_species_do_not_enter_the_gas() {
        let cfg = config(
            Shape::Ball,
            vec![
                Ingredient::new(Particle::of(Species::Electron), 4),
                Ingredient::new(Particle::of(Species::WBoson), 3),
                Ingredient::new(Particle::of(Species::Higgs), 1),
                Ingredient::new(Particle::of(Species::Gluon), 2),
                Ingredient::new(Particle::of(Species::Top), 1),
            ],
        );
        let spawned = make_state(&cfg);
        assert_eq!(spawned.state.n(), 4);
        assert_eq!(spawned.state.count_of(Particle::of(Species::Electron)), 4);
        assert!(
            spawned.warnings.iter().any(|w| w.contains("katalogu")),
            "{:?}",
            spawned.warnings
        );
    }

    /// Kwark bez Cornella nie startuje w gazie — uwięzienie to osobne laboratorium.
    #[test]
    fn quarks_without_the_strong_force_are_dropped() {
        let cfg = config(
            Shape::Pair,
            vec![
                Ingredient::new(Particle::of(Species::Up), 1),
                Ingredient::new(Particle::anti_of(Species::Up), 1),
            ],
        );
        let spawned = make_state(&cfg);
        assert!(spawned.state.is_empty());
        assert!(spawned.warnings.iter().any(|w| w.contains("uwięzieniu")));
    }
}
