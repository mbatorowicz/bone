//! Relatywistyczny leapfrog KDK na pędzie.
//!
//! ```text
//! p ← p + ½Δt F(x)
//! x ← x + Δt v(p)
//! p ← p + ½Δt F(x)
//! ```
//!
//! Siła z drugiego kopnięcia jest przenoszona do pierwszego kopnięcia następnego
//! kroku, więc na krok przypada dokładnie JEDNO liczenie sił — przy dokładnym O(N²)
//! to różnica dwukrotna w czasie działania.
//!
//! Krok czasowy jest dobierany adaptacyjnie z kryterium przyspieszenia
//! `Δt ≤ η√(ε/a_max)`, standardowego w kodach N-ciałowych. Suwak „tempo symulacji"
//! zwiększa LICZBĘ kroków na klatkę, a nie ich długość — mnożenie Δt bez żadnego
//! kryterium nie wysadzałoby układu tylko dlatego, że relatywistyka nie pozwala
//! przekroczyć c, ale trajektorie przestawałyby cokolwiek znaczyć.

use crate::sr::backends::Backend;
use crate::sr::config::PhysicsConfig;
use crate::sr::relativity as sr;
use crate::sr::state::State;

/// `Δt = min(Δt_max, η√(ε/a_max))`.
///
/// `softening` to ε, którym solver NAPRAWDĘ liczy — na siatce nie mniejsze
/// od oczka. Kryterium z zamówionym ε byłoby optymistyczne: krok za krótki
/// względem siły, którą siatka i tak wygładza.
pub fn choose_dt(state: &State, phys: &PhysicsConfig, softening: f64) -> f64 {
    if !phys.adaptive_dt {
        return phys.dt_max;
    }
    let Some(forces) = state.forces.as_ref() else {
        return phys.dt_max;
    };
    let a_max = forces
        .iter()
        .zip(state.masses.iter())
        .map(|(f, m)| f.norm() / sr::rest_mass(*m))
        .fold(0.0f64, f64::max);
    if !a_max.is_finite() || a_max <= 0.0 {
        return phys.dt_max;
    }
    let eps = if softening.is_finite() && softening > 0.0 {
        softening
    } else {
        phys.softening
    };
    phys.dt_max.min(phys.accuracy * (eps / a_max).sqrt())
}

/// Zapewnij aktualne pole sił w stanie.
///
/// Potrzebne po wczytaniu checkpointu i po zmianie G albo ε, bo wtedy zapamiętane
/// siły opisują już inną fizykę.
pub fn ensure_field(backend: &mut dyn Backend, state: &mut State, phys: &PhysicsConfig) {
    if state.forces.is_some() && state.potential.is_some() {
        return;
    }
    let field = backend.compute(&state.positions, &state.masses, phys.g, phys.softening);
    state.forces = Some(field.force);
    state.potential = Some(field.potential);
}

/// Wykonaj jeden krok. Zwraca użyty `Δt`.
pub fn step(backend: &mut dyn Backend, state: &mut State, phys: &PhysicsConfig) -> f64 {
    ensure_field(backend, state, phys);
    let softening = backend.effective_softening(phys.softening);
    let dt = choose_dt(state, phys, softening);
    let half = 0.5 * dt;

    {
        let forces = state.forces.as_ref().expect("ensure_field ustawiło siły");
        for (p, f) in state.momenta.iter_mut().zip(forces.iter()) {
            *p += *f * half;
        }
    }
    for i in 0..state.positions.len() {
        state.positions[i] += sr::velocity(state.masses[i], state.momenta[i], phys.c) * dt;
    }

    let field = backend.compute(&state.positions, &state.masses, phys.g, phys.softening);
    for (p, f) in state.momenta.iter_mut().zip(field.force.iter()) {
        *p += *f * half;
    }

    state.forces = Some(field.force);
    state.potential = Some(field.potential);
    state.time += dt;
    state.step += 1;
    dt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::backends::exact::Exact;
    use crate::vec3::{vec3, ZERO};

    fn phys() -> PhysicsConfig {
        PhysicsConfig {
            g: 1.0,
            c: 1e6,
            softening: 1e-3,
            dt_max: 1e-3,
            accuracy: 0.01,
            adaptive_dt: true,
            ..PhysicsConfig::default()
        }
    }

    /// Kołowa orbita dwóch równych mas: układ musi zostać na okręgu, a nie spaść ani
    /// odlecieć. `c` jest tu ogromne, więc mierzymy czystą mechanikę newtonowską.
    #[test]
    fn circular_binary_keeps_its_separation() {
        let ph = phys();
        let r = 1.0;
        let m = 1.0;
        // dla dwóch mas m na odległości 2r prędkość kołowa to √(Gm/(4r))
        let v = (ph.g * m / (4.0 * r)).sqrt();
        let mut state = State::new(
            vec![vec3(-r, 0.0, 0.0), vec3(r, 0.0, 0.0)],
            vec![vec3(0.0, -m * v, 0.0), vec3(0.0, m * v, 0.0)],
            vec![m, m],
        )
        .unwrap();
        let mut backend = Exact::new();

        let separation = |s: &State| (s.positions[1] - s.positions[0]).norm();
        let start = separation(&state);
        for _ in 0..4_000 {
            step(&mut backend, &mut state, &ph);
        }
        let end = separation(&state);
        assert!(
            (end / start - 1.0).abs() < 0.02,
            "odległość {start} → {end}"
        );
    }

    /// Leapfrog KDK ma zachowywać energię z dryfem ograniczonym, a nie narastającym.
    /// To główny wskaźnik jakości całkowania w tym kodzie.
    #[test]
    fn energy_drift_stays_bounded() {
        let ph = phys();
        let mut backend = Exact::new();
        let mut rng = crate::rng::Rng::seeded(1);
        let n = 60;
        let positions: Vec<_> = (0..n).map(|_| rng.normal_vec(0.0, 1.0)).collect();
        let momenta: Vec<_> = (0..n).map(|_| rng.normal_vec(0.0, 0.02)).collect();
        let mut state = State::new(positions, momenta, vec![1.0; n]).unwrap();

        let energy = |s: &State, b: &mut Exact| -> f64 {
            let field = b.compute(&s.positions, &s.masses, ph.g, ph.softening);
            let kinetic: f64 = (0..s.n())
                .map(|i| sr::kinetic_energy(s.masses[i], s.momenta[i], ph.c))
                .sum();
            kinetic + field.energy(&s.masses)
        };
        let start = energy(&state, &mut backend);
        for _ in 0..2_000 {
            step(&mut backend, &mut state, &ph);
        }
        let end = energy(&state, &mut backend);
        let drift = (end - start).abs() / start.abs();
        assert!(drift < 0.02, "dryf energii {drift}");
    }

    /// Pęd całkowity jest zachowany dokładnie, bo siły wewnętrzne się znoszą.
    #[test]
    fn total_momentum_is_conserved() {
        let ph = phys();
        let mut backend = Exact::new();
        let mut rng = crate::rng::Rng::seeded(2);
        let n = 40;
        let positions: Vec<_> = (0..n).map(|_| rng.normal_vec(0.0, 1.0)).collect();
        let mut state = State::new(positions, vec![ZERO; n], vec![1.0; n]).unwrap();
        let before = state.total_momentum();
        for _ in 0..200 {
            step(&mut backend, &mut state, &ph);
        }
        let scale: f64 = state.momenta.iter().map(|p| p.norm()).sum();
        assert!((state.total_momentum() - before).norm() / scale.max(1e-30) < 1e-10);
    }

    #[test]
    fn adaptive_step_shrinks_when_acceleration_grows() {
        let ph = phys();
        let mut backend = Exact::new();
        let close = {
            let mut s = State::new(
                vec![vec3(-0.01, 0.0, 0.0), vec3(0.01, 0.0, 0.0)],
                vec![ZERO, ZERO],
                vec![1.0, 1.0],
            )
            .unwrap();
            ensure_field(&mut backend, &mut s, &ph);
            choose_dt(&s, &ph, ph.softening)
        };
        let far = {
            let mut s = State::new(
                vec![vec3(-50.0, 0.0, 0.0), vec3(50.0, 0.0, 0.0)],
                vec![ZERO, ZERO],
                vec![1.0, 1.0],
            )
            .unwrap();
            ensure_field(&mut backend, &mut s, &ph);
            choose_dt(&s, &ph, ph.softening)
        };
        assert!(close < far, "blisko {close}, daleko {far}");
    }

    #[test]
    fn fixed_step_mode_ignores_acceleration() {
        let mut ph = phys();
        ph.adaptive_dt = false;
        let mut backend = Exact::new();
        let mut s = State::new(
            vec![vec3(-0.001, 0.0, 0.0), vec3(0.001, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![1.0, 1.0],
        )
        .unwrap();
        ensure_field(&mut backend, &mut s, &ph);
        assert_eq!(choose_dt(&s, &ph, ph.softening), ph.dt_max);
    }

    #[test]
    fn larger_effective_softening_allows_a_longer_step() {
        let ph = phys();
        let mut backend = Exact::new();
        let mut s = State::new(
            vec![vec3(-0.05, 0.0, 0.0), vec3(0.05, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![1.0, 1.0],
        )
        .unwrap();
        ensure_field(&mut backend, &mut s, &ph);
        let tight = choose_dt(&s, &ph, ph.softening);
        let loose = choose_dt(&s, &ph, ph.softening * 16.0);
        assert!(loose > tight, "ciasne {tight}, luźne {loose}");
    }

    /// Nawet przy absurdalnie dużej sile prędkość nie może przekroczyć c — to jest
    /// cała rzecz, po którą stan trzyma pęd, a nie prędkość.
    #[test]
    fn speed_never_exceeds_c_under_extreme_force() {
        let ph = PhysicsConfig {
            g: 1e6,
            c: 1.0,
            softening: 1e-3,
            dt_max: 0.01,
            adaptive_dt: false,
            ..PhysicsConfig::default()
        };
        let mut backend = Exact::new();
        let mut state = State::new(
            vec![vec3(-0.01, 0.0, 0.0), vec3(0.01, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![1.0, 1.0],
        )
        .unwrap();
        for _ in 0..500 {
            step(&mut backend, &mut state, &ph);
            for i in 0..state.n() {
                let beta = state.speed_over_c(i, ph.c);
                assert!(beta < 1.0, "β={beta}");
            }
        }
    }

    #[test]
    fn step_advances_clock_and_counter() {
        let ph = phys();
        let mut backend = Exact::new();
        let mut state = State::new(
            vec![vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)],
            vec![ZERO, ZERO],
            vec![1.0, 1.0],
        )
        .unwrap();
        let dt = step(&mut backend, &mut state, &ph);
        assert_eq!(state.step, 1);
        assert!((state.time - dt).abs() < 1e-18);
    }
}
