//! Wielkości zachowane i miary jakości symulacji.
//!
//! Bez pomiaru energii i pędu nie wiadomo, czy symulacja liczy fizykę, czy generuje
//! ładny szum. Wszystkie wielkości są liczone z tego samego potencjału, który
//! wygenerował siły, więc dryf energii mówi o jakości CAŁKOWANIA, a nie
//! o niespójności modelu.
//!
//! Solver przybliżony dodatkowo raportuje własny błąd: dla losowej próbki cząstek
//! liczymy siłę dokładnie i porównujemy. Dzięki temu przybliżenie jest widoczną
//! liczbą, a nie cichym założeniem.
//!
//! Wynik pomiaru to [`Snapshot`] z nazwanymi polami, a nie mapa nazwa → liczba.
//! Odbiorców jest kilku (panel, bieg wsadowy, testy) i każdy czyta inny podzbiór;
//! przy mapie literówka w nazwie ujawniłaby się dopiero w czasie działania, i to jako
//! brak wiersza w tabelce, a nie jako błąd.

use crate::sr::backends::exact::forces_for_rows;
use crate::sr::relativity as sr;
use crate::sr::state::State;
use crate::vec3::Vec3;

/// Błąd solvera przybliżonego względem dokładnego O(N²), znormalizowany.
#[derive(Clone, Copy, Debug)]
pub struct ForceError {
    /// Błąd średniokwadratowy w jednostkach typowej siły.
    pub rms: f64,
    /// Najgorszy pojedynczy przypadek w tej samej normalizacji.
    pub max: f64,
}

/// Jeden pomiar stanu układu.
#[derive(Clone, Copy, Debug)]
pub struct Snapshot {
    pub step: u64,
    pub time: f64,
    pub dt: f64,
    pub n: usize,

    pub kinetic: f64,
    pub potential: f64,
    pub total_energy: f64,
    /// Skumulowana energia odprowadzona przez dyssypację.
    pub energy_removed: f64,

    /// `2K/|U|`; 1 oznacza równowagę.
    pub virial: f64,
    /// `|Σp|` znormalizowane sumą `|pᵢ|` — miara tego, czy układ nie odpływa.
    pub momentum_residual: f64,
    pub angular_momentum: f64,

    pub gamma_mean: f64,
    pub gamma_max: f64,
    pub beta_mean: f64,
    pub beta_max: f64,
    pub half_mass_radius: f64,

    /// Dryf `E_tot + E_odprowadzona` względem stanu początkowego.
    pub energy_drift: f64,
    pub angular_drift: f64,
    pub half_mass_ratio: f64,

    pub effective_softening: f64,
    pub approximate: bool,
    pub force_error: Option<ForceError>,
    pub cooling_particles_per_cell: Option<f64>,
}

/// Wielkości z chwili startu, względem których mierzymy dryf.
#[derive(Clone, Copy, Debug)]
struct Reference {
    energy: f64,
    angular_momentum: f64,
    half_mass_radius: f64,
}

/// Dodatkowe liczby, które silnik zna, a stan nie.
#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    pub dt: f64,
    pub energy_removed: f64,
    pub effective_softening: f64,
    pub approximate: bool,
    pub force_error: Option<ForceError>,
    pub cooling_particles_per_cell: Option<f64>,
}

#[derive(Default)]
pub struct Diagnostics {
    pub history: Vec<Snapshot>,
    reference: Option<Reference>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Zmierz stan i dopisz pomiar do historii.
    ///
    /// `ctx.energy_removed` to skumulowana energia odprowadzona przez dyssypację. Bez
    /// niej włączenie chłodzenia zamieniłoby dryf energii — główny wskaźnik jakości
    /// całkowania — w licznik tego, ile energii celowo wyrzuciliśmy. Wielkością
    /// zachowaną w modelu z dyssypacją jest `E_tot + E_odprowadzona`.
    pub fn observe(&mut self, state: &State, c: f64, ctx: Context) -> Snapshot {
        let n = state.n();
        let kinetic: f64 = (0..n)
            .map(|i| sr::kinetic_energy(state.masses[i], state.momenta[i], c))
            .sum();
        let potential = match state.potential.as_ref() {
            Some(phi) => 0.5 * state.masses.iter().zip(phi.iter()).map(|(m, p)| m * p).sum::<f64>(),
            None => 0.0,
        };
        let total = kinetic + potential;

        let mut gamma_sum = 0.0;
        let mut gamma_max = 0.0f64;
        let mut beta_sum = 0.0;
        let mut beta_max = 0.0f64;
        let mut momentum_scale = 0.0;
        for i in 0..n {
            let g = state.gamma(i, c);
            let b = state.speed_over_c(i, c);
            gamma_sum += g;
            beta_sum += b;
            gamma_max = gamma_max.max(g);
            beta_max = beta_max.max(b);
            momentum_scale += state.momenta[i].norm();
        }
        let inv_n = 1.0 / n.max(1) as f64;

        let angular = state.angular_momentum().norm();
        let r_half = half_mass_radius(&state.positions, &state.masses);

        let reference = *self.reference.get_or_insert(Reference {
            energy: total + ctx.energy_removed,
            angular_momentum: angular,
            half_mass_radius: r_half,
        });

        let snapshot = Snapshot {
            step: state.step,
            time: state.time,
            dt: ctx.dt,
            n,
            kinetic,
            potential,
            total_energy: total,
            energy_removed: ctx.energy_removed,
            virial: if potential != 0.0 {
                2.0 * kinetic / potential.abs()
            } else {
                0.0
            },
            momentum_residual: state.total_momentum().norm() / (momentum_scale + 1e-300),
            angular_momentum: angular,
            gamma_mean: gamma_sum * inv_n,
            gamma_max,
            beta_mean: beta_sum * inv_n,
            beta_max,
            half_mass_radius: r_half,
            energy_drift: relative(total + ctx.energy_removed, reference.energy),
            angular_drift: relative(angular, reference.angular_momentum),
            half_mass_ratio: r_half / (reference.half_mass_radius + 1e-300),
            effective_softening: ctx.effective_softening,
            approximate: ctx.approximate,
            force_error: ctx.force_error,
            cooling_particles_per_cell: ctx.cooling_particles_per_cell,
        };
        self.history.push(snapshot);
        snapshot
    }

    pub fn latest(&self) -> Option<&Snapshot> {
        self.history.last()
    }

    /// Zapomnij stan odniesienia — po zmianie G albo ε poprzedni przestaje obowiązywać.
    pub fn reset_reference(&mut self) {
        self.reference = None;
    }
}

/// Promień zawierający połowę masy, licząc od środka masy.
pub fn half_mass_radius(positions: &[Vec3], masses: &[f64]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let total: f64 = masses.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let com = positions
        .iter()
        .zip(masses.iter())
        .map(|(p, m)| *p * *m)
        .sum::<Vec3>()
        / total;
    let mut radii: Vec<(f64, f64)> = positions
        .iter()
        .zip(masses.iter())
        .map(|(p, m)| ((*p - com).norm(), *m))
        .collect();
    radii.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut running = 0.0;
    for (r, m) in &radii {
        running += m;
        if running >= 0.5 * total {
            return *r;
        }
    }
    radii.last().map(|(r, _)| *r).unwrap_or(0.0)
}

/// Błąd siły solvera na losowej próbce, względem dokładnego O(N²).
pub fn measure_force_error(
    state: &State,
    forces: &[Vec3],
    g: f64,
    softening: f64,
    sample: usize,
    rng: &mut crate::rng::Rng,
) -> ForceError {
    let n = state.n();
    if n == 0 {
        return ForceError { rms: 0.0, max: 0.0 };
    }
    let rows = rng.sample_indices(n, sample.clamp(1, n));
    let reference = forces_for_rows(&state.positions, &state.masses, g, softening, &rows);

    let typical = (reference.iter().map(|f| f.norm_squared()).sum::<f64>() / rows.len() as f64)
        .sqrt();
    if typical <= 0.0 {
        return ForceError { rms: 0.0, max: 0.0 };
    }
    let deltas: Vec<f64> = rows
        .iter()
        .zip(reference.iter())
        .map(|(&i, r)| (forces[i] - *r).norm())
        .collect();
    let rms = (deltas.iter().map(|d| d * d).sum::<f64>() / deltas.len() as f64).sqrt();
    ForceError {
        rms: rms / typical,
        max: deltas.iter().copied().fold(0.0f64, f64::max) / typical,
    }
}

fn relative(value: f64, reference: f64) -> f64 {
    if reference.abs() < 1e-300 {
        return 0.0;
    }
    (value - reference) / reference.abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sr::backends::exact::Exact;
    use crate::mesh::Mesh;
    use crate::sr::backends::Backend;
    use crate::vec3::{vec3, ZERO};

    const C: f64 = 30.0;

    fn cloud(n: usize) -> State {
        let mut rng = Rng::seeded(1);
        let positions: Vec<Vec3> = (0..n).map(|_| rng.normal_vec(0.0, 3.0)).collect();
        let momenta: Vec<Vec3> = (0..n).map(|_| rng.normal_vec(0.0, 0.5)).collect();
        State::new(positions, momenta, vec![1.0; n]).unwrap()
    }

    fn with_field(mut state: State, g: f64, eps: f64) -> State {
        let field = Exact::new().compute(&state.positions, &state.masses, g, eps);
        state.forces = Some(field.force);
        state.potential = Some(field.potential);
        state
    }

    #[test]
    fn half_mass_radius_splits_the_mass() {
        // cztery cząstki na promieniach 1, 2, 3, 4 — połowa masy mieści się w r = 2
        let x = vec![
            vec3(1.0, 0.0, 0.0),
            vec3(-2.0, 0.0, 0.0),
            vec3(0.0, 3.0, 0.0),
            vec3(0.0, -4.0, 0.0),
        ];
        let r = half_mass_radius(&x, &[1.0; 4]);
        assert!((1.0..=3.0).contains(&r), "r_half={r}");
    }

    #[test]
    fn half_mass_radius_of_empty_cloud_is_zero() {
        assert_eq!(half_mass_radius(&[], &[]), 0.0);
    }

    #[test]
    fn drift_is_zero_on_the_first_observation() {
        let state = with_field(cloud(200), 0.16, 0.25);
        let mut d = Diagnostics::new();
        let snap = d.observe(&state, C, Context::default());
        assert_eq!(snap.energy_drift, 0.0);
        assert_eq!(snap.angular_drift, 0.0);
        assert!((snap.half_mass_ratio - 1.0).abs() < 1e-12);
    }

    /// Dyssypacja nie może udawać dryfu energii: wielkością zachowaną jest
    /// `E_tot + E_odprowadzona`. To najważniejsza własność tego modułu.
    #[test]
    fn removed_energy_does_not_count_as_drift() {
        let state = with_field(cloud(200), 0.16, 0.25);
        let mut d = Diagnostics::new();
        let first = d.observe(&state, C, Context::default());

        // udajemy, że chłodzenie zabrało 10% energii i tyle samo zniknęło z układu
        let removed = 0.1 * first.total_energy.abs();
        let mut cooled = state.clone();
        for p in &mut cooled.momenta {
            *p = *p * 0.5;
        }
        let after = d.observe(
            &cooled,
            C,
            Context {
                energy_removed: removed,
                ..Context::default()
            },
        );
        // Dryf liczony BEZ uwzględnienia odprowadzonej energii byłby ogromny;
        // z uwzględnieniem musi być znacznie mniejszy.
        let naive = (after.total_energy - first.total_energy).abs() / first.total_energy.abs();
        assert!(after.energy_drift.abs() < naive, "dryf nie skorygowany");
    }

    #[test]
    fn virial_equals_two_k_over_u() {
        let state = with_field(cloud(300), 0.16, 0.25);
        let mut d = Diagnostics::new();
        let snap = d.observe(&state, C, Context::default());
        let expected = 2.0 * snap.kinetic / snap.potential.abs();
        assert!((snap.virial - expected).abs() < 1e-12);
    }

    #[test]
    fn resting_cloud_has_gamma_one_and_beta_zero() {
        let n = 50;
        let state = with_field(
            State::new(
                (0..n).map(|i| vec3(i as f64, 0.0, 0.0)).collect(),
                vec![ZERO; n],
                vec![1.0; n],
            )
            .unwrap(),
            0.16,
            0.25,
        );
        let mut d = Diagnostics::new();
        let snap = d.observe(&state, C, Context::default());
        assert!((snap.gamma_mean - 1.0).abs() < 1e-12);
        assert!((snap.gamma_max - 1.0).abs() < 1e-12);
        assert_eq!(snap.beta_mean, 0.0);
        assert_eq!(snap.beta_max, 0.0);
        assert_eq!(snap.kinetic, 0.0);
    }

    #[test]
    fn reset_reference_rebases_the_drift() {
        let state = with_field(cloud(150), 0.16, 0.25);
        let mut d = Diagnostics::new();
        d.observe(&state, C, Context::default());
        let mut hotter = state.clone();
        for p in &mut hotter.momenta {
            *p = *p * 2.0;
        }
        let drifted = d.observe(&hotter, C, Context::default());
        assert!(drifted.energy_drift.abs() > 1e-6);

        d.reset_reference();
        let rebased = d.observe(&hotter, C, Context::default());
        assert_eq!(rebased.energy_drift, 0.0);
    }

    #[test]
    fn history_grows_with_observations() {
        let state = with_field(cloud(50), 0.16, 0.25);
        let mut d = Diagnostics::new();
        for _ in 0..5 {
            d.observe(&state, C, Context::default());
        }
        assert_eq!(d.history.len(), 5);
        assert!(d.latest().is_some());
    }

    /// Wzorzec porównywany z samym sobą musi dać błąd zero — inaczej sam pomiar
    /// błędu byłby źródłem błędu.
    #[test]
    fn exact_solver_measures_zero_error() {
        let state = with_field(cloud(300), 0.16, 0.25);
        let mut rng = Rng::seeded(2);
        let err = measure_force_error(
            &state,
            state.forces.as_ref().unwrap(),
            0.16,
            0.25,
            64,
            &mut rng,
        );
        assert!(err.rms < 1e-12, "rms={}", err.rms);
        assert!(err.max < 1e-12, "max={}", err.max);
    }

    /// A solver siatkowy musi dać błąd niezerowy, ale skończony — inaczej pomiar nic
    /// nie mierzy.
    #[test]
    fn mesh_solver_measures_finite_nonzero_error() {
        let mut state = cloud(2_000);
        let mut mesh = Mesh::new(32, 0.15);
        let eps = mesh.effective_softening(0.25).max(0.25);
        let field = mesh.compute(&state.positions, &state.masses, 0.16, 0.25);
        state.forces = Some(field.force);
        let mut rng = Rng::seeded(3);
        let err = measure_force_error(
            &state,
            state.forces.as_ref().unwrap(),
            0.16,
            mesh.effective_softening(eps),
            128,
            &mut rng,
        );
        assert!(err.rms > 0.0 && err.rms < 1.0, "rms={}", err.rms);
        assert!(err.max >= err.rms);
    }

    #[test]
    fn momentum_residual_is_zero_for_balanced_cloud() {
        let state = with_field(
            State::new(
                vec![vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)],
                vec![vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0)],
                vec![1.0, 1.0],
            )
            .unwrap(),
            0.16,
            0.25,
        );
        let mut d = Diagnostics::new();
        assert!(d.observe(&state, C, Context::default()).momentum_residual < 1e-15);
    }
}
