//! Wspólny kontrakt solverów liczących grawitację.
//!
//! Każdy solver zwraca siłę ORAZ potencjał w jednym przebiegu. Potencjał jest przy
//! okazji prawie darmowy, a bez niego nie da się policzyć energii — czyli jedynej
//! liczby, która mówi, czy symulacja jeszcze jest symulacją.
//!
//! Konwencja (softening Plummera, ε o wymiarze długości):
//!
//! ```text
//! φ_i = −G Σ_{j≠i} m_j / √(r_ij² + ε²)
//! F_i = −G m_i Σ_{j≠i} m_j (x_i − x_j) / (r_ij² + ε²)^{3/2}
//! U   = ½ Σ_i m_i φ_i
//! ```
//!
//! Para (φ, F) jest spójna: F = −m∇φ dokładnie dla tego samego ε. To dlatego dryf
//! energii jest sensowną miarą jakości całkowania, a nie artefaktem tego, że siła
//! i potencjał pochodzą z dwóch różnych modeli.

pub mod exact;

use crate::mesh::Mesh;
use crate::sr::config::{BackendKind, Config};
use crate::sr::state::Field;
use crate::vec3::Vec3;

pub use exact::{forces_for_rows, Exact};

pub trait Backend: Send {
    fn name(&self) -> &'static str;

    /// Policz siły i potencjał dla podanego stanu.
    fn compute(&mut self, positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> Field;

    /// Czy wynik jest przybliżony. Jeśli tak, błąd powinien być mierzalny.
    fn approximate(&self) -> bool {
        false
    }

    /// Softening, którym solver NAPRAWDĘ liczy.
    ///
    /// Solver siatkowy nie rozdzieli skali mniejszej od oczka, więc może pracować
    /// z większym ε niż zamówione. Zwracanie tej liczby pozwala pokazać ją
    /// użytkownikowi i porównywać błąd z właściwym wzorcem, zamiast po cichu liczyć
    /// co innego, niż napisano w panelu.
    fn effective_softening(&self, requested: f64) -> f64 {
        requested
    }

    fn describe(&self) -> String {
        self.name().to_string()
    }
}

/// [`Mesh`] mieszka poza gałęzią SR, bo tego samego solvera używa tryb ΛCDM.
/// Tutaj jest tylko dopasowanie go do kontraktu: siatka daje przyspieszenie,
/// a `Backend` mówi o siłach.
impl Backend for Mesh {
    fn name(&self) -> &'static str {
        "mesh"
    }

    fn compute(&mut self, positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> Field {
        if positions.len() < 2 {
            return Field::zeros(positions.len());
        }
        self.refresh(positions, masses, g, softening);
        let force = self
            .gather_acceleration(positions)
            .iter()
            .zip(masses.iter())
            .map(|(a, m)| *a * *m)
            .collect();
        Field {
            force,
            potential: self.gather_potential(positions, masses, g),
        }
    }

    fn approximate(&self) -> bool {
        true
    }

    fn effective_softening(&self, requested: f64) -> f64 {
        Mesh::effective_softening(self, requested)
    }

    fn describe(&self) -> String {
        Mesh::describe(self)
    }
}

/// Zbuduj solver zgodnie z konfiguracją.
///
/// `auto` wybiera ten solver, który na TEJ konfiguracji jest tańszy — patrz
/// [`prefer_exact`]. Póki dokładny mieści się w budżecie, liczymy dokładnie, bo jest
/// wtedy nie tylko szybszy, ale i bez błędu siły.
pub fn make_backend(cfg: &Config, n_particles: usize) -> Box<dyn Backend> {
    let kind = match cfg.solver.backend {
        BackendKind::Auto if prefer_exact(n_particles, cfg.solver.grid) => BackendKind::Exact,
        BackendKind::Auto => BackendKind::Mesh,
        explicit => explicit,
    };
    match kind {
        BackendKind::Exact => Box::new(Exact::new()),
        BackendKind::Mesh | BackendKind::Auto => {
            Box::new(Mesh::new(cfg.solver.grid, cfg.solver.box_margin))
        }
    }
}

/// Czy dokładne `O(N²)` jest na tej konfiguracji tańsze od siatki.
///
/// Progu w postaci jednej liczby cząstek tu nie ma, bo taki próg musiałby być zły:
/// koszt siatki nie zależy od `N`, tylko od siatki, i między siatką 64 a 192 różni
/// się **czterdziestokrotnie**. Granica ustawiona na 4 tys. cząstek przy siatce 96
/// oddawałaby siatce bieg, który dokładny solver liczy szybciej aż do ~12 tys. — czyli
/// traciłaby jednocześnie szybkość i dokładność, i to bez żadnej rekompensaty.
///
/// Model kosztu: dokładny rośnie jak `N²`, siatkowy jak `(2·grid)³·log₂(2·grid)`.
/// Obie stałe zmierzono na tej maszynie, ale liczy się wyłącznie ich ILORAZ, a on jest
/// znacznie mniej wrażliwy na sprzęt niż każda z osobna — obie skalują się z liczbą
/// rdzeni i przepustowością pamięci. Zmierzone punkty odniesienia (22 rdzenie):
/// `N = 4000` dokładnie ≈ 9,5 ms, siatka 96 ≈ 90 ms, siatka 192 ≈ 1480 ms.
///
/// Model myli się o czynnik ~2 na skrajnych siatkach (efekty pamięci podręcznej),
/// co przesuwa granicę o ~40%. To wystarcza: w okolicy granicy oba solvery kosztują
/// tyle samo, więc pomyłka w tę czy w tę stronę nic nie zmienia.
pub fn prefer_exact(n_particles: usize, grid: usize) -> bool {
    /// Sekundy na parę cząstek w solverze dokładnym.
    const EXACT_PER_PAIR: f64 = 6e-10;
    /// Sekundy na komórkę i jednostkę `log₂` w solverze siatkowym.
    const MESH_PER_CELL_LOG: f64 = 1.7e-9;

    let n = n_particles as f64;
    let padded = (2 * grid.max(1)) as f64;
    let mesh = MESH_PER_CELL_LOG * padded.powi(3) * padded.log2();
    EXACT_PER_PAIR * n * n <= mesh
}

/// Solver wzorcowy do mierzenia błędu — zawsze dokładny.
pub fn reference_backend() -> Box<dyn Backend> {
    Box::new(Exact::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::config::SolverConfig;

    fn cfg_with(backend: BackendKind, grid: usize) -> Config {
        Config {
            solver: SolverConfig {
                backend,
                grid,
                ..SolverConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn auto_picks_exact_for_small_clouds() {
        let cfg = cfg_with(BackendKind::Auto, 64);
        assert!(!make_backend(&cfg, 100).approximate());
    }

    #[test]
    fn auto_picks_mesh_for_large_clouds() {
        let cfg = cfg_with(BackendKind::Auto, 64);
        assert!(make_backend(&cfg, 200_000).approximate());
    }

    #[test]
    fn explicit_choice_overrides_the_cost_model() {
        let cfg = cfg_with(BackendKind::Exact, 64);
        assert!(!make_backend(&cfg, 1_000_000).approximate());
        let cfg = cfg_with(BackendKind::Mesh, 64);
        assert!(make_backend(&cfg, 2).approximate());
    }

    /// Sedno poprawki: granica MUSI rosnąć z siatką. Gęstsza siatka jest droższa,
    /// więc dokładny solver opłaca się dłużej — stała liczba cząstek tego nie widziała.
    #[test]
    fn the_crossover_grows_with_the_grid() {
        let crossover = |grid: usize| {
            (1..400)
                .map(|k| k * 1_000)
                .find(|n| !prefer_exact(*n, grid))
                .expect("granica leży w badanym zakresie")
        };
        let coarse = crossover(64);
        let fine = crossover(192);
        assert!(
            fine > 3 * coarse,
            "siatka 192 jest 40× droższa od 64, a granica przesunęła się z {coarse} \
             tylko do {fine}"
        );
    }

    /// Granica przy domyślnej siatce musi wypadać w tysiącach cząstek, a nie
    /// w setkach ani w milionach. Test pilnuje rzędu wielkości stałych modelu:
    /// pomyłka o dekadę znaczyłaby, że `auto` w praktyce wybiera zawsze to samo.
    #[test]
    fn the_default_crossover_is_in_the_right_decade() {
        assert!(prefer_exact(5_000, 64), "5 tys. cząstek powinno iść dokładnie");
        assert!(
            !prefer_exact(60_000, 64),
            "60 tys. cząstek nie ma prawa iść dokładnie"
        );
    }

    /// Zdegenerowane wejścia nie mogą panikować ani dzielić przez zero: `grid = 0`
    /// przychodzi z konfiguracji wczytanej z pliku, a `n = 0` z pustej chmury.
    #[test]
    fn degenerate_input_is_answered_not_crashed() {
        assert!(prefer_exact(0, 0));
        assert!(!prefer_exact(usize::MAX, 16));
    }

    fn cube_cloud(per_side: usize, size: f64) -> (Vec<crate::vec3::Vec3>, Vec<f64>) {
        crate::fixtures::cube_cloud(per_side, size)
    }

    /// Sedno kontraktu: siatka musi być PRZYBLIŻENIEM dokładnego solvera, a nie
    /// czymś innym. Bez tego testu błąd znaku, skali czy jednostek w jądrze
    /// przechodzi niezauważony, bo pojedyncze siły „wyglądają rozsądnie".
    #[test]
    fn mesh_agrees_with_exact_within_its_own_error() {
        const G: f64 = 43.0;
        let (x, m) = cube_cloud(8, 10.0);
        let requested = 1.5;
        let mut mesh = Mesh::new(32, 0.15);
        let approx = mesh.compute(&x, &m, G, requested);
        // Wzorzec liczymy softeningiem, którym siatka NAPRAWDĘ liczyła — inaczej
        // mierzylibyśmy nie błąd metody, tylko różnicę dwóch modeli.
        let eps = Backend::effective_softening(&mesh, requested);
        let exact = Exact::new().compute(&x, &m, G, eps);

        let typical =
            (exact.force.iter().map(|f| f.norm_squared()).sum::<f64>() / x.len() as f64).sqrt();
        let rms = (approx
            .force
            .iter()
            .zip(exact.force.iter())
            .map(|(a, b)| (*a - *b).norm_squared())
            .sum::<f64>()
            / x.len() as f64)
            .sqrt();
        assert!(rms / typical < 0.2, "błąd RMS {:.1}%", 100.0 * rms / typical);
    }

    /// Solver na dwóch cząstkach nie ma prawa być bezczynny, ale na jednej — musi.
    #[test]
    fn a_single_particle_feels_nothing() {
        for mut backend in [
            Box::new(Exact::new()) as Box<dyn Backend>,
            Box::new(Mesh::new(16, 0.15)),
        ] {
            let field = backend.compute(&[crate::vec3::ZERO], &[1.0], 43.0, 0.5);
            assert_eq!(field.force, vec![crate::vec3::ZERO], "{}", backend.name());
            assert_eq!(field.potential, vec![0.0], "{}", backend.name());
        }
    }
}
