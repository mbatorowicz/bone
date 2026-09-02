//! Dokładne O(N²) — każda para, bez odcięcia i bez list sąsiadów.
//!
//! Odległość liczymy wprost z różnicy położeń, a NIE z rozwinięcia
//! `r² = |xᵢ|² − 2xᵢ·xⱼ + |xⱼ|²`. Rozwinięcie pozwala zamienić pętlę po parach na
//! mnożenie macierzy, co ma sens tam, gdzie własna pętla jest o dwa rzędy wielkości
//! wolniejsza od bibliotecznego BLAS-u — tutaj nie jest. Płaciłoby się za to utratą
//! cyfr znaczących przy `r ≪ |x|`: różnica trzech liczb rzędu `|x|²` daje `r²` obarczone
//! błędem bezwzględnym rzędu `ε|x|²`, więc dla bliskich cząstek zostaje szum. Ratowałby
//! przed tym tylko softening, o ile byłby większy od tego szumu — czyli poprawność
//! zależałaby od nastawy, którą użytkownik może zmienić.
//!
//! Zostaje więc czysty rachunek na `(xᵢ − xⱼ)`, zrównoleglony po `i`, bez tablic
//! pośrednich i bez limitu pamięci na kafel.
//!
//! Człon własny `j = i` jest pomijany jawnie, a nie przez to, że wychodzi zero:
//! poleganie na kasowaniu się dwóch dużych liczb jest proszeniem się o kłopoty.

use rayon::prelude::*;

use crate::sr::backends::Backend;
use crate::sr::state::Field;
use crate::vec3::{Vec3, ZERO};

/// Dokładne siły. Sensowne do kilku tysięcy cząstek — koszt rośnie jak N².
#[derive(Default)]
pub struct Exact;

impl Exact {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for Exact {
    fn name(&self) -> &'static str {
        "exact"
    }

    fn compute(&mut self, positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> Field {
        let n = positions.len();
        if n < 2 {
            return Field::zeros(n);
        }
        let eps2 = softening * softening;

        let mut force = vec![ZERO; n];
        let mut potential = vec![0.0f64; n];
        force
            .par_iter_mut()
            .zip(potential.par_iter_mut())
            .enumerate()
            .for_each(|(i, (f_out, phi_out))| {
                let (pull, phi) = accumulate(i, positions, masses, eps2);
                *f_out = pull * (-g * masses[i]);
                *phi_out = -g * phi;
            });
        Field { force, potential }
    }
}

/// Suma `Σ_{j≠i} m_j (xᵢ − xⱼ)/(r²+ε²)^{3/2}` oraz `Σ_{j≠i} m_j/√(r²+ε²)`.
///
/// Zwracane bez czynnika `−G m_i`, żeby ta sama pętla obsługiwała siłę i potencjał
/// (które różnią się tylko potęgą mianownika i przedmnożnikiem).
fn accumulate(i: usize, positions: &[Vec3], masses: &[f64], eps2: f64) -> (Vec3, f64) {
    let xi = positions[i];
    let mut pull = ZERO;
    let mut phi = 0.0;
    for (j, xj) in positions.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = xi - *xj;
        let inv_r = 1.0 / (d.norm_squared() + eps2).sqrt();
        let m = masses[j];
        phi += m * inv_r;
        pull += d * (m * inv_r * inv_r * inv_r);
    }
    (pull, phi)
}

/// Dokładna siła działająca na wybrane cząstki ze strony WSZYSTKICH pozostałych.
///
/// Służy do mierzenia błędu solverów przybliżonych: koszt to O(|rows|·N) zamiast
/// O(N²), więc kontrolę można włączyć nawet przy milionie cząstek.
pub fn forces_for_rows(
    positions: &[Vec3],
    masses: &[f64],
    g: f64,
    softening: f64,
    rows: &[usize],
) -> Vec<Vec3> {
    let eps2 = softening * softening;
    rows.par_iter()
        .map(|&i| {
            let (pull, _) = accumulate(i, positions, masses, eps2);
            pull * (-g * masses[i])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    const G: f64 = 2.0;
    const EPS: f64 = 0.0;

    #[test]
    fn two_bodies_attract_each_other() {
        let x = vec![vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)];
        let m = vec![1.0, 1.0];
        let field = Exact::new().compute(&x, &m, G, EPS);
        assert!(field.force[0].x > 0.0, "lewa ma być ciągnięta w prawo");
        assert!(field.force[1].x < 0.0, "prawa ma być ciągnięta w lewo");
    }

    /// Trzecie prawo Newtona: siły wewnętrzne muszą się zsumować do zera. To
    /// najczulszy test na pomylony znak albo indeks w pętli po parach.
    #[test]
    fn internal_forces_cancel() {
        let x = vec![
            vec3(-1.0, 0.5, 0.0),
            vec3(1.0, 0.0, -0.3),
            vec3(0.2, -1.0, 0.7),
            vec3(-0.4, 0.9, 1.1),
        ];
        let m = vec![1.0, 2.0, 0.5, 3.0];
        let field = Exact::new().compute(&x, &m, G, 0.1);
        let total: Vec3 = field.force.iter().copied().sum();
        let scale: f64 = field.force.iter().map(|f| f.norm()).sum();
        assert!(total.norm() / scale < 1e-12, "residuum {}", total.norm());
    }

    /// Wartość bezwzględna względem wzoru analitycznego dla dwóch ciał.
    #[test]
    fn magnitude_matches_newton() {
        let r = 3.0;
        let x = vec![vec3(0.0, 0.0, 0.0), vec3(r, 0.0, 0.0)];
        let m = vec![2.0, 5.0];
        let field = Exact::new().compute(&x, &m, G, EPS);
        let expected = G * m[0] * m[1] / (r * r);
        assert!((field.force[0].norm() - expected).abs() / expected < 1e-12);
    }

    /// Potencjał musi być spójny z siłą: `F = −m∇φ`. Sprawdzane różnicą centralną,
    /// bo to jedyny sposób wyłapania niezgodnej konwencji ε między nimi — a to
    /// właśnie ta zgodność sprawia, że dryf energii cokolwiek znaczy.
    #[test]
    fn force_is_minus_mass_times_gradient_of_potential() {
        let base = vec![
            vec3(0.0, 0.0, 0.0),
            vec3(1.7, 0.4, -0.9),
            vec3(-1.1, 1.3, 0.6),
        ];
        let m = vec![1.0, 2.0, 1.5];
        let eps = 0.3;
        let field = Exact::new().compute(&base, &m, G, eps);

        let d = 1e-6;
        for axis in 0..3 {
            let potential_at = |offset: f64| {
                let mut x = base.clone();
                let mut p = x[0];
                p.set(axis, p.get(axis) + offset);
                x[0] = p;
                Exact::new().compute(&x, &m, G, eps).potential[0]
            };
            let grad = (potential_at(d) - potential_at(-d)) / (2.0 * d);
            let expected = -m[0] * grad;
            let got = field.force[0].get(axis);
            assert!(
                (got - expected).abs() / expected.abs().max(1e-12) < 1e-4,
                "os {axis}: F={got} vs −m∂φ={expected}"
            );
        }
    }

    #[test]
    fn single_particle_feels_nothing() {
        let field = Exact::new().compute(&[vec3(1.0, 2.0, 3.0)], &[1.0], G, EPS);
        assert_eq!(field.force[0], ZERO);
        assert_eq!(field.potential[0], 0.0);
    }

    #[test]
    fn softening_bounds_the_force_at_zero_separation() {
        let x = vec![ZERO, ZERO];
        let m = vec![1.0, 1.0];
        let field = Exact::new().compute(&x, &m, G, 0.5);
        assert!(field.force[0].norm().is_finite());
        assert!(field.potential[0].is_finite());
    }

    #[test]
    fn row_subset_matches_full_solve() {
        let x: Vec<Vec3> = (0..40)
            .map(|i| {
                let t = i as f64 * 0.37;
                vec3(t.sin() * 4.0, t.cos() * 3.0, (t * 0.7).sin() * 2.0)
            })
            .collect();
        let m: Vec<f64> = (0..40).map(|i| 1.0 + (i % 5) as f64 * 0.3).collect();
        let full = Exact::new().compute(&x, &m, G, 0.2);
        let rows = [0usize, 7, 13, 39];
        let subset = forces_for_rows(&x, &m, G, 0.2, &rows);
        for (k, &i) in rows.iter().enumerate() {
            let delta = (subset[k] - full.force[i]).norm();
            assert!(delta / full.force[i].norm() < 1e-12, "wiersz {i}");
        }
    }

    #[test]
    fn potential_energy_is_negative_for_bound_cloud() {
        let x = vec![vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)];
        let m = vec![1.0, 1.0];
        let field = Exact::new().compute(&x, &m, G, 0.1);
        assert!(field.energy(&m) < 0.0);
    }
}
