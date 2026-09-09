//! Próbkowanie `|ψ|²` metodą Metropolisa–Hastingsa.
//!
//! Chmura punktów na ekranie to realizacja miary probabilistycznej, nie zbiór
//! elektronów. Jeden elektron w `1s` jest reprezentowany tysiącami punktów,
//! bo inaczej orbitalu nie widać. To jest narzędzie rysunkowe, i diagnostyka
//! o tym mówi.

use crate::qm::hydrogen::density;
use crate::rng::Rng;
use crate::vec3::{vec3, Vec3};

pub fn metropolis(
    mut density_at: impl FnMut(Vec3) -> f64,
    n: usize,
    seed: u64,
    step: f64,
    start: Vec3,
) -> Vec<Vec3> {
    let n = n.max(1);
    let mut rng = Rng::seeded(seed);
    let step = step.max(1e-4);
    let mut pos = start;
    let mut current = density_at(pos).max(1e-300);
    let burn = (2 * n).max(800);
    for _ in 0..burn {
        walk(&mut rng, &mut pos, &mut current, step, &mut density_at);
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..6 {
            walk(&mut rng, &mut pos, &mut current, step, &mut density_at);
        }
        out.push(pos);
    }
    out
}

/// Jeden zamach Metropolisa na już istniejącej chmurze — „iskrzenie" stacjonarne
/// albo podążanie za bijącą gęstością superpozycji.
pub fn relax(
    points: &mut [Vec3],
    mut density_at: impl FnMut(Vec3) -> f64,
    seed: u64,
    step: f64,
    sweeps: usize,
) {
    if points.is_empty() {
        return;
    }
    let mut rng = Rng::seeded(seed);
    let step = step.max(1e-4);
    for p in points.iter_mut() {
        let mut current = density_at(*p).max(1e-300);
        for _ in 0..sweeps.max(1) {
            walk(&mut rng, p, &mut current, step, &mut density_at);
        }
    }
}

fn walk(
    rng: &mut Rng,
    pos: &mut Vec3,
    current: &mut f64,
    step: f64,
    density_at: &mut impl FnMut(Vec3) -> f64,
) {
    let prop = vec3(
        pos.x + step * rng.standard_normal(),
        pos.y + step * rng.standard_normal(),
        pos.z + step * rng.standard_normal(),
    );
    let d = density_at(prop).max(0.0);
    if d >= *current || rng.unit() < d / *current {
        *pos = prop;
        *current = d.max(1e-300);
    }
}

pub fn sample_orbital(z: f64, n: u32, l: u32, m: i32, count: usize, seed: u64) -> Vec<Vec3> {
    let scale = crate::qm::hydrogen::mean_radius(z, n, l).max(0.3);
    metropolis(
        |p| density(z, n, l, m, p.x, p.y, p.z),
        count,
        seed,
        0.45 * scale,
        vec3(0.0, 0.0, 0.6 * scale),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm::hydrogen::{density, mean_radius};

    #[test]
    fn one_s_sample_mean_radius_is_near_1_5() {
        let pts = sample_orbital(1.0, 1, 0, 0, 4_000, 7);
        let mean = pts.iter().map(|p| p.norm()).sum::<f64>() / pts.len() as f64;
        let expect = mean_radius(1.0, 1, 0);
        assert!(
            (mean - expect).abs() / expect < 0.12,
            "⟨r⟩ próbki = {mean}, wzór = {expect}"
        );
        assert!(pts.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn two_p_z_avoids_the_xy_plane_and_the_nucleus() {
        let pts = sample_orbital(1.0, 2, 1, 0, 3_000, 3);
        let near_nucleus = pts.iter().filter(|p| p.norm() < 0.4).count();
        let near_equator = pts
            .iter()
            .filter(|p| p.z.abs() < 0.4 && p.norm() > 1.0)
            .count();
        assert!(
            near_nucleus < pts.len() / 20,
            "za dużo punktów na jądrze: {near_nucleus}"
        );
        assert!(
            near_equator < pts.len() / 8,
            "2p_z nie powinno siadać na równiku: {near_equator}"
        );
        let north = pts.iter().filter(|p| p.z > 0.0).count();
        let south = pts.len() - north;
        let ratio = north.min(south) as f64 / north.max(south) as f64;
        assert!(ratio > 0.7, "płaty 2p_z nierówne: {north}/{south}");
    }

    #[test]
    fn relax_keeps_the_cloud_in_the_support_of_the_density() {
        let mut pts = sample_orbital(1.0, 1, 0, 0, 400, 1);
        relax(
            &mut pts,
            |p| density(1.0, 1, 0, 0, p.x, p.y, p.z),
            99,
            0.4,
            4,
        );
        let mean = pts.iter().map(|p| p.norm()).sum::<f64>() / pts.len() as f64;
        assert!(mean > 0.8 && mean < 2.5, "{mean}");
    }
}
