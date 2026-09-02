//! Wspólne chmury testowe — jedna definicja zamiast kopii w solverach.

use crate::vec3::{vec3, Vec3};

/// Jednorodna kostka o boku `size`, `per_side³` cząstek o masie 1.
pub fn cube_cloud(per_side: usize, size: f64) -> (Vec<Vec3>, Vec<f64>) {
    let h = size / per_side as f64;
    let mut positions = Vec::with_capacity(per_side.pow(3));
    for i in 0..per_side {
        for j in 0..per_side {
            for k in 0..per_side {
                positions.push(vec3(
                    (i as f64 + 0.5) * h,
                    (j as f64 + 0.5) * h,
                    (k as f64 + 0.5) * h,
                ));
            }
        }
    }
    let masses = vec![1.0; positions.len()];
    (positions, masses)
}
