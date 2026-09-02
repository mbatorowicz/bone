//! Wektor trójwymiarowy w podwójnej precyzji.
//!
//! Istnieje, bo bez niego integrator, chłodzenie i diagnostyka rozwijają każdą
//! operację na trzy niemal identyczne linie na składową. Taki kod nie tylko się
//! źle czyta — on się rozjeżdża: wystarczy jedna literówka `[1]` zamiast `[2]`
//! w jednym z trzydziestu miejsc, żeby symulacja liczyła coś prawie poprawnego,
//! czyli najgorszy możliwy rodzaj błędu. Tutaj składowe są przetwarzane raz.
//!
//! `norm` skaluje przez największą składową, zamiast liczyć `sqrt(x²+y²+z²)`
//! wprost. Naiwna wersja przepełnia się już przy składowych rzędu 1e154, a pęd
//! cząstki, która wpadła w rozbiegającą się konfigurację, potrafi tam trafić —
//! i wtedy `inf` zatruwa całą dalszą arytmetykę, zamiast pokazać dużą liczbę.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const ZERO: Vec3 = Vec3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

pub fn vec3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn get(self, axis: usize) -> f64 {
        match axis {
            0 => self.x,
            1 => self.y,
            _ => self.z,
        }
    }

    pub fn set(&mut self, axis: usize, value: f64) {
        match axis {
            0 => self.x = value,
            1 => self.y = value,
            _ => self.z = value,
        }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// Długość odporna na przepełnienie — patrz uwaga w nagłówku modułu.
    pub fn norm(self) -> f64 {
        let scale = self.x.abs().max(self.y.abs()).max(self.z.abs());
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        let s = self / scale;
        scale * s.norm_squared().sqrt()
    }

    pub fn max_abs_component(self) -> f64 {
        self.x.abs().max(self.y.abs()).max(self.z.abs())
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn min_each(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    pub fn max_each(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }

    pub fn from_f32(v: [f32; 3]) -> Self {
        Self {
            x: v[0] as f64,
            y: v[1] as f64,
            z: v[2] as f64,
        }
    }
}

impl Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;
    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    fn div(self, s: f64) -> Self {
        Self {
            x: self.x / s,
            y: self.y / s,
            z: self.z / s,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl SubAssign for Vec3 {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl std::iter::Sum for Vec3 {
    fn sum<I: Iterator<Item = Vec3>>(iter: I) -> Self {
        iter.fold(ZERO, |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_survives_overflow_scale() {
        let huge = vec3(1e200, 1e200, 0.0);
        let n = huge.norm();
        assert!(n.is_finite(), "norma przepełniła się: {n}");
        assert!((n / (1e200 * 2.0_f64.sqrt()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn norm_of_zero_is_zero() {
        assert_eq!(ZERO.norm(), 0.0);
    }

    #[test]
    fn cross_is_right_handed() {
        let c = vec3(1.0, 0.0, 0.0).cross(vec3(0.0, 1.0, 0.0));
        assert_eq!(c, vec3(0.0, 0.0, 1.0));
    }

    #[test]
    fn axis_access_round_trips() {
        let mut v = ZERO;
        for axis in 0..3 {
            v.set(axis, axis as f64 + 1.0);
        }
        assert_eq!(v, vec3(1.0, 2.0, 3.0));
        for axis in 0..3 {
            assert_eq!(v.get(axis), axis as f64 + 1.0);
        }
    }
}
