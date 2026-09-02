//! Stan układu: położenia, PĘDY i masy spoczynkowe.
//!
//! Trzymanie pędu zamiast prędkości nie jest kosmetyką. Leapfrog relatywistyczny
//! działa na `dp/dt = F`, więc pęd jest naturalną zmienną stanu; wariant trzymający
//! `v` musiałby przy każdym półkroku robić przejście v→p→v, co kosztuje dwa
//! pierwiastki i traci cyfry znaczące bez żadnego zysku.

use crate::sr::relativity as sr;
use crate::vec3::{Vec3, ZERO};

/// Pole sił i potencjał z jednego wywołania solvera.
#[derive(Clone, Debug)]
pub struct Field {
    pub force: Vec<Vec3>,
    /// Potencjał właściwy φᵢ (nie energia).
    pub potential: Vec<f64>,
}

impl Field {
    pub fn zeros(n: usize) -> Self {
        Self {
            force: vec![ZERO; n],
            potential: vec![0.0; n],
        }
    }

    /// `U = ½ Σ mᵢφᵢ` — połowa, bo każda para liczy się dwa razy.
    pub fn energy(&self, masses: &[f64]) -> f64 {
        0.5 * masses
            .iter()
            .zip(self.potential.iter())
            .map(|(m, phi)| m * phi)
            .sum::<f64>()
    }
}

#[derive(Clone, Debug)]
pub struct State {
    pub positions: Vec<Vec3>,
    pub momenta: Vec<Vec3>,
    /// Masa spoczynkowa.
    pub masses: Vec<f64>,
    pub time: f64,
    pub step: u64,
    /// Siła z ostatniego wywołania solvera. Leapfrog KDK potrzebuje jednego
    /// policzenia siły na krok, o ile przenosi ją między krokami.
    pub forces: Option<Vec<Vec3>>,
    pub potential: Option<Vec<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateError {
    ShapeMismatch { positions: usize, momenta: usize },
    MassCountMismatch { particles: usize, masses: usize },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { positions, momenta } => write!(
                f,
                "positions ({positions}) i momenta ({momenta}) muszą mieć tę samą długość"
            ),
            Self::MassCountMismatch { particles, masses } => write!(
                f,
                "liczba mas ({masses}) nie zgadza się z liczbą cząstek ({particles})"
            ),
        }
    }
}

impl std::error::Error for StateError {}

impl State {
    pub fn new(
        positions: Vec<Vec3>,
        momenta: Vec<Vec3>,
        masses: Vec<f64>,
    ) -> Result<Self, StateError> {
        if positions.len() != momenta.len() {
            return Err(StateError::ShapeMismatch {
                positions: positions.len(),
                momenta: momenta.len(),
            });
        }
        if positions.len() != masses.len() {
            return Err(StateError::MassCountMismatch {
                particles: positions.len(),
                masses: masses.len(),
            });
        }
        Ok(Self {
            positions,
            momenta,
            masses,
            time: 0.0,
            step: 0,
            forces: None,
            potential: None,
        })
    }

    pub fn n(&self) -> usize {
        self.positions.len()
    }

    pub fn total_mass(&self) -> f64 {
        self.masses.iter().sum()
    }

    pub fn velocity(&self, i: usize, c: f64) -> Vec3 {
        sr::velocity(self.masses[i], self.momenta[i], c)
    }

    pub fn gamma(&self, i: usize, c: f64) -> f64 {
        sr::gamma(self.masses[i], self.momenta[i], c)
    }

    pub fn speed_over_c(&self, i: usize, c: f64) -> f64 {
        sr::speed_over_c(self.masses[i], self.momenta[i], c)
    }

    pub fn velocities(&self, c: f64) -> Vec<Vec3> {
        (0..self.n()).map(|i| self.velocity(i, c)).collect()
    }

    pub fn center_of_mass(&self) -> Vec3 {
        let total = self.total_mass();
        if total <= 0.0 {
            return ZERO;
        }
        let weighted: Vec3 = self
            .positions
            .iter()
            .zip(self.masses.iter())
            .map(|(p, m)| *p * *m)
            .sum();
        weighted / total
    }

    pub fn total_momentum(&self) -> Vec3 {
        self.momenta.iter().copied().sum()
    }

    /// Skumulowany moment pędu względem środka masy.
    pub fn angular_momentum(&self) -> Vec3 {
        let com = self.center_of_mass();
        self.positions
            .iter()
            .zip(self.momenta.iter())
            .map(|(x, p)| (*x - com).cross(*p))
            .sum()
    }

    /// Unieważnij zapamiętane pole sił — po zmianie G albo ε przestaje być aktualne.
    pub fn invalidate_field(&mut self) {
        self.forces = None;
        self.potential = None;
    }

    pub fn is_finite(&self) -> bool {
        self.positions.iter().all(|p| p.is_finite())
            && self.momenta.iter().all(|p| p.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    fn two_body() -> State {
        State::new(
            vec![vec3(-1.0, 0.0, 0.0), vec3(1.0, 0.0, 0.0)],
            vec![vec3(0.0, 2.0, 0.0), vec3(0.0, -2.0, 0.0)],
            vec![1.0, 3.0],
        )
        .unwrap()
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let bad = State::new(vec![ZERO], vec![ZERO, ZERO], vec![1.0, 1.0]);
        assert!(matches!(bad, Err(StateError::ShapeMismatch { .. })));
        let bad = State::new(vec![ZERO, ZERO], vec![ZERO, ZERO], vec![1.0]);
        assert!(matches!(bad, Err(StateError::MassCountMismatch { .. })));
    }

    #[test]
    fn center_of_mass_is_weighted() {
        let s = two_body();
        // masy 1 i 3 na ±1 dają środek w 0.5
        assert!((s.center_of_mass().x - 0.5).abs() < 1e-12);
    }

    #[test]
    fn total_momentum_cancels_here() {
        assert!(two_body().total_momentum().norm() < 1e-12);
    }

    #[test]
    fn angular_momentum_is_nonzero_for_counter_rotation() {
        assert!(two_body().angular_momentum().norm() > 0.0);
    }

    #[test]
    fn field_energy_halves_the_sum() {
        let field = Field {
            force: vec![ZERO; 2],
            potential: vec![-2.0, -4.0],
        };
        // ½·(1·(−2) + 3·(−4)) = −7
        assert!((field.energy(&[1.0, 3.0]) + 7.0).abs() < 1e-12);
    }

    #[test]
    fn detects_non_finite_state() {
        let mut s = two_body();
        assert!(s.is_finite());
        s.momenta[0] = vec3(f64::NAN, 0.0, 0.0);
        assert!(!s.is_finite());
    }
}
