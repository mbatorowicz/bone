//! Chmura próbek `|ψ|²`: położenia i odcień. Czas jest fazą superpozycji.

use crate::vec3::Vec3;

#[derive(Clone, Debug)]
pub struct State {
    pub time: f64,
    pub step: u64,
    pub positions: Vec<Vec3>,
    pub shades: Vec<f32>,
}

impl State {
    pub fn new(positions: Vec<Vec3>, shades: Vec<f32>) -> Result<Self, String> {
        if positions.len() != shades.len() {
            return Err(format!(
                "położeń {} i odcieni {} — muszą być tej samej długości",
                positions.len(),
                shades.len()
            ));
        }
        Ok(Self {
            time: 0.0,
            step: 0,
            positions,
            shades,
        })
    }

    pub fn n(&self) -> usize {
        self.positions.len()
    }

    pub fn is_finite(&self) -> bool {
        self.positions.iter().all(|p| p.is_finite())
            && self.shades.iter().all(|s| s.is_finite())
            && self.time.is_finite()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::vec3;

    #[test]
    fn mismatched_lengths_are_an_error() {
        let err = State::new(vec![vec3(0.0, 0.0, 0.0)], vec![]).unwrap_err();
        assert!(err.contains("odcieni"));
    }
}
