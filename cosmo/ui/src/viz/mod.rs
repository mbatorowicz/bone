//! Ruchomy obraz lekcji: liczby z `bone_core::gr`, pędzel w `egui`.
//!
//! Fizyka nie mieszka tu. Ten katalog składa znaczniki z [`bone_core::gr::lorentz`]
//! w diagram, który da się ruszyć suwakiem. Minkowski to lekcje STW 1–3;
//! zegary, linijka i 4-wektor to 4–6; lekcja 7 otwiera laboratorium N-ciał.

pub mod clocks;
pub mod minkowski;

use eframe::egui::Ui;

use crate::lesson::Playback;
use crate::screen::{LabId, LessonId, Track};

/// Domyślne β na suwaku: γ = 5/4, ten sam podręcznikowy punkt co testy `gr::lorentz`.
pub const BETA_DEFAULT: f64 = 0.6;
/// Suwak nie dochodzi do 1 — transformacja Lorentza wtedy nie istnieje.
pub const BETA_MAX: f64 = 0.95;

/// Co obraz lekcji zrobił z klatką.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Placeholder,
    Drawn,
    OpenLab(LabId),
}

/// `Placeholder`, gdy lekcja nie ma jeszcze własnego obrazu.
pub fn draw(ui: &mut Ui, id: LessonId, playback: &mut Playback) -> Outcome {
    match (id.track, id.index) {
        (Track::Stw, 1..=3) => {
            minkowski::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Stw, 4..=6) => {
            clocks::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Stw, 7) => {
            if clocks::draw_nbody_door(ui) {
                Outcome::OpenLab(LabId::Nbody)
            } else {
                Outcome::Drawn
            }
        }
        _ => Outcome::Placeholder,
    }
}
