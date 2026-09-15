//! Ruchomy obraz lekcji: liczby z `bone_core::gr`, pędzel w `egui`.
//!
//! Fizyka nie mieszka tu. Ten katalog składa znaczniki z [`bone_core::gr::lorentz`]
//! w diagram, który da się ruszyć suwakiem. Minkowski (lekcje STW 1–3) jest
//! pierwszy; zegary, siatka i raytracer dojdą w kolejnych krokach.

pub mod minkowski;

use eframe::egui::Ui;

use crate::lesson::Playback;
use crate::screen::{LessonId, Track};

/// Domyślne β na suwaku: γ = 5/4, ten sam podręcznikowy punkt co testy `gr::lorentz`.
pub const BETA_DEFAULT: f64 = 0.6;
/// Suwak nie dochodzi do 1 — transformacja Lorentza wtedy nie istnieje.
pub const BETA_MAX: f64 = 0.95;

/// `true`, gdy lekcja ma własny obraz. Reszta zostaje przy placeholdrze.
pub fn draw(ui: &mut Ui, id: LessonId, playback: &mut Playback) -> bool {
    match (id.track, id.index) {
        (Track::Stw, 1..=3) => {
            minkowski::draw(ui, id.index, playback);
            true
        }
        _ => false,
    }
}
