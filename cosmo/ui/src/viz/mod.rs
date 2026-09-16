//! Ruchomy obraz lekcji: liczby z `bone_core::gr`, pędzel w `egui`.
//!
//! Fizyka nie mieszka tu. Ten katalog składa znaczniki z [`bone_core::gr::lorentz`]
//! w diagram, który da się ruszyć suwakiem. Minkowski to lekcje STW 1–3;
//! zegary, linijka i 4-wektor to 4–6; lekcja 7 otwiera laboratorium N-ciał.
//! Ścieżka geodezyjna 1–5: kula, gumowa siatka, film RK4 vs Euler.
//! Lekcja 6 otwiera stół zrzucania; sam równik rysuje [`geodesics`].
//! Ścieżka czarnej dziury 1–4: pierścienie, pęk i Einstein w 2D ([`rings`]).
//! C3 i C4 otwierają laboratorium raytracera ([`blackhole`]): klatka w tle.
//! Tensory, Einstein i PINN zostają przy placeholdrze — krok 16, bez fizyki.

pub mod blackhole;
pub mod clocks;
pub mod geodesics;
pub mod metric_grid;
pub mod minkowski;
pub mod rings;
pub mod rk4_film;
pub mod sphere;

use eframe::egui::Ui;

use crate::lesson::Playback;
use crate::screen::{LabId, LessonId, Track};

/// Domyślne β na suwaku: γ = 5/4, ten sam podręcznikowy punkt co testy `gr::lorentz`.
pub const BETA_DEFAULT: f64 = 0.6;
/// Suwak nie dochodzi do 1 — transformacja Lorentza wtedy nie istnieje.
pub const BETA_MAX: f64 = 0.95;
/// Domyślne M: te same 2M / 3M / 6M co testy [`bone_core::gr::metric`].
pub const MASS_DEFAULT: f64 = 1.0;
/// Suwak masy zostawia zapas, żeby 6M mieściło się na siatce o zasięgu 16.
pub const MASS_MAX: f64 = 2.5;
/// Startowy obrót globusa: widać oba ślady, nie sam biegun.
pub const SPIN_DEFAULT: f32 = 0.85;

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
        (Track::Geo, 1) => {
            sphere::draw(ui, playback);
            Outcome::Drawn
        }
        (Track::Geo, 2..=4) => {
            metric_grid::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Geo, 5) => {
            rk4_film::draw(ui, playback);
            Outcome::Drawn
        }
        (Track::Geo, 6) => {
            if geodesics::draw_drop_door(ui) {
                Outcome::OpenLab(LabId::Geodesics)
            } else {
                Outcome::Drawn
            }
        }
        (Track::Bh, 1..=2) => {
            rings::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Bh, 3 | 4) => {
            let open = blackhole::draw_ray_door(ui);
            rings::draw(ui, id.index, playback);
            if open {
                Outcome::OpenLab(LabId::BlackHole)
            } else {
                Outcome::Drawn
            }
        }
        _ => Outcome::Placeholder,
    }
}
