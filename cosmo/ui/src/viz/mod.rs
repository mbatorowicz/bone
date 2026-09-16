//! Ruchomy obraz lekcji: liczby z `bone_core::gr`, pędzel w `egui`.
//!
//! Fizyka nie mieszka tu. Ten katalog składa znaczniki z [`bone_core::gr::lorentz`]
//! w diagram, który da się ruszyć suwakiem. Minkowski to lekcje STW 1–3;
//! zegary, linijka i 4-wektor to 4–6; lekcja 7 otwiera laboratorium N-ciał.
//! Ścieżka geodezyjna 1–5: kula, gumowa siatka, film RK4 vs Euler.
//! Lekcja 6 otwiera stół zrzucania; sam równik rysuje [`geodesics`].
//! Ścieżka czarnej dziury 1–4: pierścienie, pęk i Einstein w 2D ([`rings`]).
//! C3 i C4 otwierają laboratorium raytracera ([`blackhole`]): klatka w tle.
//! Tensory, Einstein i PINN: algebra, pole i residual — liczby z `gr`,
//! obraz w [`tensors`], [`einstein`], [`pinn`]. Kerr i siatka PDE są na mapie
//! jako stuby: obraz to placeholder, bez `gr::kerr` i bez suwaka `a`.

pub mod blackhole;
pub mod clocks;
pub mod einstein;
pub mod geodesics;
pub mod metric_grid;
pub mod minkowski;
pub mod pinn;
pub mod rings;
pub mod rk4_film;
pub mod sphere;
pub mod tensors;

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
/// Sonda na łące: `r = 6` przy `M = 1` to ISCO z testów metryki.
pub const PROBE_DEFAULT: f64 = 6.0;

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
        (Track::Tensor, 1..=5) => {
            tensors::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Einstein, 1..=4) => {
            einstein::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        (Track::Pinn, 1..=4) => {
            pinn::draw(ui, id.index, playback);
            Outcome::Drawn
        }
        // Kerr / PDE: stub na mapie, obraz to placeholder (kroki 31 i 35).
        _ => Outcome::Placeholder,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lesson::Playback;

    #[test]
    fn next_tracks_draw_instead_of_placeholder() {
        for track in Track::NEXT {
            for id in track.lessons() {
                let ctx = eframe::egui::Context::default();
                ctx.begin_pass(eframe::egui::RawInput::default());
                eframe::egui::CentralPanel::default().show(&ctx, |ui| {
                    let mut playback = Playback::default();
                    assert_eq!(draw(ui, id, &mut playback), Outcome::Drawn, "{}", id.slug());
                });
                let _ = ctx.end_pass();
            }
        }
    }

    #[test]
    fn later_tracks_keep_the_placeholder() {
        for track in Track::LATER {
            for id in track.lessons() {
                let ctx = eframe::egui::Context::default();
                ctx.begin_pass(eframe::egui::RawInput::default());
                eframe::egui::CentralPanel::default().show(&ctx, |ui| {
                    let mut playback = Playback::default();
                    assert_eq!(
                        draw(ui, id, &mut playback),
                        Outcome::Placeholder,
                        "{}",
                        id.slug()
                    );
                });
                let _ = ctx.end_pass();
            }
        }
    }
}
