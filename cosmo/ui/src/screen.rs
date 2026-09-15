//! Mapa kursu i identyfikatory lekcji oraz laboratoriów.
//!
//! Fizyka tu nie mieszka. Ten moduł wie tylko, *gdzie* jesteśmy i dokąd można
//! przejść: trzy ścieżki, cztery chmury i stół zrzucania geodezyjnych. Tekst
//! lekcji rysuje [`crate::lesson`]. Silnik chmury zostaje w panelu sim-labu.

use eframe::egui::{self, Color32, RichText, Ui};

use crate::simulation::Mode;

/// Główny stan okna: akademia, nie zakładki.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Screen {
    #[default]
    Map,
    Lesson(LessonId),
    Lab(LabId),
}

/// Ścieżka na mapie kursu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Stw,
    Geo,
    Bh,
}

impl Track {
    pub const ALL: [Track; 3] = [Track::Stw, Track::Geo, Track::Bh];

    pub fn slug(self) -> &'static str {
        match self {
            Self::Stw => "stw",
            Self::Geo => "geo",
            Self::Bh => "bh",
        }
    }

    pub fn title(self) -> &'static str {
        match self {
            Self::Stw => "Szczególna teoria względności",
            Self::Geo => "Geodezyjna",
            Self::Bh => "Czarna dziura",
        }
    }

    pub fn blurb(self) -> &'static str {
        match self {
            Self::Stw => "Jednoczesność, stożek świetlny, Lorentz — potem laboratorium N-ciał.",
            Self::Geo => {
                "Idź prosto na zakrzywionej przestrzeni. Schwarzschild słowami, bez Γ na start."
            }
            Self::Bh => "Pierścienie 2M / 3M / 6M, soczewkowanie, obraz dysku.",
        }
    }

    pub fn lesson_count(self) -> u8 {
        match self {
            Self::Stw => 7,
            Self::Geo => 6,
            Self::Bh => 4,
        }
    }

    pub fn first(self) -> LessonId {
        LessonId {
            track: self,
            index: 1,
        }
    }

    pub fn lessons(self) -> impl Iterator<Item = LessonId> {
        (1..=self.lesson_count()).map(move |index| LessonId { track: self, index })
    }

    fn accent(self) -> Color32 {
        match self {
            Self::Stw => Color32::from_rgb(120, 180, 220),
            Self::Geo => Color32::from_rgb(160, 200, 140),
            Self::Bh => Color32::from_rgb(220, 180, 90),
        }
    }
}

/// Adres lekcji. Tytuł i kolejność ścieżki żyją tu; tekst i obraz są w [`crate::lesson`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LessonId {
    pub track: Track,
    pub index: u8,
}

impl LessonId {
    pub fn slug(self) -> String {
        format!("{}/{:02}", self.track.slug(), self.index)
    }

    pub fn title(self) -> &'static str {
        match (self.track, self.index) {
            (Track::Stw, 1) => "Dwa obserwatorzy i jedna błyskawica",
            (Track::Stw, 2) => "Scena teatralna: stożek świetlny",
            (Track::Stw, 3) => "Lorentz jako przechylanie osi",
            (Track::Stw, 4) => "Dylatacja czasu",
            (Track::Stw, 5) => "Kontrakcja długości",
            (Track::Stw, 6) => "4-wektor wydarzenia",
            (Track::Stw, 7) => "Laboratorium N-ciała",
            (Track::Geo, 1) => "Idź prosto na kuli",
            (Track::Geo, 2) => "Metryka — lokalna linijka",
            (Track::Geo, 3) => "Schwarzschild słowami",
            (Track::Geo, 4) => "Równanie geodezyjne",
            (Track::Geo, 5) => "RK4 jako film",
            (Track::Geo, 6) => "Laboratorium zrzucania",
            (Track::Bh, 1) => "Pierścienie 2M, 3M, 6M",
            (Track::Bh, 2) => "Soczewkowanie i pierścień Einsteina",
            (Track::Bh, 3) => "Raytracer: kamera i dysk",
            (Track::Bh, 4) => "Suwaki masy i nachylenia",
            _ => "Lekcja",
        }
    }

    pub fn prev(self) -> Option<Self> {
        (self.index > 1).then_some(Self {
            track: self.track,
            index: self.index - 1,
        })
    }

    pub fn next(self) -> Option<Self> {
        (self.index < self.track.lesson_count()).then_some(Self {
            track: self.track,
            index: self.index + 1,
        })
    }

    /// Ostatnia lekcja ścieżki A otwiera chmurę; B6 — stół zrzucania.
    pub fn opens_lab(self) -> Option<LabId> {
        match (self.track, self.index) {
            (Track::Stw, 7) => Some(LabId::Nbody),
            (Track::Geo, 6) => Some(LabId::Geodesics),
            _ => None,
        }
    }
}

/// Laboratoria na mapie. Chmury mają [`Mode`]; zrzucanie rysuje równik, nie punkty.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabId {
    Nbody,
    Cosmology,
    Particles,
    Atoms,
    Geodesics,
}

impl LabId {
    pub const ALL: [LabId; 5] = [
        LabId::Nbody,
        LabId::Cosmology,
        LabId::Particles,
        LabId::Atoms,
        LabId::Geodesics,
    ];

    pub const SIM: [LabId; 4] = [
        LabId::Nbody,
        LabId::Cosmology,
        LabId::Particles,
        LabId::Atoms,
    ];

    pub fn is_sim(self) -> bool {
        self.mode().is_some()
    }

    pub fn mode(self) -> Option<Mode> {
        match self {
            Self::Nbody => Some(Mode::Relativistic),
            Self::Cosmology => Some(Mode::Cosmological),
            Self::Particles => Some(Mode::Particles),
            Self::Atoms => Some(Mode::Atoms),
            Self::Geodesics => None,
        }
    }

    pub fn from_mode(mode: Mode) -> Self {
        match mode {
            Mode::Relativistic => Self::Nbody,
            Mode::Cosmological => Self::Cosmology,
            Mode::Particles => Self::Particles,
            Mode::Atoms => Self::Atoms,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Nbody => Mode::Relativistic.label(),
            Self::Cosmology => Mode::Cosmological.label(),
            Self::Particles => Mode::Particles.label(),
            Self::Atoms => Mode::Atoms.label(),
            Self::Geodesics => "Geodezyjne",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::Nbody => Mode::Relativistic.subtitle(),
            Self::Cosmology => Mode::Cosmological.subtitle(),
            Self::Particles => Mode::Particles.subtitle(),
            Self::Atoms => Mode::Atoms.subtitle(),
            Self::Geodesics => "zrzut w równiku Schwarzschilda",
        }
    }

    pub fn course_button(self) -> &'static str {
        match self {
            Self::Nbody => "Laboratorium N-ciała",
            Self::Geodesics => "Laboratorium geodezyjnych",
            _ => "Laboratorium",
        }
    }
}

/// Dokąd mapa chce przejść.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Lesson(LessonId),
    Lab(LabId),
}

/// Pasek nad lekcją i laboratorium. Przycisk zawsze wraca na mapę.
pub fn top_bar(ctx: &egui::Context, button: &str, caption: &str) -> bool {
    let mut clicked = false;
    egui::TopBottomPanel::top("academy-nav").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button(button).clicked() {
                clicked = true;
            }
            ui.separator();
            ui.label(RichText::new(caption).strong());
        });
    });
    clicked
}

/// Pasek „wstecz” nad laboratorium.
pub fn back_bar(ctx: &egui::Context, caption: &str) -> bool {
    top_bar(ctx, "Wstecz", caption)
}

pub fn draw_map(ui: &mut Ui) -> Option<Nav> {
    let mut nav = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        ui.add_space(12.0);
        ui.label(RichText::new("Bone — czasoprzestrzeń").size(22.0).strong());
        ui.label(
            RichText::new("Kurs: szczególna teoria względności, geodezyjna, czarna dziura")
                .small()
                .weak(),
        );
        ui.add_space(16.0);

        ui.label(RichText::new("Ścieżki").strong());
        ui.label(
            RichText::new(
                "Każda karta otwiera lekcje. Stub i placeholder — pełny tekst i fizyka później.",
            )
            .small()
            .weak(),
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            for track in Track::ALL {
                if let Some(next) = path_card(ui, track) {
                    nav = Some(next);
                }
            }
        });

        ui.add_space(20.0);
        ui.label(RichText::new("Laboratoria").strong());
        ui.label(
            RichText::new(
                "Cztery chmury jak wcześniej i stół zrzucania: foton albo cząstka w równiku.",
            )
            .small()
            .weak(),
        );
        ui.add_space(8.0);
        ui.horizontal_wrapped(|ui| {
            for lab in LabId::ALL {
                if let Some(next) = lab_card(ui, lab) {
                    nav = Some(next);
                }
            }
        });

        ui.add_space(24.0);
        ui.label(
            RichText::new("Następne działy (później): równania Einsteina, Kerr, PINN.")
                .small()
                .weak(),
        );
        ui.add_space(12.0);
    });
    nav
}

fn path_card(ui: &mut Ui, track: Track) -> Option<Nav> {
    let mut nav = None;
    // `horizontal_wrapped` dziedziczy układ w bok — bez `vertical` tytuł, blurb
    // i linki zlewałyby się w jeden akapit.
    ui.vertical(|ui| {
        ui.set_width(260.0);
        ui.group(|ui| {
            ui.set_min_width(240.0);
            ui.label(RichText::new(track.title()).strong().color(track.accent()));
            ui.label(RichText::new(track.blurb()).small().weak());
            ui.add_space(8.0);
            for lesson in track.lessons() {
                let line = format!("{:02}  {}", lesson.index, lesson.title());
                if ui.link(line).clicked() {
                    nav = Some(Nav::Lesson(lesson));
                }
            }
            ui.add_space(8.0);
            if ui.button("Zacznij").clicked() {
                nav = Some(Nav::Lesson(track.first()));
            }
        });
    });
    nav
}

fn lab_card(ui: &mut Ui, lab: LabId) -> Option<Nav> {
    let mut nav = None;
    ui.vertical(|ui| {
        ui.set_width(200.0);
        ui.group(|ui| {
            ui.set_min_width(180.0);
            ui.label(RichText::new(lab.label()).strong());
            ui.label(RichText::new(lab.subtitle()).small().weak());
            ui.add_space(8.0);
            if ui.button("Otwórz").clicked() {
                nav = Some(Nav::Lab(lab));
            }
        });
    });
    nav
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_starts_on_the_map() {
        assert_eq!(Screen::default(), Screen::Map);
    }

    #[test]
    fn three_tracks_have_the_planned_lesson_counts() {
        assert_eq!(Track::Stw.lesson_count(), 7);
        assert_eq!(Track::Geo.lesson_count(), 6);
        assert_eq!(Track::Bh.lesson_count(), 4);
        assert_eq!(Track::ALL.len(), 3);
    }

    #[test]
    fn every_lesson_has_a_unique_slug_and_a_title() {
        let mut slugs = std::collections::HashSet::new();
        for track in Track::ALL {
            assert_eq!(track.first().index, 1);
            for lesson in track.lessons() {
                assert!(!lesson.title().is_empty(), "{}", lesson.slug());
                assert_ne!(lesson.title(), "Lekcja", "brak tytułu: {}", lesson.slug());
                assert!(slugs.insert(lesson.slug()), "duplikat {}", lesson.slug());
            }
        }
        assert_eq!(slugs.len(), 7 + 6 + 4);
        assert_eq!(
            LessonId {
                track: Track::Stw,
                index: 1
            }
            .slug(),
            "stw/01"
        );
        assert_eq!(
            LessonId {
                track: Track::Bh,
                index: 4
            }
            .slug(),
            "bh/04"
        );
        assert_eq!(Track::Stw.first().prev(), None);
        assert_eq!(
            Track::Stw.first().next().map(|id| id.slug()),
            Some("stw/02".into())
        );
        assert_eq!(
            LessonId {
                track: Track::Stw,
                index: 7
            }
            .next(),
            None
        );
        assert_eq!(
            LessonId {
                track: Track::Stw,
                index: 7
            }
            .opens_lab(),
            Some(LabId::Nbody)
        );
        assert_eq!(Track::Stw.first().opens_lab(), None);
        assert_eq!(
            Track::Geo.lessons().last().unwrap().opens_lab(),
            Some(LabId::Geodesics)
        );
        assert_eq!(Track::Geo.first().opens_lab(), None);
    }

    #[test]
    fn sim_labs_are_exactly_the_four_existing_modes() {
        let modes: Vec<Mode> = LabId::SIM.iter().map(|lab| lab.mode().unwrap()).collect();
        assert_eq!(modes, Mode::ALL.to_vec());
        for lab in LabId::SIM {
            assert_eq!(LabId::from_mode(lab.mode().unwrap()), lab);
            assert!(lab.is_sim());
            assert!(!lab.label().is_empty());
            assert!(!lab.subtitle().is_empty());
        }
        assert_eq!(LabId::ALL.len(), 5);
        assert!(!LabId::Geodesics.is_sim());
        assert_eq!(LabId::Geodesics.mode(), None);
        assert_eq!(LabId::Geodesics.label(), "Geodezyjne");
        assert!(!LabId::Geodesics.subtitle().is_empty());
        assert_eq!(
            LabId::Geodesics.course_button(),
            "Laboratorium geodezyjnych"
        );
    }

    #[test]
    fn map_layout_without_a_window() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            assert_eq!(draw_map(ui), None);
        });
        let _ = ctx.end_pass();
    }
}
