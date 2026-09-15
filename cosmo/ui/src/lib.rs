//! Okno desktopowe: mapa kursu, a potem lekcja albo laboratorium.
//!
//! Podział na moduły idzie po tym, co się zmienia niezależnie:
//!
//! - [`screen`] — mapa, identyfikatory ścieżek i labów,
//! - [`lesson`] — ekran 60/40, parser Markdown, play/pauza,
//! - [`viz`] — ruchomy obraz lekcji (STW 1–7, geodezyjna 1–5; zrzucanie to osobny krok),
//! - [`camera`] — obrót, przesunięcie i przybliżenie, czysta geometria,
//! - [`render`] — chmura punktów na obraz, czysta arytmetyka,
//! - [`panels`] — formularz i tabela laboratorium, jedyne miejsce formularza `egui`,
//! - [`simulation`] — cztery modele pod jednym interfejsem,
//! - ten moduł — pętla klatek i decyzje: który ekran, kiedy startować, kiedy liczyć.
//!
//! Ten podział jest warunkiem testowalności, nie porządkiem dla porządku: kamera,
//! renderer i opis stanu biegu nie dotykają `egui`, więc dają się sprawdzić bez okna
//! i bez karty graficznej. Rzut na ekran i mapa kolorów wymieszane z rysowaniem
//! byłyby sprawdzalne wyłącznie okiem.

pub mod camera;
pub mod charts;
pub mod lesson;
pub mod panels;
pub mod render;
pub mod replay;
pub mod screen;
pub mod simulation;
pub mod viz;

use std::time::Instant;

use eframe::egui::{self, Sense, TextureHandle, TextureOptions};

use crate::camera::Camera;
use crate::lesson::Playback;
use crate::panels::{Action, Setup};
use crate::replay::Replay;
use crate::screen::{LabId, LessonId, Nav, Screen};
use crate::simulation::{Mode, View};
use bone_core::session::Session;

/// Ile klatek odczekać na najniższym ustawieniu szybkości.
///
/// Suwak w pozycji zero nie znaczy „stop", a „jeden krok na osiem klatek" — przy
/// zapadaniu się chmury pojedynczy krok potrafi zmienić obraz na tyle, że pełne
/// tempo jest nie do obejrzenia.
const SLOW_HOLD_FRAMES: u32 = 8;

pub struct App {
    screen: Screen,
    lesson: Playback,
    setup: Setup,
    view: Option<View>,
    running: bool,
    camera: Camera,
    texture: Option<TextureHandle>,
    status: String,
    error: String,
    warnings: Vec<String>,
    slow_hold: u32,
    ms_per_step: f64,
}

impl Default for App {
    fn default() -> Self {
        Self {
            screen: Screen::Map,
            lesson: Playback::default(),
            setup: Setup::default(),
            view: None,
            running: false,
            camera: Camera::default(),
            texture: None,
            status: "Gotowe. Wybierz model i uruchom — liczy ten komputer.".to_string(),
            error: String::new(),
            warnings: Vec::new(),
            slow_hold: 0,
            ms_per_step: 0.0,
        }
    }
}

impl App {
    /// Zbuduj bieg od nowa.
    ///
    /// Budowa jest owinięta w `catch_unwind`, bo tu alokują się tablice o rozmiarze
    /// wprost zależnym od suwaka: siatka 64³ z rozszerzeniem Hockneya to 2 mln
    /// komórek zespolonych. Panic z braku pamięci ma zostać komunikatem w panelu,
    /// a nie zniknięciem okna bez śladu.
    fn start(&mut self) {
        self.error.clear();
        self.warnings.clear();
        self.view = None;
        self.status = "Składanie warunków początkowych…".to_string();
        let out = std::path::PathBuf::from(&self.setup.out_dir);
        let record = self.setup.record;

        let built =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match self.setup.mode {
                Mode::Relativistic => Session::start_sr(self.setup.sr.clone(), out, record),
                Mode::Cosmological => Session::start_lcdm(self.setup.lcdm, out, record),
                Mode::Particles => Session::start_sm(self.setup.sm.clone(), out, record),
                Mode::Atoms => Session::start_qm(self.setup.qm.clone(), out, record),
            }));

        match built {
            Ok(Ok(session)) => {
                let mut view = View::Live(session);
                self.warnings = view.take_warnings();
                self.status = view.headline();
                self.view = Some(view);
                self.running = true;
            }
            Ok(Err(message)) => {
                self.error = message;
                self.running = false;
            }
            Err(_) => {
                self.error = "Nie udało się złożyć warunków początkowych — za duża siatka albo \
                     za mało pamięci."
                    .to_string();
                self.running = false;
            }
        }
    }

    /// Ile kroków policzyć w tej klatce; `None`, gdy jeszcze czekamy.
    fn steps_this_frame(&mut self) -> Option<u32> {
        if self.setup.speed > 0 {
            self.slow_hold = 0;
            return Some(self.setup.speed);
        }
        self.slow_hold += 1;
        if self.slow_hold < SLOW_HOLD_FRAMES {
            return None;
        }
        self.slow_hold = 0;
        Some(1)
    }

    fn tick(&mut self) {
        if !self.running {
            return;
        }
        let Some(View::Live(_)) = self.view else {
            return;
        };
        let Some(steps) = self.steps_this_frame() else {
            return;
        };
        // Nastawy runtime'owe wchodzą w życie bez restartu; startowe (liczba cząstek,
        // ziarno, geometria) są ignorowane — tym zajmuje się `Config::with_runtime_from`.
        let live_sr = self.setup.sr.clone();
        let live_dlna = self.setup.lcdm.dlna;
        let live_sm = self.setup.sm.clone();
        let live_qm = self.setup.qm.clone();
        let session = self
            .view
            .as_mut()
            .and_then(View::as_live_mut)
            .expect("żywy bieg");
        session.apply_runtime_sr(&live_sr);
        session.apply_runtime_lcdm(live_dlna);
        session.apply_runtime_sm(&live_sm);
        session.apply_runtime_qm(&live_qm);

        let started = Instant::now();
        let outcome = session.advance(steps);
        self.ms_per_step = started.elapsed().as_secs_f64() * 1000.0 / steps as f64;

        match outcome {
            Ok(report) => {
                self.warnings.extend(report.warnings);
                self.status = format!("{}  Δt={:.0} ms", report.headline, self.ms_per_step);
                if report.finished {
                    self.running = false;
                }
            }
            Err(message) => {
                self.error = message;
                self.running = false;
            }
        }
    }

    fn resume(&mut self) {
        self.error.clear();
        self.warnings.clear();
        let out = std::path::PathBuf::from(&self.setup.out_dir);
        match Session::resume(out, self.setup.record) {
            Ok(session) => {
                let mut view = View::Live(session);
                self.warnings = view.take_warnings();
                self.status = view.headline();
                self.view = Some(view);
                self.running = true;
            }
            Err(message) => {
                self.error = message;
                self.running = false;
            }
        }
    }

    fn replay(&mut self) {
        self.error.clear();
        self.warnings.clear();
        self.running = false;
        match Replay::open(&self.setup.out_dir) {
            Ok(replay) => {
                let view = View::Replay(replay);
                self.status = view.headline();
                self.view = Some(view);
            }
            Err(message) => self.error = message,
        }
    }

    fn open_lab(&mut self, lab: LabId) {
        if LabId::from_mode(self.setup.mode) != lab {
            if self.view.is_some() {
                self.stop();
            } else {
                self.running = false;
            }
        }
        self.setup.mode = lab.mode();
        self.screen = Screen::Lab(lab);
    }

    /// Lekcja STW 7: preset relatywistyczny, kinematyka SR, karta „to nie OTW”.
    fn open_course_lab(&mut self, lab: LabId) {
        if lab == LabId::Nbody {
            if self.view.is_some() {
                self.stop();
            }
            self.setup.apply_stw_nbody_door();
        }
        self.open_lab(lab);
    }

    fn open_lesson(&mut self, id: LessonId) {
        self.running = false;
        self.lesson.reset();
        self.screen = Screen::Lesson(id);
    }

    fn back_to_map(&mut self) {
        self.running = false;
        self.screen = Screen::Map;
    }

    fn follow_nav(&mut self, nav: Nav) {
        match nav {
            Nav::Lesson(id) => self.open_lesson(id),
            Nav::Lab(lab) => self.open_lab(lab),
        }
    }

    fn stop(&mut self) {
        if let Some(View::Live(session)) = self.view.take() {
            if self.setup.record {
                let mut session = session;
                let _ = session.flush_recording();
                if let Err(e) = session.save_checkpoint() {
                    self.error = format!("zapis checkpointu: {e}");
                }
            }
        }
        self.running = false;
        self.warnings.clear();
        self.status = "Zatrzymano.".to_string();
    }

    fn draw_cloud(&mut self, ui: &mut egui::Ui) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            self.camera
                .pan_by(delta.x, delta.y, rect.width(), rect.height());
        } else if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.camera.orbit(delta.x, delta.y);
        }
        let scroll = ui.input(|i| i.raw_scroll_delta.y);
        if scroll != 0.0 {
            self.camera.zoom_by(scroll);
        }

        let image = render::render(
            self.view.as_ref().map(|view| view.cloud()),
            &self.camera,
            rect.width() as usize,
            rect.height() as usize,
        );
        let texture = match &mut self.texture {
            Some(existing) if existing.size() == image.size => {
                existing.set(image, TextureOptions::LINEAR);
                existing
            }
            slot => {
                *slot = Some(
                    ui.ctx()
                        .load_texture("cloud", image, TextureOptions::LINEAR),
                );
                slot.as_mut().expect("tekstura po zapisie")
            }
        };
        // `paint_at` rysuje w już przydzielonym prostokącie. `Image` jako widget
        // alokowałby go drugi raz i przy `maintain_aspect_ratio` (domyślnie włączone)
        // potrafiłby złożyć obraz do paska albo wcale go nie pokazać.
        egui::Image::new(&*texture)
            .maintain_aspect_ratio(false)
            .fit_to_exact_size(rect.size())
            .paint_at(ui, rect);
    }

    fn draw_lab(&mut self, ctx: &egui::Context) {
        let action = egui::SidePanel::left("panel")
            .resizable(false)
            .min_width(300.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .show(ui, |ui| {
                        panels::side_panel(
                            ui,
                            &mut self.setup,
                            self.view.as_mut(),
                            self.running,
                            &self.status,
                            &self.error,
                            &self.warnings,
                        )
                    })
                    .inner
            })
            .inner;

        match action {
            Action::Start => self.start(),
            Action::TogglePause => {
                self.running = !self.running && matches!(self.view, Some(View::Live(_)));
            }
            Action::Stop => self.stop(),
            Action::Resume => self.resume(),
            Action::Replay => self.replay(),
            Action::None => {}
        }

        if let Screen::Lab(_) = self.screen {
            self.screen = Screen::Lab(LabId::from_mode(self.setup.mode));
        }

        egui::CentralPanel::default().show(ctx, |ui| self.draw_cloud(ui));
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if matches!(self.screen, Screen::Lab(_)) {
            self.tick();
            if self.running {
                ctx.request_repaint();
            }
        }
        if let Screen::Lesson(_) = self.screen {
            let dt = ctx.input(|i| i.stable_dt);
            self.lesson.tick(dt);
            if self.lesson.playing {
                ctx.request_repaint();
            }
        }

        match self.screen {
            Screen::Map => {
                let mut nav = None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    nav = screen::draw_map(ui);
                });
                if let Some(nav) = nav {
                    self.follow_nav(nav);
                }
            }
            Screen::Lesson(id) => {
                let to_map = screen::top_bar(ctx, "Mapa", id.title());
                let mut action = lesson::Action::None;
                egui::CentralPanel::default().show(ctx, |ui| {
                    action = lesson::draw(ui, id, &mut self.lesson);
                });
                if matches!(action, lesson::Action::None) && to_map {
                    action = lesson::Action::Map;
                }
                match action {
                    lesson::Action::Map => self.back_to_map(),
                    lesson::Action::Lesson(next) => self.open_lesson(next),
                    lesson::Action::Lab(lab) => self.open_course_lab(lab),
                    lesson::Action::None => {}
                }
            }
            Screen::Lab(lab) => {
                if screen::back_bar(ctx, lab.label()) {
                    self.back_to_map();
                    return;
                }
                self.draw_lab(ctx);
            }
        }
    }
}

/// # Errors
/// Gdy nie udaje się otworzyć okna — najczęściej z braku sterownika GPU.
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title("Bone — czasoprzestrzeń"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Bone",
        options,
        Box::new(|_cc| Ok(Box::new(App::default()))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slow_speed_holds_frames_before_stepping() {
        let mut app = App::default();
        app.setup.speed = 0;
        let mut stepped = 0;
        for _ in 0..SLOW_HOLD_FRAMES * 3 {
            if app.steps_this_frame().is_some() {
                stepped += 1;
            }
        }
        assert_eq!(stepped, 3, "krok wypadł {stepped} razy zamiast 3");
    }

    #[test]
    fn normal_speed_steps_every_frame() {
        let mut app = App::default();
        app.setup.speed = 4;
        for _ in 0..5 {
            assert_eq!(app.steps_this_frame(), Some(4));
        }
    }

    /// Zmiana szybkości z zera na wyższą nie może zostawić licznika oczekiwania —
    /// inaczej pierwszy krok po zmianie byłby opóźniony bez powodu.
    #[test]
    fn raising_the_speed_clears_the_hold() {
        let mut app = App::default();
        app.setup.speed = 0;
        app.steps_this_frame();
        assert!(app.slow_hold > 0);
        app.setup.speed = 2;
        app.steps_this_frame();
        assert_eq!(app.slow_hold, 0);
    }

    #[test]
    fn tick_without_a_run_does_nothing() {
        let mut app = App {
            running: true,
            ..App::default()
        };
        app.tick();
        assert!(app.error.is_empty());
        assert_eq!(app.ms_per_step, 0.0);
    }

    #[test]
    fn the_app_opens_on_the_course_map() {
        let app = App::default();
        assert_eq!(app.screen, Screen::Map);
        assert!(!matches!(app.screen, Screen::Lab(_)));
    }

    #[test]
    fn each_lab_opens_from_the_map_and_returns() {
        let mut app = App::default();
        for lab in LabId::ALL {
            app.open_lab(lab);
            assert_eq!(app.screen, Screen::Lab(lab));
            assert_eq!(app.setup.mode, lab.mode());
            assert_ne!(app.status, "Zatrzymano.");
            app.back_to_map();
            assert_eq!(app.screen, Screen::Map);
            assert!(!app.running);
        }
    }

    #[test]
    fn a_path_opens_a_lesson_and_stw_walks_into_nbody() {
        let mut app = App::default();
        let mut id = screen::Track::Stw.first();
        app.open_lesson(id);
        assert_eq!(app.screen, Screen::Lesson(id));
        assert!(app.lesson.playing);
        let mut hops = 0;
        while let Some(next) = id.next() {
            app.open_lesson(next);
            id = next;
            hops += 1;
        }
        assert_eq!(hops, 6);
        assert_eq!(id.index, 7);
        assert_eq!(lesson::step(id, true), lesson::Action::Lab(LabId::Nbody));
        app.open_course_lab(LabId::Nbody);
        assert_eq!(app.screen, Screen::Lab(LabId::Nbody));
        assert_eq!(app.setup.mode, Mode::Relativistic);
        assert_eq!(app.setup.sr_preset, "relativistic");
        assert_eq!(
            app.setup.sr.physics.kinematics,
            bone_core::sr::relativity::Kinematics::Sr
        );
        assert_eq!(
            crate::simulation::nbody_card(app.setup.sr.physics.kinematics).not_this,
            crate::simulation::NBODY_NOT_GR
        );
        app.back_to_map();
        assert_eq!(app.screen, Screen::Map);
        assert!(!app.running);
    }

    #[test]
    fn switching_lab_stops_the_previous_run() {
        let mut app = App::default();
        app.open_lab(LabId::Nbody);
        app.running = true;
        app.open_lab(LabId::Atoms);
        assert!(!app.running);
        assert_eq!(app.screen, Screen::Lab(LabId::Atoms));
        assert_eq!(app.setup.mode, Mode::Atoms);
        assert_ne!(app.status, "Zatrzymano.");
    }
}
