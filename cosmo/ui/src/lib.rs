//! Okno desktopowe: panel z nastawami i rzut chmury.
//!
//! Podział na moduły idzie po tym, co się zmienia niezależnie:
//!
//! - [`camera`] — obrót i przybliżenie, czysta geometria,
//! - [`render`] — chmura punktów na obraz, czysta arytmetyka,
//! - [`panels`] — formularz i tabela, jedyne miejsce dotykające `egui`,
//! - [`simulation`] — dwa modele pod jednym interfejsem,
//! - ten moduł — pętla klatek i decyzje: kiedy startować, kiedy liczyć.
//!
//! Ten podział jest warunkiem testowalności, nie porządkiem dla porządku: kamera,
//! renderer i opis stanu biegu nie dotykają `egui`, więc dają się sprawdzić bez okna
//! i bez karty graficznej. Rzut na ekran i mapa kolorów wymieszane z rysowaniem
//! byłyby sprawdzalne wyłącznie okiem.

pub mod camera;
pub mod panels;
pub mod render;
pub mod replay;
pub mod simulation;

use std::time::Instant;

use eframe::egui::{self, Sense, TextureHandle, TextureOptions};

use crate::camera::Camera;
use crate::panels::{Action, Setup};
use bone_core::session::Session;
use crate::replay::Replay;
use crate::simulation::{Mode, View};

/// Ile klatek odczekać na najniższym ustawieniu szybkości.
///
/// Suwak w pozycji zero nie znaczy „stop", a „jeden krok na osiem klatek" — przy
/// zapadaniu się chmury pojedynczy krok potrafi zmienić obraz na tyle, że pełne
/// tempo jest nie do obejrzenia.
const SLOW_HOLD_FRAMES: u32 = 8;

pub struct App {
    setup: Setup,
    view: Option<View>,
    running: bool,
    camera: Camera,
    drag_origin: Option<egui::Pos2>,
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
            setup: Setup::default(),
            view: None,
            running: false,
            camera: Camera::default(),
            drag_origin: None,
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

        let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match self.setup.mode {
                Mode::Relativistic => {
                    Session::start_sr(self.setup.sr.clone(), out, record)
                }
                Mode::Cosmological => Session::start_lcdm(self.setup.lcdm, out, record),
            }
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
                self.error =
                    "Nie udało się złożyć warunków początkowych — za duża siatka albo \
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
        let session = self
            .view
            .as_mut()
            .and_then(View::as_live_mut)
            .expect("żywy bieg");
        session.apply_runtime_sr(&live_sr);
        session.apply_runtime_lcdm(live_dlna);

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
        if response.dragged() {
            if let (Some(previous), Some(current)) =
                (self.drag_origin, response.interact_pointer_pos())
            {
                let delta = current - previous;
                self.camera.orbit(delta.x, delta.y);
            }
            self.drag_origin = response.interact_pointer_pos();
        } else {
            self.drag_origin = None;
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
                *slot = Some(ui.ctx().load_texture("cloud", image, TextureOptions::LINEAR));
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
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.tick();
        if self.running {
            ctx.request_repaint();
        }

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

        egui::CentralPanel::default().show(ctx, |ui| self.draw_cloud(ui));
    }
}

/// # Errors
/// Gdy nie udaje się otworzyć okna — najczęściej z braku sterownika GPU.
pub fn run() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([960.0, 640.0])
            .with_title("Bone — grawitacja N ciał"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("Bone", options, Box::new(|_cc| Ok(Box::new(App::default()))))
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
}
