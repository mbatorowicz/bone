//! Laboratorium raytracera: obraz dysku Schwarzschilda w oknie.
//!
//! Liczby biorą się z [`bone_core::gr::raytrace`]. Ten plik składa klatkę
//! 320×180 w `ColorImage`, nakłada pierścienie `2M` / `3M` / `6M` i woła
//! silnik w wątku tła — suwak ma pokazać „liczę…”, a nie zamrozić egui.
//! Spin metryki zostaje zerem: to nie Kerr.

use std::f64::consts::TAU;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;

use bone_core::gr::raytrace::{
    self, Camera, Config, CAMERA_R_OVER_M, FOV_Y, HEIGHT_DEFAULT, INCLINATION, WIDTH_DEFAULT,
};
use bone_core::gr::{horizon_radius, isco_radius, photon_sphere_radius};
use eframe::egui::{
    self, Color32, ColorImage, FontId, Pos2, Rect, RichText, Shape, Stroke, TextureHandle,
    TextureOptions, Ui, Vec2,
};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::screen::LabId;

/// Masa nie schodzi do zera: kamera kursu i dysk wymagają M > 0.
pub const MASS_MIN: f64 = 0.40;
/// Kamera bliżej niż to wpada pod horyzont przy `MASS_MAX`.
pub const DIST_MIN: f64 = 12.0;
pub const DIST_MAX: f64 = 80.0;
/// Domyślnie `30` — to samo `r` co [`CAMERA_R_OVER_M`] przy M = 1.
pub const DIST_DEFAULT: f64 = CAMERA_R_OVER_M;
/// θ od osi z. Zero (twarz) jest biegunem: tetrada kamery by się wywróciła.
pub const INCLINE_MIN: f64 = 0.25;
pub const INCLINE_MAX: f64 = 1.45;
/// Start jak [`INCLINATION`]: ~75°, placek a nie oczko.
pub const INCLINE_DEFAULT: f64 = INCLINATION;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const RING_H: Color32 = Color32::from_rgb(90, 90, 100);
const RING_PH: Color32 = Color32::from_rgb(220, 180, 90);
const RING_ISCO: Color32 = Color32::from_rgb(120, 180, 220);
const BUSY: Color32 = Color32::from_rgb(240, 210, 120);

const RING_POINTS: usize = 64;

/// Suwaki, które składają kadr. `r` kamery jest absolutne, więc M puchnie
/// horyzont na obrazie zamiast tylko zmieniać etykietę.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shot {
    pub mass: f64,
    pub incline: f64,
    pub distance: f64,
}

impl Shot {
    pub fn clamped(mass: f64, incline: f64, distance: f64) -> Self {
        let mass = clamp_mass(mass);
        Self {
            mass,
            incline: clamp_incline(incline),
            distance: clamp_distance(distance, mass),
        }
    }
}

impl Default for Shot {
    fn default() -> Self {
        Self::clamped(MASS_DEFAULT, INCLINE_DEFAULT, DIST_DEFAULT)
    }
}

struct Job {
    shot: Shot,
    rx: Receiver<Result<ColorImage, String>>,
}

/// Stan stołu raytracera. Liczenie jest w `Job`, obraz w `pixels`.
pub struct Lab {
    pub mass: f64,
    pub incline: f64,
    pub distance: f64,
    width: u32,
    height: u32,
    job: Option<Job>,
    displayed: Option<Shot>,
    pixels: Option<ColorImage>,
    texture: Option<TextureHandle>,
    tex_shot: Option<Shot>,
    error: String,
}

impl Default for Lab {
    fn default() -> Self {
        let shot = Shot::default();
        Self {
            mass: shot.mass,
            incline: shot.incline,
            distance: shot.distance,
            width: WIDTH_DEFAULT,
            height: HEIGHT_DEFAULT,
            job: None,
            displayed: None,
            pixels: None,
            texture: None,
            tex_shot: None,
            error: String::new(),
        }
    }
}

impl Lab {
    pub fn shot(&self) -> Shot {
        Shot::clamped(self.mass, self.incline, self.distance)
    }

    /// C3/C4 niosą M i i; odległość zostaje, chyba że weszłaby pod horyzont.
    pub fn sync_from_lesson(&mut self, mass: f64, incline: f32) {
        self.mass = clamp_mass(mass);
        self.incline = clamp_incline(f64::from(incline));
        self.distance = clamp_distance(self.distance, self.mass);
    }

    pub fn is_busy(&self) -> bool {
        self.job.is_some() || self.displayed != Some(self.shot())
    }

    pub fn status(&self) -> &str {
        if self.is_busy() {
            "liczę…"
        } else if !self.error.is_empty() {
            self.error.as_str()
        } else {
            "gotowe"
        }
    }

    /// Odbierz klatkę albo rzuć liczenie, jeśli suwak rozjechał się z obrazem.
    pub fn poll(&mut self) {
        if let Some(job) = self.job.as_mut() {
            match job.rx.try_recv() {
                Ok(Ok(image)) => {
                    let shot = job.shot;
                    self.job = None;
                    self.displayed = Some(shot);
                    self.pixels = Some(image);
                    self.error.clear();
                }
                Ok(Err(message)) => {
                    self.job = None;
                    self.error = message;
                }
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    self.job = None;
                    self.error = "wątek raytracera padł".to_string();
                }
            }
        }
        if self.job.is_some() {
            return;
        }
        let want = self.shot();
        if self.displayed == Some(want) {
            return;
        }
        match config_for(want, self.width, self.height) {
            Ok(cfg) => {
                self.error.clear();
                self.job = Some(Job {
                    shot: want,
                    rx: spawn_render(cfg),
                });
            }
            Err(e) => self.error = e.to_string(),
        }
    }
}

pub fn clamp_mass(mass: f64) -> f64 {
    if !mass.is_finite() {
        MASS_DEFAULT
    } else {
        mass.clamp(MASS_MIN, MASS_MAX)
    }
}

pub fn clamp_incline(v: f64) -> f64 {
    if !v.is_finite() {
        INCLINE_DEFAULT
    } else {
        v.clamp(INCLINE_MIN, INCLINE_MAX)
    }
}

pub fn clamp_distance(distance: f64, mass: f64) -> f64 {
    let mass = clamp_mass(mass);
    let floor = DIST_MIN.max(2.5 * mass);
    let raw = if distance.is_finite() {
        distance
    } else {
        DIST_DEFAULT
    };
    raw.clamp(floor, DIST_MAX)
}

/// Kadr z suwaków. Kamera kursu to M = 1, θ ≈ 75°, r = 30.
pub fn config_for(shot: Shot, width: u32, height: u32) -> Result<Config, raytrace::RaytraceError> {
    let shot = Shot::clamped(shot.mass, shot.incline, shot.distance);
    let mut cfg = Config::with_size(shot.mass, width, height)?;
    cfg.camera = Camera::new(shot.distance, shot.incline, 0.0, FOV_Y)?;
    let horizon = cfg.metric.horizon_radius();
    if cfg.camera.r() <= horizon {
        return Err(raytrace::RaytraceError::CameraInside {
            r: cfg.camera.r(),
            horizon,
        });
    }
    Ok(cfg)
}

/// Płaski rzut punktu równika `(ρ, φ)` na kadr. Overlay, nie geodezyjna.
pub fn project_ring_point(
    shot: Shot,
    rho: f64,
    phi: f64,
    width: f64,
    height: f64,
) -> Option<[f32; 2]> {
    let r = shot.distance;
    let th = shot.incline;
    let cam_right = rho * phi.sin();
    let cam_up = -rho * phi.cos() * th.cos();
    let cam_fwd = r - rho * phi.cos() * th.sin();
    if !cam_fwd.is_finite() || cam_fwd <= 1e-6 {
        return None;
    }
    let sx = cam_right / cam_fwd;
    let sy = cam_up / cam_fwd;
    let tany = (0.5 * FOV_Y).tan();
    if tany <= 0.0 || width <= 0.0 || height <= 0.0 {
        return None;
    }
    let tanx = tany * (width / height);
    let ndc_x = sx / tanx;
    let ndc_y = sy / tany;
    Some([(0.5 * (ndc_x + 1.0)) as f32, (0.5 * (1.0 - ndc_y)) as f32])
}

/// Rozpiętość pionowa pierścienia w ułamku kadru. Nachylenie spłaszcza.
pub fn ring_span_y(shot: Shot, rho: f64, width: f64, height: f64) -> Option<f32> {
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    let mut n = 0;
    for i in 0..RING_POINTS {
        let phi = TAU * (i as f64) / (RING_POINTS as f64);
        if let Some(p) = project_ring_point(shot, rho, phi, width, height) {
            y_min = y_min.min(p[1]);
            y_max = y_max.max(p[1]);
            n += 1;
        }
    }
    (n >= 8).then_some(y_max - y_min)
}

fn spawn_render(cfg: Config) -> Receiver<Result<ColorImage, String>> {
    let (tx, rx) = mpsc::channel();
    let _ = thread::Builder::new()
        .name("bh-raytrace".into())
        .spawn(move || {
            let out = raytrace::render(cfg)
                .map(|buf| color_image(&buf))
                .map_err(|e| e.to_string());
            let _ = tx.send(out);
        });
    rx
}

fn color_image(buf: &raytrace::Buffer) -> ColorImage {
    ColorImage::from_rgba_unmultiplied([buf.width as usize, buf.height as usize], &buf.rgba)
}

/// Drzwi z lekcji C3 i C4: animacja 2D zostaje, przycisk otwiera klatkę.
pub fn draw_ray_door(ui: &mut Ui) -> bool {
    let mut open = false;
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui
            .add_sized(
                [280.0, 32.0],
                egui::Button::new("Otwórz laboratorium raytracera"),
            )
            .clicked()
        {
            open = true;
        }
        ui.label(
            RichText::new(format!(
                "wejście → {} · 320×180 w tle · spin = 0",
                LabId::BlackHole.label()
            ))
            .small()
            .weak()
            .monospace(),
        );
    });
    ui.label(
        RichText::new("Suwaki M / i / r zostają. Okno ma liczyć piksele, nie zamarzać.")
            .small()
            .weak(),
    );
    ui.add_space(6.0);
    open
}

/// Cały stół: suwaki z lewej, klatka w środku.
pub fn draw_lab(ctx: &egui::Context, lab: &mut Lab) {
    egui::SidePanel::left("bh-ray-panel")
        .resizable(false)
        .min_width(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| controls(ui, lab));
        });
    egui::CentralPanel::default().show(ctx, |ui| paint_frame(ui, lab));
}

fn controls(ui: &mut Ui, lab: &mut Lab) {
    ui.add_space(8.0);
    ui.label(RichText::new("Raytracer Schwarzschilda").strong());
    ui.label(
        RichText::new("G = c = 1 · spin = 0. Piksel = geodezyjna zerowa wstecz.")
            .small()
            .weak(),
    );
    ui.add_space(10.0);
    ui.add(
        egui::Slider::new(&mut lab.mass, MASS_MIN..=MASS_MAX)
            .text("M")
            .fixed_decimals(2),
    );
    lab.mass = clamp_mass(lab.mass);
    ui.add(
        egui::Slider::new(&mut lab.incline, INCLINE_MIN..=INCLINE_MAX)
            .text("i")
            .fixed_decimals(2),
    );
    lab.incline = clamp_incline(lab.incline);
    ui.add(
        egui::Slider::new(&mut lab.distance, DIST_MIN..=DIST_MAX)
            .text("r")
            .fixed_decimals(1),
    );
    lab.distance = clamp_distance(lab.distance, lab.mass);
    ui.add_space(8.0);
    let shot = lab.shot();
    ui.label(
        RichText::new(format!(
            "2M={:.2}  3M={:.2}  6M={:.2}",
            horizon_radius(shot.mass),
            photon_sphere_radius(shot.mass),
            isco_radius(shot.mass)
        ))
        .monospace(),
    );
    ui.label(
        RichText::new(format!(
            "θ = {:.0}°  ·  r / M = {:.1}  ·  spin = 0",
            shot.incline.to_degrees(),
            shot.distance / shot.mass
        ))
        .small()
        .weak()
        .monospace(),
    );
    ui.add_space(8.0);
    let busy = lab.is_busy();
    ui.label(RichText::new(lab.status()).strong().color(if busy {
        BUSY
    } else {
        Color32::from_rgb(160, 200, 140)
    }));
    ui.label(
        RichText::new(format!("{}×{} · Kerr nie wchodzi", lab.width, lab.height))
            .small()
            .weak()
            .monospace(),
    );
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        ring_tile(ui, "2M", "horyzont", RING_H);
        ring_tile(ui, "3M", "foton", RING_PH);
        ring_tile(ui, "6M", "ISCO", RING_ISCO);
    });
    ui.add_space(10.0);
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(22, 28, 36))
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.label(
                RichText::new("suwak rusza kadr. obraz przychodzi, gdy wątek skończy")
                    .italics()
                    .color(Color32::from_rgb(200, 230, 170)),
            );
        });
}

fn ring_tile(ui: &mut Ui, title: &str, hint: &str, color: Color32) {
    ui.group(|ui| {
        ui.set_width(80.0);
        ui.label(RichText::new(title).strong().color(color));
        ui.label(RichText::new(hint).small().weak());
    });
}

fn paint_frame(ui: &mut Ui, lab: &mut Lab) {
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, BG);
    let inner = fit_frame(rect, lab.width as f32, lab.height as f32);
    if lab.pixels.is_some() && lab.tex_shot != lab.displayed {
        let image = lab.pixels.clone().expect("pixels");
        match &mut lab.texture {
            Some(existing) if existing.size() == image.size => {
                existing.set(image, TextureOptions::LINEAR);
            }
            slot => {
                *slot = Some(
                    ui.ctx()
                        .load_texture("bh-ray", image, TextureOptions::LINEAR),
                );
            }
        }
        lab.tex_shot = lab.displayed;
    }
    if let Some(texture) = &lab.texture {
        egui::Image::new(texture)
            .maintain_aspect_ratio(false)
            .fit_to_exact_size(inner.size())
            .paint_at(ui, inner);
    }
    let shot = lab.displayed.unwrap_or_else(|| lab.shot());
    stroke_overlay(ui, inner, shot, lab.width as f64, lab.height as f64);
    if lab.is_busy() {
        painter.rect_filled(
            Rect::from_min_size(inner.left_top(), Vec2::new(inner.width(), 28.0)),
            0.0,
            Color32::from_rgba_unmultiplied(8, 10, 16, 180),
        );
        painter.text(
            inner.left_top() + Vec2::new(12.0, 6.0),
            egui::Align2::LEFT_TOP,
            "liczę…",
            FontId::proportional(16.0),
            BUSY,
        );
    }
}

fn fit_frame(avail: Rect, w: f32, h: f32) -> Rect {
    if w <= 0.0 || h <= 0.0 {
        return avail;
    }
    let aspect = w / h;
    let mut rw = avail.width();
    let mut rh = rw / aspect;
    if rh > avail.height() {
        rh = avail.height();
        rw = rh * aspect;
    }
    Rect::from_center_size(avail.center(), Vec2::new(rw, rh))
}

fn stroke_overlay(ui: &Ui, rect: Rect, shot: Shot, width: f64, height: f64) {
    let painter = ui.painter_at(rect);
    let rings = [
        (horizon_radius(shot.mass), RING_H, "2M"),
        (photon_sphere_radius(shot.mass), RING_PH, "3M"),
        (isco_radius(shot.mass), RING_ISCO, "6M"),
    ];
    for (rho, color, name) in rings {
        let mut pts = Vec::with_capacity(RING_POINTS);
        for i in 0..=RING_POINTS {
            let phi = TAU * (i as f64) / (RING_POINTS as f64);
            if let Some(p) = project_ring_point(shot, rho, phi, width, height) {
                pts.push(to_screen(rect, p));
            }
        }
        if pts.len() >= 2 {
            painter.add(Shape::line(pts, Stroke::new(1.4, color)));
        }
        if let Some(p) = project_ring_point(shot, rho, 0.35, width, height) {
            painter.text(
                to_screen(rect, p) + Vec2::new(6.0, 0.0),
                egui::Align2::LEFT_CENTER,
                name,
                FontId::proportional(12.0),
                color,
            );
        }
    }
    painter.text(
        rect.left_bottom() + Vec2::new(8.0, -8.0),
        egui::Align2::LEFT_BOTTOM,
        "overlay 2M / 3M / 6M · spin = 0",
        FontId::proportional(11.0),
        LABEL,
    );
}

fn to_screen(rect: Rect, frac: [f32; 2]) -> Pos2 {
    Pos2::new(
        rect.left() + frac[0] * rect.width(),
        rect.top() + frac[1] * rect.height(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::gr::raytrace::{HEIGHT_TINY, WIDTH_TINY};
    use std::time::{Duration, Instant};

    fn tiny_lab() -> Lab {
        Lab {
            width: WIDTH_TINY,
            height: HEIGHT_TINY,
            ..Lab::default()
        }
    }

    #[test]
    fn sliders_stay_in_range() {
        assert_eq!(clamp_mass(MASS_DEFAULT), MASS_DEFAULT);
        assert_eq!(clamp_mass(0.0), MASS_MIN);
        assert_eq!(clamp_mass(40.0), MASS_MAX);
        assert_eq!(clamp_mass(f64::NAN), MASS_DEFAULT);
        assert_eq!(clamp_incline(INCLINE_DEFAULT), INCLINE_DEFAULT);
        assert_eq!(clamp_incline(0.0), INCLINE_MIN);
        assert_eq!(clamp_incline(9.0), INCLINE_MAX);
        assert_eq!(clamp_distance(DIST_DEFAULT, 1.0), DIST_DEFAULT);
        assert_eq!(clamp_distance(1.0, 1.0), DIST_MIN);
        assert!(clamp_distance(12.0, MASS_MAX) >= 2.5 * MASS_MAX);
    }

    #[test]
    fn default_shot_matches_course_camera() {
        let cfg = config_for(Shot::default(), WIDTH_DEFAULT, HEIGHT_DEFAULT).unwrap();
        let course = Config::course(MASS_DEFAULT).unwrap();
        assert_eq!(cfg.width, WIDTH_DEFAULT);
        assert_eq!(cfg.height, HEIGHT_DEFAULT);
        assert!((cfg.camera.r() - course.camera.r()).abs() < 1e-12);
        assert!((cfg.camera.theta() - course.camera.theta()).abs() < 1e-12);
        assert_eq!(cfg.camera.phi(), 0.0);
    }

    #[test]
    fn overlay_flattens_when_inclination_grows() {
        let w = WIDTH_DEFAULT as f64;
        let h = HEIGHT_DEFAULT as f64;
        let face = Shot::clamped(1.0, INCLINE_MIN, DIST_DEFAULT);
        let edge = Shot::clamped(1.0, INCLINE_MAX, DIST_DEFAULT);
        let face_y = ring_span_y(face, 6.0, w, h).expect("twarz");
        let edge_y = ring_span_y(edge, 6.0, w, h).expect("krawędź");
        assert!(
            face_y > edge_y * 1.5,
            "twarz {face_y} miała być wyższa niż krawędź {edge_y}"
        );
    }

    #[test]
    fn mass_grows_the_coordinate_rings() {
        let w = WIDTH_DEFAULT as f64;
        let h = HEIGHT_DEFAULT as f64;
        let small = Shot::clamped(MASS_MIN, INCLINE_DEFAULT, DIST_DEFAULT);
        let big = Shot::clamped(MASS_MAX, INCLINE_DEFAULT, DIST_DEFAULT);
        let a = ring_span_y(small, horizon_radius(small.mass), w, h).unwrap();
        let b = ring_span_y(big, horizon_radius(big.mass), w, h).unwrap();
        assert!(b > a, "większe M miało puchnąć 2M: {a} vs {b}");
    }

    #[test]
    fn tiny_buffer_becomes_a_color_image() {
        let cfg = config_for(Shot::default(), WIDTH_TINY, HEIGHT_TINY).unwrap();
        let buf = raytrace::render(cfg).unwrap();
        let image = color_image(&buf);
        assert_eq!(image.size, [WIDTH_TINY as usize, HEIGHT_TINY as usize]);
        assert_eq!(image.pixels.len(), (WIDTH_TINY * HEIGHT_TINY) as usize);
        let cx = 9 * WIDTH_TINY as usize + 16;
        assert_eq!(image.pixels[cx], Color32::BLACK);
    }

    #[test]
    fn poll_fills_a_tiny_image_from_a_background_thread() {
        let mut lab = tiny_lab();
        lab.poll();
        assert!(lab.is_busy());
        assert_eq!(lab.status(), "liczę…");
        let start = Instant::now();
        while lab.is_busy() && start.elapsed() < Duration::from_secs(30) {
            thread::sleep(Duration::from_millis(20));
            lab.poll();
        }
        assert!(!lab.is_busy(), "status = {}", lab.status());
        assert!(lab.pixels.is_some());
        assert_eq!(lab.displayed, Some(lab.shot()));
        assert_eq!(lab.status(), "gotowe");
    }

    #[test]
    fn lesson_sync_clamps_and_keeps_spin_zero_in_the_engine() {
        let mut lab = Lab::default();
        lab.sync_from_lesson(0.0, 0.0);
        assert_eq!(lab.mass, MASS_MIN);
        assert_eq!(lab.incline, INCLINE_MIN);
        let cfg = config_for(lab.shot(), 8, 8).unwrap();
        assert_eq!(cfg.camera.phi(), 0.0);
        assert!((cfg.metric.mass() - MASS_MIN).abs() < 1e-15);
    }

    #[test]
    fn ray_door_and_lab_paint_without_a_window() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            assert!(!draw_ray_door(ui));
        });
        let _ = ctx.end_pass();
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        let mut lab = tiny_lab();
        draw_lab(&ctx, &mut lab);
        let _ = ctx.end_pass();
        assert!(lab.job.is_none(), "paint nie woła silnika");
    }
}
