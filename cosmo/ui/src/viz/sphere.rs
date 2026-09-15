//! Kula: wielkie koło („idź prosto”) kontra równoleżnik z mapy.
//!
//! Liczba z lekcji to `s = R α`. Tu R = 1, α to kąt przy środku między A i B.
//! Równoleżnik ma ten sam Δλ, ale krótszy promień `cos φ` — dlatego jest
//! dłuższy i nie jest geodezyjną. Suwak tylko obraca kamerę.

use std::f32::consts::TAU;
use std::f64::consts::PI;

use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use super::SPIN_DEFAULT;
use crate::lesson::Playback;

/// Start mrówki, szerokość jak umiarkowane szerokości geograficzne.
pub const START: (f64, f64) = (0.70, -0.35);
/// Meta na tej samej szerokości, Δλ ≈ 160° — geodezyjna ucieka ku biegunowi.
pub const END: (f64, f64) = (0.70, 2.45);
const R: f64 = 1.0;
const TILT: f32 = 0.42;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(40, 50, 64);
const GRID_BACK: Color32 = Color32::from_rgb(22, 28, 36);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const GEO: Color32 = Color32::from_rgb(120, 180, 220);
const MAP: Color32 = Color32::from_rgb(160, 200, 140);
const POINT: Color32 = Color32::from_rgb(240, 220, 140);

/// Jednostkowy kierunek na sferze. `y` to oś biegunów.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dir {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Dir {
    pub fn from_lat_lon(lat: f64, lon: f64) -> Self {
        Self {
            x: lat.cos() * lon.cos(),
            y: lat.sin(),
            z: lat.cos() * lon.sin(),
        }
        .normalized()
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn normalized(self) -> Self {
        let n = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n < 1e-18 {
            Self {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
            }
        }
    }

    /// Kąt przy środku, w `[0, π]`.
    pub fn angle(self, other: Self) -> f64 {
        self.dot(other).clamp(-1.0, 1.0).acos()
    }

    pub fn slerp(self, other: Self, t: f64) -> Self {
        let omega = self.angle(other);
        if omega < 1e-12 {
            return self;
        }
        let so = omega.sin();
        let a = ((1.0 - t) * omega).sin() / so;
        let b = (t * omega).sin() / so;
        Self {
            x: a * self.x + b * other.x,
            y: a * self.y + b * other.y,
            z: a * self.z + b * other.z,
        }
        .normalized()
    }
}

pub fn start() -> Dir {
    Dir::from_lat_lon(START.0, START.1)
}

pub fn end() -> Dir {
    Dir::from_lat_lon(END.0, END.1)
}

/// `s = R α` na wielkim kole.
pub fn great_circle_arc(radius: f64, a: Dir, b: Dir) -> f64 {
    radius * a.angle(b)
}

pub fn dlon() -> f64 {
    END.1 - START.1
}

/// Długość równoleżnika: `R cos φ · |Δλ|`.
pub fn parallel_arc(radius: f64, lat: f64, delta_lon: f64) -> f64 {
    radius * lat.cos() * delta_lon.abs()
}

pub fn parallel_point(lat: f64, lon0: f64, lon1: f64, t: f64) -> Dir {
    Dir::from_lat_lon(lat, lon0 + t * (lon1 - lon0))
}

pub fn clamp_spin(spin: f32) -> f32 {
    if !spin.is_finite() {
        SPIN_DEFAULT
    } else {
        spin.rem_euclid(TAU)
    }
}

pub fn draw(ui: &mut Ui, playback: &mut Playback) {
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut playback.spin, 0.0..=TAU)
                .text("obrót")
                .fixed_decimals(2),
        );
        playback.spin = clamp_spin(playback.spin);
        ui.label(
            RichText::new(format!(
                "s = Rα = {:.3}  ·  mapa = {:.3}",
                great_circle_arc(R, start(), end()),
                parallel_arc(R, START.0, dlon())
            ))
            .monospace(),
        );
    });
    ui.label(
        RichText::new("niebieski: nie skręcaj  ·  zielony: „prosto” po mapie (równoleżnik)")
            .small()
            .weak(),
    );
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    paint(ui, rect, playback.spin, playback.t);
}

fn paint(ui: &Ui, rect: Rect, spin: f32, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let pad = 18.0;
    let inner = rect.shrink(pad);
    if inner.width() < 16.0 || inner.height() < 16.0 {
        return;
    }
    let cam = Cam {
        yaw: clamp_spin(spin),
        tilt: TILT,
        c: inner.center(),
        scale: inner.width().min(inner.height()) * 0.42,
    };
    let a = start();
    let b = end();
    let t = f64::from(phase.rem_euclid(TAU) / TAU);

    for i in 0..12 {
        let lon = f64::from(i) * (2.0 * PI / 12.0);
        let mut prev = Dir::from_lat_lon(-0.5 * PI + 0.04, lon);
        for k in 1..=24 {
            let lat = -0.5 * PI + 0.04 + f64::from(k) * ((PI - 0.08) / 24.0);
            let p = Dir::from_lat_lon(lat, lon);
            stroke_segment(&painter, &cam, prev, p);
            prev = p;
        }
    }
    for j in 1..8 {
        let lat = -0.5 * PI + f64::from(j) * (PI / 8.0);
        let mut prev = Dir::from_lat_lon(lat, 0.0);
        for k in 1..=32 {
            let lon = f64::from(k) * (2.0 * PI / 32.0);
            let p = Dir::from_lat_lon(lat, lon);
            stroke_segment(&painter, &cam, prev, p);
            prev = p;
        }
    }

    let mut prev_g = a;
    let mut prev_m = a;
    for k in 1..=48 {
        let u = f64::from(k) / 48.0;
        let g = a.slerp(b, u);
        let m = parallel_point(START.0, START.1, END.1, u);
        stroke_path(&painter, &cam, prev_g, g, GEO, 2.2);
        stroke_path(&painter, &cam, prev_m, m, MAP, 2.2);
        prev_g = g;
        prev_m = m;
    }

    dot(&painter, &cam, a, POINT, "A");
    dot(&painter, &cam, b, POINT, "B");
    dot(&painter, &cam, a.slerp(b, t), GEO, "");
    dot(
        &painter,
        &cam,
        parallel_point(START.0, START.1, END.1, t),
        MAP,
        "",
    );

    painter.text(
        Pos2::new(inner.left() + 6.0, inner.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        "wielkie koło jest krótsze — równoleżnik skręca, choć etykieta φ stoi",
        FontId::proportional(11.0),
        LABEL,
    );
}

struct Cam {
    yaw: f32,
    tilt: f32,
    c: Pos2,
    scale: f32,
}

fn project(cam: &Cam, p: Dir) -> (Pos2, f32) {
    let x = p.x as f32;
    let y = p.y as f32;
    let z = p.z as f32;
    let cy = cam.yaw.cos();
    let sy = cam.yaw.sin();
    let x1 = x * cy - z * sy;
    let z1 = x * sy + z * cy;
    let ct = cam.tilt.cos();
    let st = cam.tilt.sin();
    let y2 = y * ct - z1 * st;
    let z2 = y * st + z1 * ct;
    (
        Pos2::new(cam.c.x + cam.scale * x1, cam.c.y - cam.scale * y2),
        z2,
    )
}

fn stroke_segment(painter: &egui::Painter, cam: &Cam, a: Dir, b: Dir) {
    let (pa, da) = project(cam, a);
    let (pb, db) = project(cam, b);
    let back = da + db < 0.0;
    let color = if back { GRID_BACK } else { GRID };
    let w = if back { 0.7 } else { 1.0 };
    painter.line_segment([pa, pb], Stroke::new(w, color));
}

fn stroke_path(painter: &egui::Painter, cam: &Cam, a: Dir, b: Dir, color: Color32, width: f32) {
    let (pa, da) = project(cam, a);
    let (pb, db) = project(cam, b);
    let w = if da + db < 0.0 { width * 0.55 } else { width };
    painter.line_segment([pa, pb], Stroke::new(w, color));
}

fn dot(painter: &egui::Painter, cam: &Cam, p: Dir, color: Color32, label: &str) {
    let (pos, depth) = project(cam, p);
    let r = if depth < 0.0 { 3.2 } else { 5.2 };
    painter.circle_filled(pos, r, color);
    if !label.is_empty() {
        painter.text(
            pos + Vec2::new(7.0, -6.0),
            egui::Align2::LEFT_BOTTOM,
            label,
            FontId::proportional(12.0),
            color,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn great_circle_is_radius_times_angle() {
        let a = start();
        let b = end();
        assert!((a.dot(a) - 1.0).abs() < 1e-12);
        assert!((b.dot(b) - 1.0).abs() < 1e-12);
        let alpha = a.angle(b);
        assert!(alpha > 1.0 && alpha < PI);
        assert!((great_circle_arc(R, a, b) - alpha).abs() < 1e-15);
        assert!((great_circle_arc(2.0, a, b) - 2.0 * alpha).abs() < 1e-15);
    }

    #[test]
    fn map_straight_is_longer_than_the_geodesic() {
        let a = start();
        let b = end();
        let geo = great_circle_arc(R, a, b);
        let par = parallel_arc(R, START.0, dlon());
        assert!(
            par > geo + 0.15,
            "równoleżnik {par} miał być wyraźnie dłuższy niż {geo}"
        );
        assert!((START.0 - END.0).abs() < 1e-15);
    }

    #[test]
    fn slerp_hits_the_endpoints_and_stays_on_the_sphere() {
        let a = start();
        let b = end();
        let p0 = a.slerp(b, 0.0);
        let p1 = a.slerp(b, 1.0);
        assert!((p0.x - a.x).abs() < 1e-12 && (p0.y - a.y).abs() < 1e-12);
        assert!((p1.x - b.x).abs() < 1e-12 && (p1.y - b.y).abs() < 1e-12);
        let mid = a.slerp(b, 0.5);
        assert!((mid.dot(mid) - 1.0).abs() < 1e-12);
        assert!(mid.y > a.y, "geodezyjna miała iść ku biegunowi");
        let par = parallel_point(START.0, START.1, END.1, 0.5);
        assert!((par.y - START.0.sin()).abs() < 1e-12);
        assert!(mid.y > par.y);
    }

    #[test]
    fn spin_wraps_and_rejects_nan() {
        assert!((clamp_spin(0.0) - 0.0).abs() < 1e-6);
        assert!(clamp_spin(TAU) < 1e-5);
        assert!(clamp_spin(f32::NAN).is_finite());
        assert!((clamp_spin(SPIN_DEFAULT) - SPIN_DEFAULT).abs() < 1e-6);
    }

    #[test]
    fn geo_lesson_1_paints_without_a_window() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            let mut playback = Playback {
                playing: false,
                t: 1.2,
                ..Default::default()
            };
            draw(ui, &mut playback);
            assert!((playback.spin - SPIN_DEFAULT).abs() < 1e-6);
        });
        let _ = ctx.end_pass();
    }
}
