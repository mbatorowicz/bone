//! Pierścienie `2M` / `3M` / `6M`, pęk promieni i pierścień Einsteina w 2D.
//!
//! Liczby biorą się z [`bone_core::gr::metric`] i [`bone_core::gr::geodesic`].
//! Ten plik nie woła [`bone_core::gr::raytrace::Buffer`]: klatka 320×180
//! zostaje laboratorium z kroku 14. Tu jest przekrój, suwak i ruch.

use std::f32::consts::TAU;
use std::f64::consts::FRAC_PI_2;
use std::sync::Mutex;

use bone_core::gr::geodesic::{self, GeodesicState, STATE_LEN};
use bone_core::gr::raytrace::{CAMERA_R_OVER_M, DISK_OUTER_OVER_M};
use bone_core::gr::rk4::Scratch;
use bone_core::gr::{horizon_radius, isco_radius, photon_sphere_radius, Schwarzschild};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Shape, Stroke, Ui};

use super::{MASS_DEFAULT, MASS_MAX, SPIN_DEFAULT};
use crate::lesson::Playback;

/// Start pęku: dość daleko, żeby parametr zderzenia zdążył być „z łąki”.
pub const R_START_OVER_M: f64 = 22.0;
pub const DROP_H: f64 = 0.1;
pub const DROP_STEPS: usize = 1200;
/// Nachylenie: 0 = twarzą, max ≈ 80°. `playback.spin` trzyma ten kąt.
pub const INCLINE_MAX: f32 = 1.40;
/// Oko i źródło jak kamera kursu: `D_l = D_ls = 30M`.
pub const EINSTEIN_D_OVER_M: f64 = CAMERA_R_OVER_M;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const HORIZON_FILL: Color32 = Color32::from_rgb(12, 14, 18);
const RING_H: Color32 = Color32::from_rgb(90, 90, 100);
const RING_PH: Color32 = Color32::from_rgb(220, 180, 90);
const RING_ISCO: Color32 = Color32::from_rgb(120, 180, 220);
const EINSTEIN: Color32 = Color32::from_rgb(240, 210, 120);
const CAP: Color32 = Color32::from_rgb(220, 120, 100);
const DISK: Color32 = Color32::from_rgb(210, 140, 80);
const ESC: Color32 = Color32::from_rgb(120, 180, 220);
const NOW: Color32 = Color32::from_rgb(230, 230, 240);
const CAM: Color32 = Color32::from_rgb(160, 200, 140);
const STAR: Color32 = Color32::from_rgb(240, 220, 140);

/// Parametry zderzenia pęku, w jednostkach `b_c`. Zero plus pary ±.
const B_OVER_CRIT: [f64; 6] = [0.0, 0.80, 1.00, 1.25, 1.50, 2.20];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayFate {
    Horizon,
    Disk,
    Escape,
}

impl RayFate {
    fn color(self) -> Color32 {
        match self {
            Self::Horizon => CAP,
            Self::Disk => DISK,
            Self::Escape => ESC,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Ray {
    pub b: f64,
    pub points: Vec<[f64; 2]>,
    pub periapsis: f64,
    pub captured: bool,
}

impl Ray {
    pub fn fate(&self, mass: f64) -> RayFate {
        if self.captured {
            return RayFate::Horizon;
        }
        let m = mass.max(0.0);
        let inner = isco_radius(m);
        let outer = disk_outer(m);
        if self.periapsis >= inner && self.periapsis <= outer {
            RayFate::Disk
        } else {
            RayFate::Escape
        }
    }
}

struct Bundle {
    key: i32,
    rays: Vec<Ray>,
}

static BUNDLE: Mutex<Option<Bundle>> = Mutex::new(None);

pub fn clamp_mass(mass: f64) -> f64 {
    if !mass.is_finite() {
        MASS_DEFAULT
    } else {
        mass.clamp(0.0, MASS_MAX)
    }
}

pub fn clamp_incline(v: f32) -> f32 {
    if !v.is_finite() {
        SPIN_DEFAULT.clamp(0.0, INCLINE_MAX)
    } else {
        v.clamp(0.0, INCLINE_MAX)
    }
}

/// `cos i`: twarz = 1, krawędź → 0.
pub fn disk_aspect(incline: f32) -> f32 {
    clamp_incline(incline).cos()
}

pub fn photon_b_crit(mass: f64) -> f64 {
    3.0 * 3.0_f64.sqrt() * mass.max(0.0)
}

/// Słabe minięcie: `α ≈ 4M/b`.
pub fn weak_deflection(mass: f64, b: f64) -> Option<f64> {
    if !mass.is_finite() || !b.is_finite() || mass < 0.0 || b.abs() < 1e-12 {
        return None;
    }
    Some(4.0 * mass / b.abs())
}

/// `b_E = √(4M D_l D_ls / D_s)`, `D_s = D_l + D_ls`.
pub fn einstein_b(mass: f64, d_l: f64, d_ls: f64) -> f64 {
    if mass <= 0.0 || d_l <= 0.0 || d_ls <= 0.0 {
        return 0.0;
    }
    let d_s = d_l + d_ls;
    (4.0 * mass * d_l * d_ls / d_s).sqrt()
}

pub fn einstein_b_course(mass: f64) -> f64 {
    let m = mass.max(0.0);
    let d = EINSTEIN_D_OVER_M * m.max(1e-9);
    if m <= 0.0 {
        0.0
    } else {
        einstein_b(m, d, d)
    }
}

pub fn disk_outer(mass: f64) -> f64 {
    DISK_OUTER_OVER_M * mass.max(0.0)
}

pub fn camera_r(mass: f64) -> f64 {
    CAMERA_R_OVER_M * mass.max(0.0)
}

/// `u^φ` na sferze fotonowej. `None` przy M = 0.
pub fn photon_omega(mass: f64) -> Option<f64> {
    let bh = Schwarzschild::new(mass).ok()?;
    Some(GeodesicState::circular_photon(bh).ok()?.u_phi)
}

/// `u^φ` na ISCO. `None` przy M = 0.
pub fn isco_omega(mass: f64) -> Option<f64> {
    let bh = Schwarzschild::new(mass).ok()?;
    let s = GeodesicState::circular_timelike(bh, isco_radius(mass)).ok()?;
    Some(s.u_phi)
}

fn metric_key(mass: f64) -> i32 {
    (clamp_mass(mass) * 50.0).round() as i32
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "czarne 2M · złote 3M (foton) · niebieskie 6M (ISCO)",
        2 => "pęk zerowych geodezyjnych · złota obwódka = pierścień Einsteina",
        3 => "sitko wstecz: czerwony horyzont · pomarańczowy dysk · niebieski niebo",
        _ => "masa skaluje · nachylenie spłaszcza talerz · spin metryki = 0",
    }
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut playback.mass, 0.0..=MASS_MAX)
                .text("M")
                .fixed_decimals(2),
        );
        playback.mass = clamp_mass(playback.mass);
        if lesson == 4 {
            ui.add(
                egui::Slider::new(&mut playback.spin, 0.0..=INCLINE_MAX)
                    .text("i")
                    .fixed_decimals(2),
            );
            playback.spin = clamp_incline(playback.spin);
        }
        let m = playback.mass;
        ui.label(
            RichText::new(format!(
                "2M={:.2}  3M={:.2}  6M={:.2}  b_E={:.2}",
                horizon_radius(m),
                photon_sphere_radius(m),
                isco_radius(m),
                einstein_b_course(m)
            ))
            .monospace(),
        );
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let aspect = if lesson == 4 {
        disk_aspect(playback.spin)
    } else {
        1.0
    };
    paint(ui, rect, lesson, playback.mass, aspect, playback.t);
}

fn paint(ui: &Ui, rect: Rect, lesson: u8, mass: f64, aspect: f32, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 || inner.height() < 16.0 {
        return;
    }
    let m = clamp_mass(mass);
    let span = view_span(lesson, m);
    let scale = inner.width().min(inner.height()) * 0.46 / span as f32;
    let view = View {
        c: inner.center(),
        scale,
        aspect: aspect.clamp(0.04, 1.0),
    };
    if lesson == 2 || lesson == 3 {
        paint_bundle(&painter, &view, lesson, m, phase);
    }
    if lesson == 3 || lesson == 4 {
        paint_disk(&painter, &view, m);
    }
    paint_rings(&painter, &view, m, phase, lesson);
    if lesson == 2 || lesson == 4 {
        stroke_ellipse(&painter, &view, einstein_b_course(m), EINSTEIN, 1.6);
        label_ring(&painter, &view, einstein_b_course(m), "E", EINSTEIN);
    }
    if lesson == 2 {
        paint_source_and_eye(&painter, &view, m);
    }
    if lesson == 3 {
        paint_camera(&painter, &view, m);
    }
    painter.text(
        Pos2::new(inner.left() + 6.0, inner.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        footer(lesson, m, aspect),
        FontId::proportional(11.0),
        LABEL,
    );
}

fn footer(lesson: u8, mass: f64, aspect: f32) -> String {
    match lesson {
        1 => format!("M = {mass:.2}  ·  trzy zawody, nie trzy nazwy"),
        2 => format!(
            "b_c / M = 3√3 ≈ {:.3}  ·  b_E / M ≈ {:.3}",
            photon_b_crit(1.0),
            einstein_b_course(mass) / mass.max(1e-9)
        ),
        3 => "każdy piksel przyszłego obrazu to jeden taki promień wstecz".into(),
        _ => format!(
            "i = {:.0}°  ·  cos i = {aspect:.2}  ·  spin = 0",
            f64::from(aspect.acos()) * 180.0 / std::f64::consts::PI
        ),
    }
}

fn view_span(lesson: u8, mass: f64) -> f64 {
    let m = mass.max(0.35);
    match lesson {
        1 => (9.0 * m).max(10.0),
        4 => (disk_outer(m) + 4.0 * m).max(14.0),
        _ => (R_START_OVER_M * m + 4.0).max(16.0),
    }
}

struct View {
    c: Pos2,
    scale: f32,
    aspect: f32,
}

impl View {
    fn map(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(
            self.c.x + self.scale * x as f32,
            self.c.y - self.scale * self.aspect * y as f32,
        )
    }
}

fn paint_rings(painter: &egui::Painter, view: &View, mass: f64, phase: f32, lesson: u8) {
    let rh = horizon_radius(mass);
    let rph = photon_sphere_radius(mass);
    let risco = isco_radius(mass);
    if rh > 1e-6 {
        fill_ellipse(painter, view, rh, HORIZON_FILL);
        stroke_ellipse(painter, view, rh, RING_H, 2.0);
        label_ring(painter, view, rh, "2M", RING_H);
    }
    if rph > rh + 1e-6 {
        stroke_ellipse(painter, view, rph, RING_PH, 2.0);
        label_ring(painter, view, rph, "3M", RING_PH);
        if let Some(w) = photon_omega(mass) {
            let phi = w * f64::from(phase);
            let p = view.map(rph * phi.cos(), rph * phi.sin());
            painter.circle_filled(p, 5.0, RING_PH);
        }
    }
    if risco > rph + 1e-6 {
        stroke_ellipse(painter, view, risco, RING_ISCO, 2.0);
        label_ring(painter, view, risco, "6M", RING_ISCO);
        if let Some(w) = isco_omega(mass) {
            let phi = w * f64::from(phase);
            let p = view.map(risco * phi.cos(), risco * phi.sin());
            painter.circle_filled(p, 5.5, RING_ISCO);
        }
    }
    if lesson == 1 && mass <= 1e-9 {
        painter.circle_stroke(view.c, 10.0, Stroke::new(1.0, LABEL));
    }
}

fn paint_disk(painter: &egui::Painter, view: &View, mass: f64) {
    let inner = isco_radius(mass);
    let outer = disk_outer(mass);
    if outer <= inner + 1e-6 {
        return;
    }
    let n = 48;
    for k in 0..n {
        let a0 = f64::from(k) * (std::f64::consts::TAU / f64::from(n));
        let a1 = f64::from(k + 1) * (std::f64::consts::TAU / f64::from(n));
        let p0 = view.map(outer * a0.cos(), outer * a0.sin());
        let p1 = view.map(outer * a1.cos(), outer * a1.sin());
        let q0 = view.map(inner * a0.cos(), inner * a0.sin());
        painter.line_segment([p0, p1], Stroke::new(2.4, DISK));
        painter.line_segment(
            [p0, q0],
            Stroke::new(1.1, Color32::from_rgba_unmultiplied(210, 140, 80, 90)),
        );
    }
}

fn paint_bundle(painter: &egui::Painter, view: &View, lesson: u8, mass: f64, phase: f32) {
    let rays = bundle(mass);
    let t = f64::from(phase.rem_euclid(TAU) / TAU);
    let b_e = einstein_b_course(mass);
    for ray in &rays {
        let color = if lesson == 2 {
            if ray.captured {
                CAP
            } else if (ray.b.abs() - b_e).abs() < 0.45 * mass.max(0.2) {
                EINSTEIN
            } else {
                ESC
            }
        } else {
            ray.fate(mass).color()
        };
        let width = if lesson == 2 && (ray.b.abs() - b_e).abs() < 0.45 * mass.max(0.2) {
            2.2
        } else {
            1.3
        };
        stroke_path(painter, view, &ray.points, color, width);
        if let Some(p) = point_on(&ray.points, t) {
            painter.circle_filled(view.map(p[0], p[1]), 3.4, NOW);
        }
    }
}

fn paint_source_and_eye(painter: &egui::Painter, view: &View, mass: f64) {
    let x = R_START_OVER_M * mass.max(0.35);
    let star = view.map(x, 0.0);
    let eye = view.map(-x * 0.92, 0.0);
    painter.circle_filled(star, 5.0, STAR);
    painter.text(
        star + egui::vec2(7.0, -4.0),
        egui::Align2::LEFT_BOTTOM,
        "źródło",
        FontId::proportional(11.0),
        STAR,
    );
    painter.circle_stroke(eye, 6.0, Stroke::new(1.6, CAM));
    painter.circle_filled(eye, 2.2, CAM);
    painter.text(
        eye + egui::vec2(-8.0, -4.0),
        egui::Align2::RIGHT_BOTTOM,
        "oko",
        FontId::proportional(11.0),
        CAM,
    );
}

fn paint_camera(painter: &egui::Painter, view: &View, mass: f64) {
    let r = R_START_OVER_M * mass.max(0.35);
    let p = view.map(-r * 0.92, 0.0);
    painter.line_segment(
        [p + egui::vec2(-10.0, -7.0), p + egui::vec2(8.0, 0.0)],
        Stroke::new(2.0, CAM),
    );
    painter.line_segment(
        [p + egui::vec2(-10.0, 7.0), p + egui::vec2(8.0, 0.0)],
        Stroke::new(2.0, CAM),
    );
    painter.text(
        p + egui::vec2(-8.0, 14.0),
        egui::Align2::RIGHT_TOP,
        "kamera",
        FontId::proportional(11.0),
        CAM,
    );
}

fn ellipse_pts(view: &View, r: f64, n: usize) -> Vec<Pos2> {
    (0..n)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            view.map(r * a.cos(), r * a.sin())
        })
        .collect()
}

fn fill_ellipse(painter: &egui::Painter, view: &View, r: f64, color: Color32) {
    if r <= 0.0 {
        return;
    }
    painter.add(Shape::convex_polygon(
        ellipse_pts(view, r, 48),
        color,
        Stroke::NONE,
    ));
}

fn stroke_ellipse(painter: &egui::Painter, view: &View, r: f64, color: Color32, width: f32) {
    if r <= 1e-9 {
        return;
    }
    let pts = ellipse_pts(view, r, 64);
    let n = pts.len();
    for i in 0..n {
        painter.line_segment([pts[i], pts[(i + 1) % n]], Stroke::new(width, color));
    }
}

fn label_ring(painter: &egui::Painter, view: &View, r: f64, text: &str, color: Color32) {
    if r <= 1e-9 {
        return;
    }
    let p = view.map(r * 0.72, r * 0.72);
    painter.text(
        p,
        egui::Align2::LEFT_BOTTOM,
        text,
        FontId::monospace(11.0),
        color,
    );
}

fn stroke_path(painter: &egui::Painter, view: &View, pts: &[[f64; 2]], color: Color32, width: f32) {
    for w in pts.windows(2) {
        let a = view.map(w[0][0], w[0][1]);
        let b = view.map(w[1][0], w[1][1]);
        painter.line_segment([a, b], Stroke::new(width, color));
    }
}

fn point_on(pts: &[[f64; 2]], t: f64) -> Option<[f64; 2]> {
    if pts.is_empty() {
        return None;
    }
    let u = t.clamp(0.0, 0.999) * (pts.len() - 1) as f64;
    Some(pts[u.floor() as usize])
}

fn bundle(mass: f64) -> Vec<Ray> {
    let key = metric_key(mass);
    {
        let guard = BUNDLE.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(cached) = guard.as_ref() {
            if cached.key == key {
                return cached.rays.clone();
            }
        }
    }
    let rays = trace_bundle(clamp_mass(mass));
    let mut guard = BUNDLE.lock().unwrap_or_else(|p| p.into_inner());
    *guard = Some(Bundle {
        key,
        rays: rays.clone(),
    });
    rays
}

pub fn trace_bundle(mass: f64) -> Vec<Ray> {
    let m = clamp_mass(mass);
    let mut rays = Vec::new();
    if m <= 1e-9 {
        let x = R_START_OVER_M * 0.35;
        for &k in &B_OVER_CRIT {
            if k == 0.0 {
                rays.push(straight(0.0, x));
            } else {
                let b = k * 4.0;
                rays.push(straight(b, x));
                rays.push(straight(-b, x));
            }
        }
        return rays;
    }
    let bc = photon_b_crit(m);
    for &k in &B_OVER_CRIT {
        if k == 0.0 {
            rays.push(trace_ray(m, 0.0));
        } else {
            rays.push(trace_ray(m, k * bc));
            rays.push(trace_ray(m, -k * bc));
        }
    }
    rays
}

fn straight(b: f64, x: f64) -> Ray {
    Ray {
        b,
        points: vec![[x, b], [-x, b]],
        periapsis: b.abs(),
        captured: false,
    }
}

fn trace_ray(mass: f64, b: f64) -> Ray {
    let Some(bh) = Schwarzschild::new(mass).ok() else {
        return straight(b, R_START_OVER_M);
    };
    let x_line = R_START_OVER_M * mass.max(0.35);
    let Some(start) = launch_null(bh, b, x_line) else {
        return Ray {
            b,
            points: Vec::new(),
            periapsis: b.abs(),
            captured: true,
        };
    };
    let mut y = start.to_array();
    let mut scratch = Scratch::with_len(STATE_LEN);
    let mut points = Vec::with_capacity(DROP_STEPS / 3 + 2);
    points.push(xy(start));
    let mut peri = start.r;
    let mut captured = false;
    let r_h = horizon_radius(mass);
    let r_out = x_line + 2.0 * mass.max(0.2);
    for i in 0..DROP_STEPS {
        if geodesic::step(bh, 0.0, &mut y, DROP_H, &mut scratch).is_err() {
            captured = true;
            break;
        }
        let Ok(s) = GeodesicState::from_slice(&y) else {
            captured = true;
            break;
        };
        if !s.r.is_finite() || s.r <= r_h + 0.05 * mass.max(0.2) {
            captured = true;
            points.push(xy(s));
            peri = peri.min(s.r.max(r_h));
            break;
        }
        peri = peri.min(s.r);
        if i % 3 == 0 {
            points.push(xy(s));
        }
        if s.r >= r_out && s.u_r > 0.0 {
            points.push(xy(s));
            break;
        }
    }
    Ray {
        b,
        points,
        periapsis: peri,
        captured,
    }
}

fn launch_null(bh: Schwarzschild, b: f64, x_line: f64) -> Option<GeodesicState> {
    let x = x_line.max(b.abs() + 1.0);
    let r = (x * x + b * b).sqrt();
    let phi = b.atan2(x);
    let energy = 1.0;
    let f = bh.f(r).ok()?;
    if f <= 0.0 {
        return None;
    }
    let ur2 = energy * energy - f * b * b / (r * r);
    if ur2 < -1e-12 {
        return None;
    }
    Some(GeodesicState {
        t: 0.0,
        r,
        theta: FRAC_PI_2,
        phi,
        u_t: energy / f,
        u_r: -ur2.max(0.0).sqrt(),
        u_theta: 0.0,
        u_phi: b / (r * r),
    })
}

fn xy(s: GeodesicState) -> [f64; 2] {
    let st = s.theta.sin();
    [s.r * st * s.phi.cos(), s.r * st * s.phi.sin()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_rings_match_the_metric_module() {
        let m = MASS_DEFAULT;
        assert!((horizon_radius(m) - 2.0).abs() < 1e-15);
        assert!((photon_sphere_radius(m) - 3.0).abs() < 1e-15);
        assert!((isco_radius(m) - 6.0).abs() < 1e-15);
        assert!((photon_b_crit(m) - 3.0 * 3.0_f64.sqrt()).abs() < 1e-15);
        assert_eq!(clamp_mass(-2.0), 0.0);
        assert_eq!(clamp_mass(99.0), MASS_MAX);
        assert_eq!(clamp_mass(f64::NAN), MASS_DEFAULT);
    }

    #[test]
    fn photon_laps_faster_than_the_isco_particle() {
        let w_ph = photon_omega(1.0).unwrap();
        let w_is = isco_omega(1.0).unwrap();
        assert!(w_ph > 3.0 * w_is, "{w_ph} vs {w_is}");
        assert!(photon_omega(0.0).is_none());
        assert!(isco_omega(0.0).is_none());
        let bh = Schwarzschild::new(1.0).unwrap();
        let ph = GeodesicState::circular_photon(bh).unwrap();
        assert!((ph.r - 3.0).abs() < 1e-15);
        let isco = GeodesicState::circular_timelike(bh, 6.0).unwrap();
        assert!((isco.r - 6.0).abs() < 1e-15);
    }

    #[test]
    fn weak_deflection_falls_with_impact_parameter() {
        let a = weak_deflection(1.0, 10.0).unwrap();
        let b = weak_deflection(1.0, 20.0).unwrap();
        assert!((a - 0.4).abs() < 1e-15);
        assert!(b < a);
        assert_eq!(weak_deflection(1.0, 0.0), None);
        assert_eq!(weak_deflection(-1.0, 4.0), None);
    }

    #[test]
    fn einstein_ring_grows_like_sqrt_mass_and_vanishes_flat() {
        assert_eq!(einstein_b(0.0, 30.0, 30.0), 0.0);
        let b1 = einstein_b(1.0, 30.0, 30.0);
        let b4 = einstein_b(4.0, 30.0, 30.0);
        assert!((b1 - 60.0_f64.sqrt()).abs() < 1e-12);
        assert!((b4 - 2.0 * b1).abs() < 1e-12);
        let course = einstein_b_course(1.0);
        assert!((course - b1).abs() < 1e-12);
        assert!(course > photon_b_crit(1.0));
    }

    #[test]
    fn small_impact_is_captured_large_escapes() {
        let rays = trace_bundle(1.0);
        let cap = rays.iter().find(|r| r.b.abs() < 1e-9).expect("oś");
        assert!(cap.captured, "b = 0 miało tonąć");
        let wide = rays
            .iter()
            .max_by(|a, b| a.b.abs().total_cmp(&b.b.abs()))
            .expect("pęk");
        assert!(!wide.captured, "duże b miało minąć");
        assert!(wide.periapsis > photon_sphere_radius(1.0));
        let edge = rays
            .iter()
            .find(|r| (r.b.abs() - photon_b_crit(1.0)).abs() < 0.05)
            .expect("krawędź b_c");
        assert!(edge.captured || edge.periapsis < 4.0);
    }

    #[test]
    fn face_on_is_a_circle_edge_on_flattens() {
        assert!((disk_aspect(0.0) - 1.0).abs() < 1e-6);
        assert!(disk_aspect(INCLINE_MAX) < 0.25);
        assert_eq!(clamp_incline(-1.0), 0.0);
        assert_eq!(clamp_incline(9.0), INCLINE_MAX);
        assert!((disk_outer(1.0) - DISK_OUTER_OVER_M).abs() < 1e-15);
        assert!((camera_r(1.0) - CAMERA_R_OVER_M).abs() < 1e-15);
    }

    #[test]
    fn bh_lessons_paint_without_a_window() {
        for index in 1..=4 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.1,
                    mass: MASS_DEFAULT,
                    spin: SPIN_DEFAULT,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
                assert!((playback.mass - MASS_DEFAULT).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }
}
