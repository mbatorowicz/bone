//! Ruchomy obraz ścieżki Kerr: wleczenie, ergo, pęk pierścieni, zapowiedź cienia.
//!
//! Liczby biorą się z [`bone_core::gr::kerr`]. Ten plik nie woła
//! [`bone_core::gr::raytrace::Buffer`]: klatka 320×180 zostaje laboratorium.
//! Suwak `a` w raytracerze jest w [`blackhole`]. Ten plik zostaje przekrojem.

use std::f64::consts::FRAC_PI_2;

use bone_core::gr::Kerr;
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Shape, Stroke, Ui, Vec2};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::lesson::Playback;
use crate::screen::LabId;

/// Suwak `a/M` nie dochodzi do 1: ekstremalny Kerr jest kreską tej maty.
pub const CHI_MAX: f32 = 0.998;
/// Start: wleczenie widać od razu, zero jest na suwaku.
pub const CHI_DEFAULT: f32 = 0.85;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const HORIZON_FILL: Color32 = Color32::from_rgb(12, 14, 18);
const RING_H: Color32 = Color32::from_rgb(90, 90, 100);
const RING_PH: Color32 = Color32::from_rgb(220, 180, 90);
const RING_PH_M: Color32 = Color32::from_rgb(180, 140, 70);
const RING_ISCO: Color32 = Color32::from_rgb(120, 180, 220);
const RING_ISCO_M: Color32 = Color32::from_rgb(80, 130, 170);
const ERGO: Color32 = Color32::from_rgb(200, 110, 90);
const ERGO_FILL: Color32 = Color32::from_rgb(52, 28, 26);
const ARROW: Color32 = Color32::from_rgb(230, 170, 90);
const NOW: Color32 = Color32::from_rgb(230, 230, 240);
const SHADOW: Color32 = Color32::from_rgb(18, 16, 22);
const GHOST: Color32 = Color32::from_rgb(70, 78, 92);
const CAM: Color32 = Color32::from_rgb(160, 200, 140);

pub fn clamp_mass(mass: f64) -> f64 {
    if !mass.is_finite() {
        MASS_DEFAULT
    } else {
        mass.clamp(0.0, MASS_MAX)
    }
}

pub fn clamp_chi(chi: f32) -> f32 {
    if !chi.is_finite() {
        0.0
    } else {
        chi.clamp(0.0, CHI_MAX)
    }
}

/// `Kerr { M, a = χ M }`. Zero masy to Minkowski, nie `0/0`.
pub fn kerr_of(mass: f64, chi: f32) -> Option<Kerr> {
    let m = clamp_mass(mass);
    let a = f64::from(clamp_chi(chi)) * m;
    Kerr::new(m, a).ok()
}

/// ω ZAMO: `−g_tφ / g_φφ`. Gaśnie przy `a = 0`.
pub fn zamo_omega(bh: Kerr, r: f64, theta: f64) -> Option<f64> {
    let gtphi = bh.g_t_phi(r, theta).ok()?;
    let gphiphi = bh.g_phi_phi(r, theta).ok()?;
    if !gphiphi.is_finite() || gphiphi.abs() < 1e-18 {
        return None;
    }
    let w = -gtphi / gphiphi;
    w.is_finite().then_some(w)
}

/// `Ω = ±√M / (r^{3/2} ± a √M)` — ten sam wzór co koło w `gr::kerr`.
pub fn circular_omega(bh: Kerr, r: f64, prograde: bool) -> Option<f64> {
    let m = bh.mass();
    let a = bh.spin();
    if m <= 0.0 || r <= 0.0 {
        return None;
    }
    let sqrt_m = m.sqrt();
    let r32 = r.powf(1.5);
    let denom = if prograde {
        r32 + a * sqrt_m
    } else {
        r32 - a * sqrt_m
    };
    if !denom.is_finite() || denom.abs() < 1e-18 {
        return None;
    }
    let w = if prograde {
        sqrt_m / denom
    } else {
        -sqrt_m / denom
    };
    w.is_finite().then_some(w)
}

/// Parametr zderzenia fotonu na orbicie kołowej: `b = L/E`.
///
/// Przy `a = 0` to `3√3 M`. Przy spinie lewa i prawa krawędź się rozjeżdżają.
pub fn photon_impact(bh: Kerr, prograde: bool) -> Option<f64> {
    let r = if prograde {
        bh.photon_plus()
    } else {
        bh.photon_minus()
    };
    if r <= 0.0 {
        return None;
    }
    let omega = circular_omega(bh, r, prograde)?;
    let gtt = bh.g_tt(r, FRAC_PI_2).ok()?;
    let gtphi = bh.g_t_phi(r, FRAC_PI_2).ok()?;
    let gphiphi = bh.g_phi_phi(r, FRAC_PI_2).ok()?;
    let e = -(gtt + gtphi * omega);
    let ell = gtphi + gphiphi * omega;
    if !e.is_finite() || !ell.is_finite() || e.abs() < 1e-18 {
        return None;
    }
    let b = ell / e;
    b.is_finite().then_some(b)
}

/// Drzwi z lekcji 4: animacja zostaje, laboratorium bierze a/M z suwaka.
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
                "wejście → {} · 320×180 w tle · a/M z tej karty",
                LabId::BlackHole.label()
            ))
            .small()
            .weak()
            .monospace(),
        );
    });
    ui.label(
        RichText::new("Laboratorium doliczy cień. Ścieżka C nadal startuje od a = 0.")
            .small()
            .weak(),
    );
    ui.add_space(6.0);
    open
}

/// `true`, gdy lekcja 4 otwiera raytracer.
pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) -> bool {
    let mut open = false;
    if lesson == 4 {
        open = draw_ray_door(ui);
    }
    playback.mass = clamp_mass(playback.mass);
    playback.spin = clamp_chi(playback.spin);
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut playback.spin, 0.0..=CHI_MAX)
                .text("a/M")
                .fixed_decimals(3),
        );
        playback.spin = clamp_chi(playback.spin);
        ui.label(RichText::new(hud(playback)).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    if let Some(bh) = kerr_of(playback.mass, playback.spin) {
        match lesson {
            1 => paint_drag(ui, rect, bh, playback.t),
            2 => paint_ergo(ui, rect, bh),
            3 => paint_split(ui, rect, bh, playback.t),
            _ => paint_shadow(ui, rect, bh),
        }
    }
    open
}

fn hud(playback: &Playback) -> String {
    let Some(bh) = kerr_of(playback.mass, playback.spin) else {
        return "Kerr —".into();
    };
    let chi = f64::from(clamp_chi(playback.spin));
    if chi < 1e-6 {
        format!(
            "a/M=0  r+={:.2}  γ={:.2}  ISCO={:.2}",
            bh.horizon_radius(),
            bh.photon_plus(),
            bh.isco_plus()
        )
    } else {
        format!(
            "a/M={chi:.3}  r+={:.2}  γ+={:.2}/{:.2}  ISCO+={:.2}/{:.2}",
            bh.horizon_radius(),
            bh.photon_plus(),
            bh.photon_minus(),
            bh.isco_plus(),
            bh.isco_minus()
        )
    }
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "strzałki to ω ZAMO z g_tφ · a = 0 gasi kratkę, wracają 2M / 3M / 6M",
        2 => "czarne r+ · rdzawe ergo · na równiku 2M, na biegunach klei się do horyzontu",
        3 => "złote foton± · niebieskie ISCO± · plus z prądem, minus pod prąd",
        _ => "zapowiedź cienia: b z prądem ≠ b pod prąd · klatka 320×180 zostaje w labie",
    }
}

struct View {
    c: Pos2,
    scale: f32,
}

impl View {
    fn map(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(
            self.c.x + self.scale * x as f32,
            self.c.y - self.scale * y as f32,
        )
    }
}

fn view_at(inner: Rect, span: f64) -> View {
    let span = span.max(1.0);
    View {
        c: inner.center(),
        scale: inner.width().min(inner.height()) * 0.46 / span as f32,
    }
}

fn paint_drag(ui: &Ui, rect: Rect, bh: Kerr, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = bh.mass().max(0.35);
    let view = view_at(inner, (9.0 * m).max(10.0));
    let rh = bh.horizon_radius();
    fill_circle(&painter, &view, rh, HORIZON_FILL);
    stroke_circle(&painter, &view, rh, RING_H, 2.0);
    label_ring(&painter, &view, rh, "r+", RING_H);
    if bh.spin().abs() < 1e-9 {
        stroke_circle(&painter, &view, bh.photon_plus(), RING_PH, 2.0);
        label_ring(&painter, &view, bh.photon_plus(), "3M", RING_PH);
        stroke_circle(&painter, &view, bh.isco_plus(), RING_ISCO, 2.0);
        label_ring(&painter, &view, bh.isco_plus(), "6M", RING_ISCO);
        paint_dot(&painter, &view, bh.photon_plus(), phase, RING_PH, true);
        paint_dot(&painter, &view, bh.isco_plus(), phase, RING_ISCO, true);
    } else {
        if let Ok(ergo) = bh.ergo(FRAC_PI_2) {
            stroke_circle(&painter, &view, ergo, ERGO, 1.6);
            label_ring(&painter, &view, ergo, "ergo", ERGO);
        }
        paint_arrows(&painter, &view, bh, phase);
    }
    footer(
        &painter,
        inner,
        format!(
            "g_tφ(6M) = {:.3}  ·  ω_ZAMO(6M) = {:.3}",
            bh.g_t_phi(6.0 * bh.mass().max(1e-9), FRAC_PI_2)
                .unwrap_or(0.0),
            zamo_omega(bh, 6.0 * bh.mass().max(1e-9), FRAC_PI_2).unwrap_or(0.0)
        ),
    );
}

fn paint_ergo(ui: &Ui, rect: Rect, bh: Kerr) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = bh.mass().max(0.35);
    let view = view_at(inner, (2.6 * m).max(4.0));
    let rh = bh.horizon_radius();
    let ergo_pts = meridional_pts(&view, |theta| bh.ergo(theta).unwrap_or(rh), 96);
    if ergo_pts.len() >= 3 {
        painter.add(Shape::convex_polygon(ergo_pts, ERGO_FILL, Stroke::NONE));
    }
    stroke_meridian(
        &painter,
        &view,
        |theta| bh.ergo(theta).unwrap_or(rh),
        ERGO,
        2.2,
    );
    fill_circle(&painter, &view, rh, HORIZON_FILL);
    stroke_circle(&painter, &view, rh, RING_H, 2.0);
    painter.text(
        view.map(0.0, rh.max(0.4)),
        egui::Align2::CENTER_BOTTOM,
        "r+",
        FontId::monospace(11.0),
        RING_H,
    );
    if let Ok(eq) = bh.ergo(FRAC_PI_2) {
        painter.text(
            view.map(eq, 0.0) + Vec2::new(6.0, 0.0),
            egui::Align2::LEFT_CENTER,
            "2M",
            FontId::monospace(11.0),
            ERGO,
        );
    }
    painter.line_segment(
        [view.map(-2.4 * m, 0.0), view.map(2.4 * m, 0.0)],
        Stroke::new(1.0, GHOST),
    );
    painter.line_segment(
        [view.map(0.0, -2.4 * m), view.map(0.0, 2.4 * m)],
        Stroke::new(1.0, GHOST),
    );
    footer(
        &painter,
        inner,
        format!(
            "równik ergo = {:.2}  ·  biegun = {:.2}  ·  r+ = {:.2}",
            bh.ergo(FRAC_PI_2).unwrap_or(0.0),
            bh.ergo(0.0).unwrap_or(0.0),
            rh
        ),
    );
}

fn paint_split(ui: &Ui, rect: Rect, bh: Kerr, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = bh.mass().max(0.35);
    let outer = bh.isco_minus().max(6.0 * m) + 1.5 * m;
    let view = view_at(inner, outer.max(10.0));
    let rh = bh.horizon_radius();
    fill_circle(&painter, &view, rh, HORIZON_FILL);
    stroke_circle(&painter, &view, rh, RING_H, 2.2);
    label_ring(&painter, &view, rh, "r+", RING_H);
    let glued = bh.spin().abs() < 1e-9;
    stroke_circle(&painter, &view, bh.photon_plus(), RING_PH, 2.0);
    label_ring(
        &painter,
        &view,
        bh.photon_plus(),
        if glued { "3M" } else { "γ+" },
        RING_PH,
    );
    if !glued {
        stroke_circle(&painter, &view, bh.photon_minus(), RING_PH_M, 1.7);
        label_ring(&painter, &view, bh.photon_minus(), "γ−", RING_PH_M);
    }
    stroke_circle(&painter, &view, bh.isco_plus(), RING_ISCO, 2.0);
    label_ring(
        &painter,
        &view,
        bh.isco_plus(),
        if glued { "6M" } else { "I+" },
        RING_ISCO,
    );
    if !glued {
        stroke_circle(&painter, &view, bh.isco_minus(), RING_ISCO_M, 1.7);
        label_ring(&painter, &view, bh.isco_minus(), "I−", RING_ISCO_M);
    }
    paint_orbit_dot(&painter, &view, bh, bh.photon_plus(), true, phase, RING_PH);
    paint_orbit_dot(&painter, &view, bh, bh.isco_plus(), true, phase, RING_ISCO);
    if !glued {
        paint_orbit_dot(
            &painter,
            &view,
            bh,
            bh.photon_minus(),
            false,
            phase,
            RING_PH_M,
        );
        paint_orbit_dot(
            &painter,
            &view,
            bh,
            bh.isco_minus(),
            false,
            phase,
            RING_ISCO_M,
        );
    }
    footer(
        &painter,
        inner,
        format!(
            "γ+={:.2}  γ−={:.2}  I+={:.2}  I−={:.2}",
            bh.photon_plus(),
            bh.photon_minus(),
            bh.isco_plus(),
            bh.isco_minus()
        ),
    );
}

fn paint_shadow(ui: &Ui, rect: Rect, bh: Kerr) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = bh.mass().max(0.35);
    let b_schw = 3.0 * 3.0_f64.sqrt() * m;
    let b_plus = photon_impact(bh, true).map(f64::abs).unwrap_or(b_schw);
    let b_minus = photon_impact(bh, false).map(f64::abs).unwrap_or(b_schw);
    let span = b_minus.max(b_plus).max(b_schw) + 2.0 * m;
    let view = view_at(inner, span.max(8.0));
    stroke_circle(&painter, &view, b_schw, GHOST, 1.2);
    let n = 64;
    let mut blob = Vec::with_capacity(n);
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        let rad = ellipse_radius(b_plus, b_minus, a);
        blob.push(view.map(rad * a.cos(), rad * a.sin() * 0.92));
    }
    painter.add(Shape::convex_polygon(blob, SHADOW, Stroke::NONE));
    painter.circle_filled(view.c, 3.0, RING_H);
    painter.text(
        view.map(-b_minus, 0.0) + Vec2::new(-6.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        "pod prąd",
        FontId::proportional(11.0),
        RING_PH_M,
    );
    painter.text(
        view.map(b_plus, 0.0) + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        "z prądem",
        FontId::proportional(11.0),
        RING_PH,
    );
    let cam = view.map(span * 0.92, 0.0);
    painter.circle_stroke(cam, 6.0, Stroke::new(1.6, CAM));
    painter.circle_filled(cam, 2.2, CAM);
    footer(
        &painter,
        inner,
        format!(
            "b+={:.2}  b−={:.2}  b_c(a=0)={:.2}  ·  plama {} osi",
            b_plus,
            b_minus,
            b_schw,
            if (b_plus - b_minus).abs() < 0.05 * m.max(0.2) {
                "na"
            } else {
                "nie na"
            }
        ),
    );
}

fn ellipse_radius(b_plus: f64, b_minus: f64, angle: f64) -> f64 {
    let c = angle.cos();
    let s = angle.sin();
    let rx = if c >= 0.0 { b_plus } else { b_minus };
    let ry = 0.5 * (b_plus + b_minus);
    let denom = (c * c) / (rx * rx).max(1e-12) + (s * s) / (ry * ry).max(1e-12);
    if denom <= 0.0 {
        ry
    } else {
        1.0 / denom.sqrt()
    }
}

fn paint_arrows(painter: &egui::Painter, view: &View, bh: Kerr, phase: f32) {
    let m = bh.mass().max(0.2);
    let rh = bh.horizon_radius();
    let n_r = 4;
    for i in 0..n_r {
        let r = rh + (1.2 + 1.6 * f64::from(i)) * m;
        let Some(w) = zamo_omega(bh, r, FRAC_PI_2) else {
            continue;
        };
        let strength = (w.abs() * 8.0 * m).clamp(0.12, 0.85) as f32;
        let n_a = 8;
        for k in 0..n_a {
            let phi =
                std::f64::consts::TAU * (f64::from(k) + 0.04 * f64::from(phase)) / f64::from(n_a);
            let p = view.map(r * phi.cos(), r * phi.sin());
            let t = Vec2::new(-phi.sin() as f32, -phi.cos() as f32);
            let len = 16.0 * strength;
            arrow(painter, p, t * len);
        }
        let phi = w * f64::from(phase) * 0.35 + f64::from(i);
        let p = view.map(r * phi.cos(), r * phi.sin());
        painter.circle_filled(p, 3.6, NOW);
    }
}

fn arrow(painter: &egui::Painter, from: Pos2, dir: Vec2) {
    if dir.length() < 2.0 {
        return;
    }
    let to = from + dir;
    painter.line_segment([from, to], Stroke::new(1.6, ARROW));
    let n = dir.normalized();
    let back = to - n * 7.0;
    let side = Vec2::new(-n.y, n.x) * 3.5;
    painter.line_segment([to, back + side], Stroke::new(1.6, ARROW));
    painter.line_segment([to, back - side], Stroke::new(1.6, ARROW));
}

fn paint_dot(painter: &egui::Painter, view: &View, r: f64, phase: f32, color: Color32, plus: bool) {
    if r <= 1e-6 {
        return;
    }
    let sign = if plus { 1.0 } else { -1.0 };
    let phi = f64::from(phase) * sign;
    painter.circle_filled(view.map(r * phi.cos(), r * phi.sin()), 5.0, color);
}

fn paint_orbit_dot(
    painter: &egui::Painter,
    view: &View,
    bh: Kerr,
    r: f64,
    prograde: bool,
    phase: f32,
    color: Color32,
) {
    let w = circular_omega(bh, r, prograde).unwrap_or(if prograde { 1.0 } else { -1.0 });
    let phi = w * f64::from(phase) * 0.55;
    if r > 1e-6 {
        painter.circle_filled(view.map(r * phi.cos(), r * phi.sin()), 5.0, color);
    }
}

fn fill_circle(painter: &egui::Painter, view: &View, r: f64, color: Color32) {
    if r <= 0.0 {
        return;
    }
    painter.add(Shape::convex_polygon(
        circle_pts(view, r, 48),
        color,
        Stroke::NONE,
    ));
}

fn stroke_circle(painter: &egui::Painter, view: &View, r: f64, color: Color32, width: f32) {
    if r <= 1e-9 {
        return;
    }
    let pts = circle_pts(view, r, 64);
    let n = pts.len();
    for i in 0..n {
        painter.line_segment([pts[i], pts[(i + 1) % n]], Stroke::new(width, color));
    }
}

fn circle_pts(view: &View, r: f64, n: usize) -> Vec<Pos2> {
    (0..n)
        .map(|k| {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            view.map(r * a.cos(), r * a.sin())
        })
        .collect()
}

fn meridional_pts(view: &View, r_of: impl Fn(f64) -> f64, n: usize) -> Vec<Pos2> {
    (0..n)
        .map(|k| {
            let theta = std::f64::consts::TAU * k as f64 / n as f64;
            let r = r_of(theta);
            view.map(r * theta.sin(), r * theta.cos())
        })
        .collect()
}

fn stroke_meridian(
    painter: &egui::Painter,
    view: &View,
    r_of: impl Fn(f64) -> f64,
    color: Color32,
    width: f32,
) {
    let pts = meridional_pts(view, r_of, 96);
    let n = pts.len();
    if n < 2 {
        return;
    }
    for i in 0..n {
        painter.line_segment([pts[i], pts[(i + 1) % n]], Stroke::new(width, color));
    }
}

fn label_ring(painter: &egui::Painter, view: &View, r: f64, text: &str, color: Color32) {
    if r <= 1e-9 {
        return;
    }
    painter.text(
        view.map(r * 0.72, r * 0.72),
        egui::Align2::LEFT_BOTTOM,
        text,
        FontId::monospace(11.0),
        color,
    );
}

fn footer(painter: &egui::Painter, inner: Rect, text: String) {
    painter.text(
        Pos2::new(inner.left() + 6.0, inner.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        text,
        FontId::proportional(11.0),
        LABEL,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::MASS_DEFAULT;

    #[test]
    fn zero_spin_rings_are_the_schwarzschild_circles() {
        let bh = kerr_of(MASS_DEFAULT, 0.0).expect("a = 0");
        assert!((bh.horizon_radius() - 2.0).abs() < 1e-15);
        assert!((bh.photon_plus() - 3.0).abs() < 1e-15);
        assert!((bh.photon_minus() - 3.0).abs() < 1e-15);
        assert!((bh.isco_plus() - 6.0).abs() < 1e-12);
        assert!((bh.isco_minus() - 6.0).abs() < 1e-12);
        assert!((bh.ergo(FRAC_PI_2).unwrap() - 2.0).abs() < 1e-15);
        assert_eq!(bh.g_t_phi(6.0, FRAC_PI_2).unwrap(), 0.0);
        assert!(zamo_omega(bh, 6.0, FRAC_PI_2).unwrap().abs() < 1e-15);
    }

    #[test]
    fn spinning_splits_inside_and_outside_six() {
        let bh = kerr_of(1.0, 0.9).expect("χ = 0.9");
        assert!(bh.horizon_radius() < 2.0);
        assert!(bh.photon_plus() < 3.0);
        assert!(bh.photon_minus() > 3.0);
        assert!(bh.isco_plus() < 6.0);
        assert!(bh.isco_minus() > 6.0);
        assert!((bh.ergo(FRAC_PI_2).unwrap() - 2.0).abs() < 1e-12);
        assert!(bh.ergo(0.0).unwrap() < 2.0);
        let w = zamo_omega(bh, 6.0, FRAC_PI_2).unwrap();
        assert!(w > 0.0, "ω={w}");
    }

    #[test]
    fn photon_impact_at_zero_spin_is_three_root_three() {
        let bh = kerr_of(1.0, 0.0).unwrap();
        let b = photon_impact(bh, true).unwrap();
        assert!((b.abs() - 3.0 * 3.0_f64.sqrt()).abs() < 1e-9, "b={b}");
        let left = photon_impact(bh, true).unwrap().abs();
        let right = photon_impact(bh, false).unwrap().abs();
        assert!((left - right).abs() < 1e-9);
    }

    #[test]
    fn spinning_shadow_is_asymmetric() {
        let bh = kerr_of(1.0, 0.9).unwrap();
        let plus = photon_impact(bh, true).unwrap().abs();
        let minus = photon_impact(bh, false).unwrap().abs();
        assert!(plus < minus, "{plus} vs {minus}");
        assert!((plus - minus).abs() > 0.5);
    }

    #[test]
    fn chi_clamp_keeps_kerr_legal() {
        assert_eq!(clamp_chi(-1.0), 0.0);
        assert_eq!(clamp_chi(CHI_MAX), CHI_MAX);
        assert_eq!(clamp_chi(2.0), CHI_MAX);
        assert_eq!(clamp_chi(f32::NAN), 0.0);
        assert!(kerr_of(1.0, CHI_MAX).is_some());
        assert!(kerr_of(0.0, CHI_DEFAULT).is_some());
        assert_eq!(clamp_mass(f64::NAN), MASS_DEFAULT);
    }

    #[test]
    fn kerr_lessons_paint_without_a_window() {
        for index in 1..=4 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.1,
                    mass: MASS_DEFAULT,
                    spin: CHI_DEFAULT,
                    ..Default::default()
                };
                assert!(!draw(ui, index, &mut playback));
                assert!((f64::from(playback.spin) - f64::from(CHI_DEFAULT)).abs() < 1e-6);
            });
            let _ = ctx.end_pass();
        }
    }

    #[test]
    fn ray_door_is_closed_without_a_click() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            assert!(!draw_ray_door(ui));
        });
        let _ = ctx.end_pass();
    }
}
