//! Gumowa siatka Schwarzschilda: Flamm, suwak M, trzy promienie, orbita 6M.
//!
//! Współrzędna `r` to etykieta. Oczko o stałym Δr ma właściwą długość
//! `√g_rr Δr` z [`bone_core::gr::metric`]. Zanurzenie Flamma
//! `z = 2 √(2M(r − 2M))` składa z tego studnię, którą widać. Lekcja 4
//! sadza na ISCO punkt, którego E i L biorą się z [`GeodesicState`].

use std::f32::consts::TAU;

use bone_core::gr::{
    horizon_radius, isco_radius, photon_sphere_radius, GeodesicState, MetricError, Schwarzschild,
};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::lesson::Playback;

/// Zasięg etykiety `r` na siatce. 6M przy `MASS_MAX` jeszcze się mieści.
pub const R_MAX: f64 = 16.0;
/// Δr taśmy z lekcji 2: ten sam skok etykiety, inna długość.
pub const TAPE_DR: f64 = 0.5;
const TILT: f32 = 0.58;
const NR: usize = 10;
const NPHI: usize = 20;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(52, 64, 80);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const HORIZON: Color32 = Color32::from_rgb(20, 22, 28);
const RING_H: Color32 = Color32::from_rgb(90, 90, 100);
const RING_PH: Color32 = Color32::from_rgb(220, 180, 90);
const RING_ISCO: Color32 = Color32::from_rgb(120, 180, 220);
const TAPE_NEAR: Color32 = Color32::from_rgb(220, 180, 90);
const TAPE_FAR: Color32 = Color32::from_rgb(160, 200, 140);
const PARTICLE: Color32 = Color32::from_rgb(230, 230, 240);

pub fn clamp_mass(mass: f64) -> f64 {
    if !mass.is_finite() {
        MASS_DEFAULT
    } else {
        mass.clamp(0.0, MASS_MAX)
    }
}

pub fn metric(mass: f64) -> Result<Schwarzschild, MetricError> {
    Schwarzschild::new(clamp_mass(mass))
}

/// Zanurzenie Flamma. Zero na horyzoncie i przy `M = 0`.
pub fn flamm_z(r: f64, mass: f64) -> f64 {
    let m = clamp_mass(mass);
    if m <= 0.0 || !r.is_finite() {
        return 0.0;
    }
    let hole = horizon_radius(m);
    if r <= hole {
        return 0.0;
    }
    2.0 * (2.0 * m * (r - hole)).sqrt()
}

/// `ds = √g_rr Δr` — lokalna linijka wzdłuż promienia, poza horyzontem.
pub fn radial_proper(mass: f64, r: f64, dr: f64) -> Result<f64, MetricError> {
    let bh = metric(mass)?;
    Ok(bh.g_rr(r)?.sqrt() * dr)
}

fn r_inner(mass: f64) -> f64 {
    let m = clamp_mass(mass);
    if m <= 1e-9 {
        0.5
    } else {
        (horizon_radius(m) + 0.18 * m.max(0.2)).min(R_MAX * 0.45)
    }
}

/// ISCO: położenie w równiku i stałe E, L z silnika. `None` przy M = 0.
pub fn isco_on_sheet(mass: f64, lambda: f64) -> Option<(f64, f64, f64, f64, f64)> {
    let m = clamp_mass(mass);
    if m <= 1e-9 {
        return None;
    }
    let bh = Schwarzschild::new(m).ok()?;
    let s = GeodesicState::circular_timelike(bh, isco_radius(m)).ok()?;
    let phi = s.u_phi * lambda;
    Some((
        s.r * phi.cos(),
        s.r * phi.sin(),
        s.r,
        s.energy(bh).ok()?,
        s.angular_momentum(bh).ok()?,
    ))
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
        let m = playback.mass;
        ui.label(
            RichText::new(format!(
                "2M={:.2}  3M={:.2}  6M={:.2}",
                horizon_radius(m),
                photon_sphere_radius(m),
                isco_radius(m)
            ))
            .monospace(),
        );
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    paint(ui, rect, lesson, playback.mass, playback.t);
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        2 => "to samo Δr, inna taśma  ·  blisko studni oczko jest dłuższe",
        3 => "trzy kółka, nie jedno  ·  2M / 3M / 6M rosną z M",
        _ => "punkt na ISCO  ·  E i L z geodezyjnej, nie z plakatu",
    }
}

fn paint(ui: &Ui, rect: Rect, lesson: u8, mass: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(16.0);
    if inner.width() < 16.0 || inner.height() < 16.0 {
        return;
    }
    let m = clamp_mass(mass);
    let z_edge = flamm_z(R_MAX, m);
    let extent = (R_MAX + z_edge * 0.35).max(4.0);
    let cam = Cam {
        c: inner.center() + Vec2::new(0.0, inner.height() * 0.06),
        scale: inner.width().min(inner.height()) * 0.42 / extent as f32,
        tilt: TILT,
    };
    let inner_r = r_inner(m);

    draw_grid(&painter, &cam, m, inner_r);
    if m > 1e-6 {
        fill_horizon(&painter, &cam, m);
        if lesson >= 3 {
            ring(&painter, &cam, photon_sphere_radius(m), m, RING_PH, "3M");
            ring(&painter, &cam, isco_radius(m), m, RING_ISCO, "6M");
            ring(&painter, &cam, horizon_radius(m), m, RING_H, "2M");
        } else {
            ring(&painter, &cam, horizon_radius(m), m, RING_H, "");
        }
    }

    if lesson == 2 {
        draw_tapes(&painter, &cam, m);
    }
    if lesson == 4 {
        draw_orbit(&painter, &cam, m, phase);
    }

    painter.text(
        Pos2::new(inner.left() + 6.0, inner.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        footer(lesson, m),
        FontId::proportional(11.0),
        LABEL,
    );
}

fn footer(lesson: u8, mass: f64) -> String {
    match lesson {
        2 => {
            let near_r = tape_near_r(mass);
            let far_r = tape_far_r(mass);
            let near = radial_proper(mass, near_r, TAPE_DR).unwrap_or(TAPE_DR);
            let far = radial_proper(mass, far_r, TAPE_DR).unwrap_or(TAPE_DR);
            format!("Δr = {TAPE_DR}  →  ds(blisko) = {near:.3}   ds(daleko) = {far:.3}")
        }
        3 => "horyzont puchnie z M; dalekie oczka zostają niemal zeszytowe".into(),
        _ if mass <= 1e-9 => "M = 0: prosta, jak test geodezyjnej bez masy".into(),
        _ => match isco_on_sheet(mass, 0.0) {
            Some((_, _, _, e, l)) => format!("ISCO  E = {e:.4}  L = {l:.4}"),
            None => "brak ISCO".into(),
        },
    }
}

fn tape_near_r(mass: f64) -> f64 {
    let m = clamp_mass(mass);
    if m <= 1e-9 {
        2.0
    } else {
        (4.0 * m).clamp(r_inner(m) + TAPE_DR, R_MAX - 2.0)
    }
}

fn tape_far_r(mass: f64) -> f64 {
    (12.0_f64).min(R_MAX - TAPE_DR).max(tape_near_r(mass) + 2.0)
}

struct Cam {
    c: Pos2,
    scale: f32,
    tilt: f32,
}

fn project(cam: &Cam, x: f64, y: f64, z: f64) -> Pos2 {
    let ct = cam.tilt.cos();
    let st = cam.tilt.sin();
    let y_s = y as f32 * ct + (-z as f32) * st;
    Pos2::new(
        cam.c.x + cam.scale * x as f32,
        cam.c.y - cam.scale * y_s,
    )
}

fn sheet_point(r: f64, phi: f64, mass: f64) -> (f64, f64, f64) {
    (r * phi.cos(), r * phi.sin(), flamm_z(r, mass))
}

fn draw_grid(painter: &egui::Painter, cam: &Cam, mass: f64, inner_r: f64) {
    for i in 0..=NR {
        let r = inner_r + (R_MAX - inner_r) * f64::from(i as u32) / NR as f64;
        let mut prev = None;
        for k in 0..=NPHI {
            let phi = std::f64::consts::TAU * f64::from(k as u32) / NPHI as f64;
            let (x, y, z) = sheet_point(r, phi, mass);
            let p = project(cam, x, y, z);
            if let Some(q) = prev {
                painter.line_segment([q, p], Stroke::new(1.0, GRID));
            }
            prev = Some(p);
        }
    }
    for k in 0..NPHI {
        let phi = std::f64::consts::TAU * f64::from(k as u32) / NPHI as f64;
        let mut prev = None;
        for i in 0..=NR {
            let r = inner_r + (R_MAX - inner_r) * f64::from(i as u32) / NR as f64;
            let (x, y, z) = sheet_point(r, phi, mass);
            let p = project(cam, x, y, z);
            if let Some(q) = prev {
                painter.line_segment([q, p], Stroke::new(1.0, GRID));
            }
            prev = Some(p);
        }
    }
}

fn fill_horizon(painter: &egui::Painter, cam: &Cam, mass: f64) {
    let r = horizon_radius(mass).max(0.05);
    let mut pts = Vec::with_capacity(NPHI + 1);
    for k in 0..=NPHI {
        let phi = std::f64::consts::TAU * f64::from(k as u32) / NPHI as f64;
        let (x, y, z) = sheet_point(r, phi, mass);
        pts.push(project(cam, x, y, z));
    }
    painter.add(egui::Shape::convex_polygon(
        pts,
        HORIZON,
        Stroke::new(1.5, RING_H),
    ));
}

fn ring(painter: &egui::Painter, cam: &Cam, r: f64, mass: f64, color: Color32, label: &str) {
    if r <= 0.05 || r > R_MAX {
        return;
    }
    let mut prev = None;
    let mut first = None;
    for k in 0..=NPHI {
        let phi = std::f64::consts::TAU * f64::from(k as u32) / NPHI as f64;
        let (x, y, z) = sheet_point(r, phi, mass);
        let p = project(cam, x, y, z);
        if first.is_none() {
            first = Some(p);
        }
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(2.0, color));
        }
        prev = Some(p);
    }
    if !label.is_empty() {
        let (x, y, z) = sheet_point(r, 0.35, mass);
        let p = project(cam, x, y, z);
        painter.text(
            p + Vec2::new(6.0, 0.0),
            egui::Align2::LEFT_CENTER,
            label,
            FontId::monospace(11.0),
            color,
        );
    }
}

fn draw_tapes(painter: &egui::Painter, cam: &Cam, mass: f64) {
    tape(painter, cam, mass, tape_near_r(mass), TAPE_NEAR);
    tape(painter, cam, mass, tape_far_r(mass), TAPE_FAR);
}

fn tape(painter: &egui::Painter, cam: &Cam, mass: f64, r: f64, color: Color32) {
    let phi = 0.35;
    let (x0, y0, z0) = sheet_point(r, phi, mass);
    let (x1, y1, z1) = sheet_point(r + TAPE_DR, phi, mass);
    let a = project(cam, x0, y0, z0);
    let b = project(cam, x1, y1, z1);
    painter.line_segment([a, b], Stroke::new(3.5, color));
    painter.circle_filled(a, 3.0, color);
    painter.circle_filled(b, 3.0, color);
}

fn draw_orbit(painter: &egui::Painter, cam: &Cam, mass: f64, phase: f32) {
    if mass <= 1e-9 {
        let t = f64::from(phase.rem_euclid(TAU) / TAU);
        let x = -0.8 * R_MAX + t * 1.6 * R_MAX;
        let a = project(cam, -0.8 * R_MAX, 2.5, 0.0);
        let b = project(cam, 0.8 * R_MAX, 2.5, 0.0);
        painter.line_segment([a, b], Stroke::new(1.5, TAPE_FAR));
        painter.circle_filled(project(cam, x, 2.5, 0.0), 5.5, PARTICLE);
        return;
    }
    let r = isco_radius(mass);
    ring(painter, cam, r, mass, RING_ISCO, "");
    let lambda = f64::from(phase.rem_euclid(TAU)) * 8.0;
    if let Some((x, y, _, _, _)) = isco_on_sheet(mass, lambda) {
        let z = flamm_z(r, mass);
        painter.circle_filled(project(cam, x, y, z), 6.0, PARTICLE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_slider_stays_nonnegative() {
        assert_eq!(clamp_mass(1.0), 1.0);
        assert_eq!(clamp_mass(-4.0), 0.0);
        assert_eq!(clamp_mass(99.0), MASS_MAX);
        assert_eq!(clamp_mass(f64::NAN), MASS_DEFAULT);
        assert!(metric(0.0).is_ok());
        assert!(metric(MASS_MAX).is_ok());
    }

    #[test]
    fn rings_match_the_metric_module() {
        let m = MASS_DEFAULT;
        assert!((horizon_radius(m) - 2.0).abs() < 1e-15);
        assert!((photon_sphere_radius(m) - 3.0).abs() < 1e-15);
        assert!((isco_radius(m) - 6.0).abs() < 1e-15);
        let bh = metric(m).unwrap();
        assert!((bh.horizon_radius() - 2.0 * m).abs() < 1e-15);
        assert!((bh.photon_sphere_radius() - 3.0 * m).abs() < 1e-15);
        assert!((bh.isco_radius() - 6.0 * m).abs() < 1e-15);
    }

    #[test]
    fn flamm_is_flat_without_mass_and_zero_on_the_horizon() {
        assert_eq!(flamm_z(8.0, 0.0), 0.0);
        assert_eq!(flamm_z(2.0, 1.0), 0.0);
        assert!((flamm_z(6.0, 1.0) - 2.0 * (2.0 * 4.0_f64).sqrt()).abs() < 1e-15);
        assert!(flamm_z(12.0, 1.0) > flamm_z(6.0, 1.0));
        assert!(flamm_z(8.0, 2.0) > flamm_z(8.0, 1.0));
    }

    #[test]
    fn radial_ruler_uses_sqrt_g_rr() {
        let ds = radial_proper(1.0, 4.0, TAPE_DR).unwrap();
        assert!((ds - TAPE_DR * 2.0_f64.sqrt()).abs() < 1e-15);
        let far = radial_proper(1.0, 12.0, TAPE_DR).unwrap();
        assert!(ds > far);
        let flat = radial_proper(0.0, 4.0, TAPE_DR).unwrap();
        assert!((flat - TAPE_DR).abs() < 1e-15);
        assert!(radial_proper(1.0, 2.0, TAPE_DR).is_err());
    }

    #[test]
    fn isco_energy_and_l_match_geodesic_tests() {
        let (_, _, r, e, l) = isco_on_sheet(1.0, 0.0).unwrap();
        assert!((r - 6.0).abs() < 1e-15);
        let want_e = 4.0 / 18.0_f64.sqrt();
        let want_l = 12.0_f64.sqrt();
        assert!((e - want_e).abs() < 1e-15);
        assert!((l - want_l).abs() < 1e-15);
        let (x, y, _, _, _) = isco_on_sheet(1.0, 0.0).unwrap();
        assert!((x - 6.0).abs() < 1e-12 && y.abs() < 1e-12);
        assert!(isco_on_sheet(0.0, 0.0).is_none());
    }

    #[test]
    fn geo_lessons_2_to_4_paint_without_a_window() {
        for index in 2..=4 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.2,
                    mass: MASS_DEFAULT,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
                assert!((playback.mass - MASS_DEFAULT).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }
}
