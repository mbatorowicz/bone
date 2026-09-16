//! Ruchomy obraz ścieżki Einstein: trampolina, pętla, szalki, łąka próżni.
//!
//! Liczby biorą się z [`bone_core::gr::einstein`] i [`bone_core::gr::metric`].
//! UI nie zgaduje metryki. Kerr i siatka PDE tu nie wchodzą.

use std::f32::consts::TAU;
use std::f64::consts::{FRAC_PI_2, PI};

use bone_core::gr::{
    dust, field_residual, horizon_radius, isco_radius, photon_sphere_radius, vacuum, Schwarzschild,
    Tensor02,
};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use super::metric_grid::{self, flamm_z, R_MAX};
use super::MASS_MAX;
use crate::lesson::Playback;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(52, 64, 80);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const GOLD: Color32 = Color32::from_rgb(220, 180, 90);
const BLUE: Color32 = Color32::from_rgb(120, 180, 220);
const GREEN: Color32 = Color32::from_rgb(160, 200, 140);
const BRICK: Color32 = Color32::from_rgb(220, 130, 110);
const NUM: Color32 = Color32::from_rgb(230, 230, 240);
const HORIZON: Color32 = Color32::from_rgb(90, 90, 100);

pub fn clamp_mass(mass: f64) -> f64 {
    metric_grid::clamp_mass(mass)
}

pub fn clamp_radius(mass: f64, r: f64) -> f64 {
    let m = clamp_mass(mass);
    let floor = if m <= 1e-9 {
        0.8
    } else {
        horizon_radius(m) + 0.35 * m.max(0.2)
    };
    let raw = if r.is_finite() { r } else { 6.0 * m.max(1.0) };
    raw.clamp(floor, 16.0)
}

pub fn clamp_density(density: f64) -> f64 {
    if density.is_finite() {
        density.clamp(0.0, 3.0)
    } else {
        0.0
    }
}

pub fn kretschmann_book(mass: f64, r: f64) -> f64 {
    48.0 * mass * mass / r.powi(6)
}

pub fn tensor_max_abs(t: Tensor02) -> f64 {
    t.components()
        .into_iter()
        .flatten()
        .fold(0.0_f64, |a, x| a.max(x.abs()))
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    playback.mass = clamp_mass(playback.mass);
    playback.probe = clamp_radius(playback.mass, playback.probe);
    playback.density = clamp_density(playback.density);
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
        match lesson {
            3 => {
                ui.add(
                    egui::Slider::new(&mut playback.density, 0.0..=3.0)
                        .text("ρ")
                        .fixed_decimals(2),
                );
                playback.density = clamp_density(playback.density);
            }
            _ => {
                ui.add(
                    egui::Slider::new(&mut playback.probe, 2.2..=14.0)
                        .text("r")
                        .fixed_decimals(2),
                );
                playback.probe = clamp_radius(playback.mass, playback.probe);
            }
        }
        ui.label(RichText::new(hud(lesson, playback)).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    match lesson {
        1 => paint_trampoline(ui, rect, playback.mass, playback.t),
        2 => paint_loop(ui, rect, playback.mass, playback.probe, playback.t),
        3 => paint_pans(ui, rect, playback.mass, playback.probe, playback.density),
        _ => paint_meadow(ui, rect, playback.mass, playback.probe, playback.t),
    }
}

fn hud(lesson: u8, playback: &Playback) -> String {
    let m = playback.mass;
    let r = clamp_radius(m, playback.probe);
    match lesson {
        1 => format!(
            "2M={:.2}  3M={:.2}  6M={:.2}",
            horizon_radius(m),
            photon_sphere_radius(m),
            isco_radius(m)
        ),
        2 => match curvature_at(m, r) {
            Some(c) => format!("K = {:.4e}   książka = {:.4e}", c, kretschmann_book(m, r)),
            None => "horyzont".into(),
        },
        3 => {
            let (g, t8, res) = pans(m, r, playback.density);
            format!("|G|={g:.2e}  |8πT|={t8:.2e}  |res|={res:.2e}")
        }
        _ => match einstein_max(m, r) {
            Some(g) => format!("|G| = {g:.2e}   (próżnia chce 0)"),
            None => "horyzont".into(),
        },
    }
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "mata z kulą  ·  kreska nie skręca kierownicą — „prosto” na macie jest łukiem",
        2 => "pętla na macie  ·  Kretschmann 48 M²/r⁶, nie ozdoba",
        3 => "dwie szalki  ·  próżnia ρ = 0 jest G = 0; pył na płaskiej macie nie stoi",
        _ => "łąka T = 0  ·  trzy kółka są skutkiem rozwiązania, nie plakatem",
    }
}

fn curvature_at(mass: f64, r: f64) -> Option<f64> {
    Schwarzschild::new(clamp_mass(mass))
        .ok()?
        .kretschmann(r, FRAC_PI_2)
        .ok()
}

fn einstein_max(mass: f64, r: f64) -> Option<f64> {
    let c = Schwarzschild::new(clamp_mass(mass))
        .ok()?
        .curvature(r, FRAC_PI_2)
        .ok()?;
    Some(tensor_max_abs(c.einstein()))
}

fn pans(mass: f64, r: f64, density: f64) -> (f64, f64, f64) {
    let g = einstein_max(mass, r).unwrap_or(0.0);
    let rho = clamp_density(density);
    let t = dust(rho, [1.0, 0.0, 0.0, 0.0], Tensor02::minkowski());
    let t8 = 8.0 * PI * tensor_max_abs(t);
    let vacuum_g = Schwarzschild::new(clamp_mass(mass))
        .ok()
        .and_then(|bh| bh.curvature(r, FRAC_PI_2).ok())
        .map(|c| c.einstein())
        .unwrap_or_else(vacuum);
    let res = tensor_max_abs(field_residual(vacuum_g, t));
    (g, t8, res)
}

fn paint_trampoline(ui: &Ui, rect: Rect, mass: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(22.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = clamp_mass(mass);
    let hole = if m <= 1e-9 { 0.4 } else { horizon_radius(m) };
    let mut prev = None;
    let samples = 64;
    for i in 0..=samples {
        let r = hole + (R_MAX - hole) * f64::from(i) / f64::from(samples);
        let z = flamm_z(r, m);
        let p = flamm_px(inner, r, z);
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(2.0, GRID));
        }
        prev = Some(p);
        let p_m = flamm_px(inner, -r, z);
        if i > 0 {
            let r_prev = hole + (R_MAX - hole) * f64::from(i - 1) / f64::from(samples);
            let q_m = flamm_px(inner, -r_prev, flamm_z(r_prev, m));
            painter.line_segment([q_m, p_m], Stroke::new(2.0, GRID));
        }
    }
    if m > 1e-6 {
        let h = flamm_px(inner, hole, 0.0);
        let h_m = flamm_px(inner, -hole, 0.0);
        painter.line_segment([h_m, h], Stroke::new(3.0, HORIZON));
    }
    let marble_r = hole + 1.2 + 5.0 * (0.5 + 0.5 * f64::from(phase.sin()));
    let z = flamm_z(marble_r, m);
    painter.circle_filled(flamm_px(inner, marble_r, z), 6.0, NUM);
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        "przekrój Flamma  ·  kreska toczy się po macie, nie po siłach",
        FontId::proportional(11.0),
        LABEL,
    );
}

fn flamm_px(inner: Rect, r: f64, z: f64) -> Pos2 {
    let x = inner.center().x + (r / R_MAX) as f32 * inner.width() * 0.42;
    let y = inner.center().y + inner.height() * 0.12 - (z / 8.0) as f32 * inner.height() * 0.45;
    Pos2::new(x, y)
}

fn paint_loop(ui: &Ui, rect: Rect, mass: f64, r: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(24.0);
    if inner.width() < 16.0 {
        return;
    }
    let c = inner.center();
    let rad = inner.width().min(inner.height()) * 0.32;
    let m = clamp_mass(mass);
    let k = curvature_at(m, r).unwrap_or(0.0);
    let squash = (1.0 + (k * 1.0e3).tanh() as f32 * 0.35).clamp(1.0, 1.4);
    let mut prev = None;
    let n = 48;
    for i in 0..=n {
        let a = TAU * (i as f32) / (n as f32);
        let p = Pos2::new(c.x + rad * a.cos() * squash, c.y + rad * a.sin() / squash);
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(2.0, BRICK));
        }
        prev = Some(p);
    }
    let a = phase;
    let ant = Pos2::new(c.x + rad * a.cos() * squash, c.y + rad * a.sin() / squash);
    let a2 = phase + 0.35;
    let ant2 = Pos2::new(
        c.x + (rad - 10.0) * a2.cos() * squash,
        c.y + (rad - 10.0) * a2.sin() / squash,
    );
    painter.circle_filled(ant, 5.5, GOLD);
    painter.circle_filled(ant2, 5.5, BLUE);
    painter.text(
        Pos2::new(inner.center().x, inner.top() + 8.0),
        egui::Align2::CENTER_TOP,
        "dwie mrówki  ·  pętla wraca inna, gdy mata pamięta masę",
        FontId::proportional(11.0),
        LABEL,
    );
}

fn paint_pans(ui: &Ui, rect: Rect, mass: f64, r: f64, density: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(20.0);
    if inner.width() < 40.0 {
        return;
    }
    let (g, t8, res) = pans(mass, r, density);
    let left = Rect::from_min_max(
        Pos2::new(inner.left(), inner.top() + 28.0),
        Pos2::new(inner.center().x - 12.0, inner.bottom() - 28.0),
    );
    let right = Rect::from_min_max(
        Pos2::new(inner.center().x + 12.0, inner.top() + 28.0),
        Pos2::new(inner.right(), inner.bottom() - 28.0),
    );
    pan(&painter, left, g, 0.2, GREEN, "G  łąka");
    pan(&painter, right, t8, 80.0, BRICK, "8π T  pył");
    painter.text(
        Pos2::new(inner.center().x, inner.bottom() - 6.0),
        egui::Align2::CENTER_BOTTOM,
        format!("|G − 8πT| = {res:.3e}   ρ = {density:.2}"),
        FontId::monospace(12.0),
        NUM,
    );
}

fn pan(painter: &egui::Painter, rect: Rect, value: f64, scale: f64, color: Color32, title: &str) {
    painter.rect_stroke(rect, 6.0, Stroke::new(1.0, GRID), egui::StrokeKind::Middle);
    painter.text(
        Pos2::new(rect.center().x, rect.top() + 8.0),
        egui::Align2::CENTER_TOP,
        title,
        FontId::proportional(12.0),
        LABEL,
    );
    let fill = ((value / scale).clamp(0.0, 1.0) as f32) * rect.height() * 0.7;
    let floor = Rect::from_min_max(
        Pos2::new(rect.left() + 16.0, rect.bottom() - 16.0 - fill),
        Pos2::new(rect.right() - 16.0, rect.bottom() - 16.0),
    );
    painter.rect_filled(floor, 3.0, color);
}

fn paint_meadow(ui: &Ui, rect: Rect, mass: f64, r_probe: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(24.0);
    if inner.width() < 16.0 {
        return;
    }
    let m = clamp_mass(mass);
    let c = inner.center();
    let scale = inner.width().min(inner.height()) * 0.42 / 16.0;
    let px = |r: f64| r as f32 * scale;
    ring(&painter, c, px(horizon_radius(m)), HORIZON, "2M");
    ring(&painter, c, px(photon_sphere_radius(m)), GOLD, "3M");
    ring(&painter, c, px(isco_radius(m)), BLUE, "6M");
    painter.circle_filled(c, 5.0, BRICK);
    let ang = phase;
    let p = c + Vec2::new(px(r_probe) * ang.cos(), px(r_probe) * ang.sin());
    painter.circle_filled(p, 6.0, NUM);
    let g = einstein_max(m, r_probe).unwrap_or(f64::NAN);
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 8.0),
        egui::Align2::LEFT_TOP,
        format!("T = 0  na łące   |G| = {g:.2e}"),
        FontId::monospace(12.0),
        GREEN,
    );
}

fn ring(painter: &egui::Painter, c: Pos2, radius: f32, color: Color32, label: &str) {
    if radius < 2.0 {
        return;
    }
    painter.circle_stroke(c, radius, Stroke::new(2.0, color));
    painter.text(
        c + Vec2::new(radius + 6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        FontId::proportional(11.0),
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::MASS_DEFAULT;

    #[test]
    fn kretschmann_on_the_slider_matches_the_engine() {
        let mass = MASS_DEFAULT;
        let r = 6.0;
        let book = kretschmann_book(mass, r);
        let got = curvature_at(mass, r).expect("6M jest poza horyzontem");
        assert!((got - book).abs() / book < 1e-5);
        let engine = Schwarzschild::new(mass)
            .unwrap()
            .kretschmann(r, FRAC_PI_2)
            .unwrap();
        assert!((got - engine).abs() < 1e-15);
    }

    #[test]
    fn vacuum_pans_are_empty() {
        let (g, t8, res) = pans(1.0, 6.0, 0.0);
        assert!(g < 1e-6, "G={g}");
        assert!(t8 < 1e-15);
        assert!(res < 1e-6, "res={res}");
    }

    #[test]
    fn dust_on_the_meadow_leaves_a_residual() {
        let (_, t8, res) = pans(1.0, 6.0, 1.0);
        assert!(t8 > 1.0);
        assert!(res > 1.0);
        let t = dust(1.0, [1.0, 0.0, 0.0, 0.0], Tensor02::minkowski());
        assert!((t.components()[0][0] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn probe_stays_outside_the_horizon() {
        assert!(clamp_radius(1.0, 6.0) > 2.0);
        assert!(clamp_radius(1.0, 0.1) > 2.0);
    }

    #[test]
    fn einstein_lessons_paint_without_a_window() {
        for index in 1..=4 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.1,
                    mass: MASS_DEFAULT,
                    probe: 6.0,
                    density: 0.0,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
                assert!((playback.mass - MASS_DEFAULT).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }
}
