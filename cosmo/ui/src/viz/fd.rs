//! Ruchomy obraz ścieżki siatka PDE: węzły, FTCS, leapfrog, półka vs garnek.
//!
//! Liczby biorą się z [`bone_core::gr::fd`]. UI nie woła CUDA i nie zgaduje
//! metryki. Play/pauza. Bez labu. Residual analitycznego ciepła na węzłach
//! ma być ≈ 0 na ekranie.

use std::f32::consts::TAU;

use bone_core::gr::fd::{
    heat_dt, heat_node_residual, run_heat, run_wave, wave_dt, wave_node_residual, Mesh, C_WAVE,
    K_HEAT, LAMBDA_SAFE, R_SAFE,
};
use bone_core::gr::pinn::{Net, HIDDEN};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::lesson::Playback;
use crate::viz::metric_grid::flamm_z;

/// Komórki odcinka [0, 1]. 17 koralików jeszcze czyta się okiem.
pub const CELLS: usize = 16;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(28, 34, 44);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const TEAL: Color32 = Color32::from_rgb(120, 200, 190);
const GOLD: Color32 = Color32::from_rgb(220, 180, 90);
const BRICK: Color32 = Color32::from_rgb(220, 130, 110);

pub fn clamp_x(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.05, 0.95)
    } else {
        0.5
    }
}

pub fn clamp_r(r: f64) -> f64 {
    if r.is_finite() {
        r.clamp(0.15, 0.85)
    } else {
        R_SAFE
    }
}

pub fn clamp_lambda(lambda: f64) -> f64 {
    if lambda.is_finite() {
        lambda.clamp(0.2, 1.2)
    } else {
        LAMBDA_SAFE
    }
}

pub fn film_time(phase: f32) -> f64 {
    f64::from(phase.rem_euclid(TAU) / TAU) * 0.35
}

fn mesh() -> Mesh {
    Mesh::new(CELLS).expect("CELLS ≥ 2")
}

fn steps_heat(r: f64, t: f64) -> usize {
    let Ok(dt) = heat_dt(K_HEAT, mesh().dx(), r) else {
        return 0;
    };
    if dt <= 0.0 {
        return 0;
    }
    (t / dt).floor() as usize
}

fn steps_wave(lambda: f64, t: f64) -> usize {
    let Ok(dt) = wave_dt(C_WAVE, mesh().dx(), lambda) else {
        return 0;
    };
    if dt <= 0.0 {
        return 0;
    }
    (t / dt).floor() as usize
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    playback.mass = if playback.mass.is_finite() {
        playback.mass.clamp(0.0, MASS_MAX)
    } else {
        MASS_DEFAULT
    };
    match lesson {
        1 => {
            if playback.probe > 1.5 {
                playback.probe = 0.5;
            }
            playback.probe = clamp_x(playback.probe);
        }
        2 => {
            if playback.probe > 1.5 {
                playback.probe = R_SAFE;
            }
            playback.probe = clamp_r(playback.probe);
        }
        3 => {
            if playback.probe > 1.5 {
                playback.probe = LAMBDA_SAFE;
            }
            playback.probe = clamp_lambda(playback.probe);
        }
        _ => {}
    }
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        match lesson {
            1 => {
                ui.add(
                    egui::Slider::new(&mut playback.probe, 0.05..=0.95)
                        .text("x")
                        .fixed_decimals(2),
                );
                playback.probe = clamp_x(playback.probe);
            }
            2 => {
                ui.add(
                    egui::Slider::new(&mut playback.probe, 0.15..=0.85)
                        .text("r")
                        .fixed_decimals(2),
                );
                playback.probe = clamp_r(playback.probe);
            }
            3 => {
                ui.add(
                    egui::Slider::new(&mut playback.probe, 0.2..=1.2)
                        .text("λ")
                        .fixed_decimals(2),
                );
                playback.probe = clamp_lambda(playback.probe);
            }
            _ => {
                ui.add(
                    egui::Slider::new(&mut playback.mass, 0.0..=MASS_MAX)
                        .text("M")
                        .fixed_decimals(2),
                );
                playback.mass = if playback.mass.is_finite() {
                    playback.mass.clamp(0.0, MASS_MAX)
                } else {
                    MASS_DEFAULT
                };
            }
        }
        ui.label(RichText::new(hud(lesson, playback)).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let t = film_time(playback.t);
    match lesson {
        1 => paint_nodes(ui, rect, playback.probe, t),
        2 => paint_heat(ui, rect, playback.probe, t),
        3 => paint_wave(ui, rect, playback.probe, t),
        _ => paint_shelf(ui, rect, playback.mass, t),
    }
}

fn hud(lesson: u8, playback: &Playback) -> String {
    let t = film_time(playback.t);
    let rh = heat_node_residual(K_HEAT, 0.5, t);
    match lesson {
        1 => {
            let x = clamp_x(playback.probe);
            format!("N={}  x={x:.2}  R_węzły = {rh:.2e}", mesh().nodes())
        }
        2 => {
            let r = clamp_r(playback.probe);
            if r > 0.5 {
                format!("r={r:.2}  CFL pęka  R_węzły = {rh:.2e}")
            } else {
                format!("r={r:.2}  R_węzły = {rh:.2e}")
            }
        }
        3 => {
            let lambda = clamp_lambda(playback.probe);
            let rw = wave_node_residual(C_WAVE, 0.5, t);
            if lambda > 1.0 {
                format!("λ={lambda:.2}  CFL pęka  R_fala = {rw:.2e}")
            } else {
                format!("λ={lambda:.2}  R_fala = {rw:.2e}")
            }
        }
        _ => format!("1×N vs 10×N³  R_węzły = {rh:.2e}"),
    }
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "koraliki na drucie  ·  wzmacniacz PINN ma suwaki, nie węzły",
        2 => "garnek FTCS  ·  r > 1/2 i koraliki szaleją",
        3 => "struna leapfrog  ·  nowa wysokość z dwóch starych",
        _ => "siatka umie u_i  ·  półka trzyma g_μν  ·  strzałka przekreślona",
    }
}

fn paint_nodes(ui: &Ui, rect: Rect, x: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 40.0 {
        return;
    }
    let left = left_half(inner);
    let right = right_half(inner);
    let r = R_SAFE;
    let u = run_heat(mesh(), K_HEAT, r, steps_heat(r, t)).unwrap_or_default();
    beads(&painter, left, &u, Some(clamp_x(x)), false);
    painter.text(
        Pos2::new(left.left() + 8.0, left.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "węzły  u_i",
        FontId::proportional(11.0),
        LABEL,
    );
    paint_net(&painter, right, x, t);
}

fn paint_net(painter: &egui::Painter, rect: Rect, x: f64, t: f64) {
    let net = Net::biased();
    let hidden = net.hidden(x, t);
    let u = net.eval(x, t);
    let left = rect.left() + 28.0;
    let mid = rect.center().x;
    let right = rect.right() - 28.0;
    let ins = [
        Pos2::new(left, rect.center().y - 22.0),
        Pos2::new(left, rect.center().y + 22.0),
    ];
    let mut hs = [Pos2::ZERO; HIDDEN];
    let hspan = rect.height() * 0.62;
    let h0 = rect.center().y - hspan * 0.5;
    for (i, slot) in hs.iter_mut().enumerate() {
        *slot = Pos2::new(mid, h0 + hspan * (i as f32) / (HIDDEN as f32 - 1.0));
    }
    let out = Pos2::new(right, rect.center().y);
    for p in ins {
        for h in hs {
            painter.line_segment([p, h], Stroke::new(1.0, GRID));
        }
    }
    for (h, act) in hs.iter().zip(hidden) {
        painter.line_segment([*h, out], Stroke::new(1.0, GRID));
        let bright = ((act + 1.0) * 0.5).clamp(0.0, 1.0);
        let c = Color32::from_rgb(
            (40.0 + 180.0 * bright) as u8,
            (80.0 + 140.0 * bright) as u8,
            (90.0 + 120.0 * bright) as u8,
        );
        painter.circle_filled(*h, 5.0, c);
    }
    painter.circle_filled(ins[0], 7.0, TEAL);
    painter.circle_filled(ins[1], 7.0, TEAL);
    painter.circle_filled(out, 10.0, GOLD);
    painter.text(
        ins[0] + Vec2::new(-10.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        "x",
        FontId::monospace(11.0),
        TEAL,
    );
    painter.text(
        ins[1] + Vec2::new(-10.0, 0.0),
        egui::Align2::RIGHT_CENTER,
        "t",
        FontId::monospace(11.0),
        TEAL,
    );
    painter.text(
        out + Vec2::new(10.0, 0.0),
        egui::Align2::LEFT_CENTER,
        format!("u={u:.2}"),
        FontId::monospace(12.0),
        GOLD,
    );
    painter.text(
        Pos2::new(rect.left() + 8.0, rect.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "wzmacniacz PINN",
        FontId::proportional(11.0),
        LABEL,
    );
}

fn paint_heat(ui: &Ui, rect: Rect, r: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 40.0 {
        return;
    }
    let r = clamp_r(r);
    let u = run_heat(mesh(), K_HEAT, r, steps_heat(r, t)).unwrap_or_default();
    let explode = r > 0.5;
    beads(&painter, inner, &u, None, explode);
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 4.0),
        egui::Align2::LEFT_TOP,
        if explode {
            "FTCS  r > 1/2  koraliki szaleją"
        } else {
            "garnek FTCS  sin(πx) e^{−π² k t}"
        },
        FontId::proportional(11.0),
        LABEL,
    );
}

fn paint_wave(ui: &Ui, rect: Rect, lambda: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 40.0 {
        return;
    }
    let lambda = clamp_lambda(lambda);
    let u = run_wave(mesh(), C_WAVE, lambda, steps_wave(lambda, t)).unwrap_or_default();
    let explode = lambda > 1.0;
    beads(&painter, inner, &u, None, explode);
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 4.0),
        egui::Align2::LEFT_TOP,
        if explode {
            "leapfrog  λ > 1  struna pęka liczbami"
        } else {
            "struna leapfrog  sin(πx) cos(c π t)"
        },
        FontId::proportional(11.0),
        LABEL,
    );
}

fn paint_shelf(ui: &Ui, rect: Rect, mass: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(20.0);
    if inner.width() < 40.0 {
        return;
    }
    let left = left_half(inner);
    let right = right_half(inner);
    let u = run_heat(mesh(), K_HEAT, R_SAFE, steps_heat(R_SAFE, t)).unwrap_or_default();
    beads(&painter, left, &u, None, false);
    painter.text(
        Pos2::new(left.left() + 8.0, left.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "garnek  1 liczba × N",
        FontId::proportional(11.0),
        LABEL,
    );
    let c = right.center();
    let m = mass.clamp(0.0, MASS_MAX);
    let scale = right.width().min(right.height()) * 0.02;
    painter.circle_stroke(c, (2.0 * m) as f32 * scale * 8.0, Stroke::new(2.0, GRID));
    painter.circle_stroke(c, (3.0 * m) as f32 * scale * 8.0, Stroke::new(2.0, GOLD));
    painter.circle_stroke(c, (6.0 * m) as f32 * scale * 8.0, Stroke::new(2.0, TEAL));
    let z = flamm_z(6.0 * m.max(0.2), m);
    painter.text(
        Pos2::new(right.center().x, right.bottom() - 8.0),
        egui::Align2::CENTER_BOTTOM,
        format!("półka Schwarzschilda  z(6M)={z:.2}"),
        FontId::proportional(11.0),
        LABEL,
    );
    painter.text(
        Pos2::new(right.left() + 8.0, right.top() + 4.0),
        egui::Align2::LEFT_TOP,
        "10 kratek × N³",
        FontId::proportional(11.0),
        LABEL,
    );
    let a = Pos2::new(left.right() - 8.0, left.center().y);
    let b = Pos2::new(right.left() + 8.0, right.center().y);
    painter.line_segment([a, b], Stroke::new(2.0, BRICK));
    let mid = Pos2::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5);
    painter.line_segment(
        [mid + Vec2::new(-10.0, -10.0), mid + Vec2::new(10.0, 10.0)],
        Stroke::new(3.0, BRICK),
    );
    painter.line_segment(
        [mid + Vec2::new(-10.0, 10.0), mid + Vec2::new(10.0, -10.0)],
        Stroke::new(3.0, BRICK),
    );
}

fn beads(painter: &egui::Painter, rect: Rect, u: &[f64], highlight: Option<f64>, explode: bool) {
    if u.len() < 2 {
        return;
    }
    let n = u.len();
    let amp = rect.height() * 0.36;
    let y0 = rect.center().y + 8.0;
    let mut prev = None;
    let hi = highlight.map(|x| {
        let i = (x * (n as f64 - 1.0)).round() as usize;
        i.min(n - 1)
    });
    for (i, &val) in u.iter().enumerate() {
        let xf = i as f32 / (n as f32 - 1.0);
        let px = rect.left() + rect.width() * xf;
        let clamped = val.clamp(-2.5, 2.5);
        let py = y0 - (clamped as f32) * amp;
        let p = Pos2::new(px, py);
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(1.5, if explode { BRICK } else { GRID }));
        }
        prev = Some(p);
        let hot = (val.abs() / 1.2).clamp(0.0, 1.0) as f32;
        let color = if explode && val.abs() > 1.2 {
            BRICK
        } else if hi == Some(i) {
            GOLD
        } else {
            Color32::from_rgb(
                (40.0 + 180.0 * hot) as u8,
                (90.0 + 80.0 * (1.0 - hot)) as u8,
                (110.0 + 80.0 * (1.0 - hot)) as u8,
            )
        };
        let rad = if hi == Some(i) { 7.0 } else { 5.0 };
        painter.circle_filled(p, rad, color);
    }
}

fn left_half(inner: Rect) -> Rect {
    Rect::from_min_max(
        inner.left_top(),
        Pos2::new(inner.center().x - 10.0, inner.bottom()),
    )
}

fn right_half(inner: Rect) -> Rect {
    Rect::from_min_max(
        Pos2::new(inner.center().x + 10.0, inner.top()),
        inner.right_bottom(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_heat_residual_on_nodes_vanishes() {
        let r = heat_node_residual(K_HEAT, 0.5, 0.05);
        assert!(r.abs() < 2e-6, "R={r}");
    }

    #[test]
    fn analytic_wave_residual_on_nodes_vanishes() {
        let r = wave_node_residual(C_WAVE, 0.4, 0.1);
        assert!(r.abs() < 2e-4, "R={r}");
    }

    #[test]
    fn film_time_stays_in_the_pot() {
        assert!((film_time(0.0) - 0.0).abs() < 1e-15);
        assert!(film_time(TAU) < 1e-6);
        assert!(film_time(TAU * 0.5) > 0.1);
    }

    #[test]
    fn default_cfl_stays_under_the_lid() {
        const {
            assert!(R_SAFE <= 0.5);
            assert!(LAMBDA_SAFE <= 1.0);
        }
        assert!((clamp_r(R_SAFE) - R_SAFE).abs() < 1e-15);
        assert!((clamp_r(2.0) - 0.85).abs() < 1e-15);
        assert!((clamp_lambda(LAMBDA_SAFE) - LAMBDA_SAFE).abs() < 1e-15);
        assert!((clamp_lambda(3.0) - 1.2).abs() < 1e-15);
    }

    #[test]
    fn pde_lessons_paint_without_a_window() {
        for index in 1..=4 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.1,
                    probe: 0.5,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
            });
            let _ = ctx.end_pass();
        }
    }
}
