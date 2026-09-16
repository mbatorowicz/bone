//! Ruchomy obraz ścieżki PINN: sieć, residual, ciepło/fala, półka vs garnek.
//!
//! Liczby biorą się z [`bone_core::gr::pinn`]. UI nie woła PyTorcha i nie
//! zgaduje metryki. Kerr zostaje za mapą.

use std::f32::consts::TAU;

use bone_core::gr::pinn::{heat_exact, heat_residual, wave_exact, wave_residual, Net, HIDDEN};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui, Vec2};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::lesson::Playback;
use crate::viz::metric_grid::flamm_z;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(28, 34, 44);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const TEAL: Color32 = Color32::from_rgb(120, 200, 190);
const GOLD: Color32 = Color32::from_rgb(220, 180, 90);
const BRICK: Color32 = Color32::from_rgb(220, 130, 110);
const NUM: Color32 = Color32::from_rgb(230, 230, 240);
const GREEN: Color32 = Color32::from_rgb(160, 200, 140);

pub const K_HEAT: f64 = 0.3;
pub const C_WAVE: f64 = 1.0;

pub fn clamp_x(x: f64) -> f64 {
    if x.is_finite() {
        x.clamp(0.05, 0.95)
    } else {
        0.5
    }
}

pub fn pde_time(phase: f32) -> f64 {
    f64::from(phase.rem_euclid(TAU) / TAU) * 0.35
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    if playback.probe > 1.5 {
        playback.probe = 0.5;
    }
    playback.probe = clamp_x(playback.probe);
    playback.mass = if playback.mass.is_finite() {
        playback.mass.clamp(0.0, MASS_MAX)
    } else {
        MASS_DEFAULT
    };
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut playback.probe, 0.05..=0.95)
                .text("x")
                .fixed_decimals(2),
        );
        playback.probe = clamp_x(playback.probe);
        ui.label(RichText::new(hud(lesson, playback)).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let t = pde_time(playback.t);
    match lesson {
        1 => paint_net(ui, rect, playback.probe, t),
        2 => paint_residual(ui, rect, playback.probe, t),
        3 => paint_pdes(ui, rect, t),
        _ => paint_shelf(ui, rect, playback.mass, t),
    }
}

fn hud(lesson: u8, playback: &Playback) -> String {
    let x = clamp_x(playback.probe);
    let t = pde_time(playback.t);
    match lesson {
        1 => {
            let u = Net::biased().eval(x, t);
            format!("u(x, t) = {u:.3}")
        }
        2 => {
            let exact = heat_residual(K_HEAT, x, t, |xx, tt| heat_exact(K_HEAT, xx, tt));
            let lie = heat_residual(K_HEAT, x, t, |xx, tt| Net::biased().eval(xx, tt));
            format!("R_sin = {exact:.2e}   R_sieć = {lie:.2e}")
        }
        3 => {
            let rh = heat_residual(K_HEAT, x, t, |xx, tt| heat_exact(K_HEAT, xx, tt));
            let rw = wave_residual(C_WAVE, x, t, |xx, tt| wave_exact(C_WAVE, xx, tt));
            format!("R_ciepło = {rh:.2e}   R_fala = {rw:.2e}")
        }
        _ => "garnek umie u(x, t)  ·  półka trzyma g_μν".into(),
    }
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "wzmacniacz 2→8→1  ·  suwaki są f64, tanh jest schodkiem",
        2 => "termometr residualu  ·  analityczny sinus zostawia R ≈ 0",
        3 => "dwa paski, jeden fortel  ·  ciepło stygnie, struna drga",
        _ => "nie dziś  ·  zgadywanie studni jest droższe niż garnek",
    }
}

fn paint_net(ui: &Ui, rect: Rect, x: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(24.0);
    if inner.width() < 40.0 {
        return;
    }
    let net = Net::biased();
    let hidden = net.hidden(x, t);
    let u = net.eval(x, t);
    let left = inner.left() + 36.0;
    let mid = inner.center().x;
    let right = inner.right() - 36.0;
    let in_y = [inner.center().y - 28.0, inner.center().y + 28.0];
    let ins = [Pos2::new(left, in_y[0]), Pos2::new(left, in_y[1])];
    let mut hs = [Pos2::ZERO; HIDDEN];
    let hspan = inner.height() * 0.72;
    let h0 = inner.center().y - hspan * 0.5;
    for (i, slot) in hs.iter_mut().enumerate() {
        *slot = Pos2::new(mid, h0 + hspan * (i as f32) / (HIDDEN as f32 - 1.0));
    }
    let out = Pos2::new(right, inner.center().y);
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
        painter.circle_filled(*h, 7.0, c);
    }
    painter.circle_filled(ins[0], 9.0, TEAL);
    painter.circle_filled(ins[1], 9.0, TEAL);
    painter.circle_filled(out, 12.0, GOLD);
    painter.text(ins[0] + Vec2::new(-14.0, 0.0), egui::Align2::RIGHT_CENTER, "x", FontId::monospace(12.0), TEAL);
    painter.text(ins[1] + Vec2::new(-14.0, 0.0), egui::Align2::RIGHT_CENTER, "t", FontId::monospace(12.0), TEAL);
    painter.text(out + Vec2::new(14.0, 0.0), egui::Align2::LEFT_CENTER, format!("u={u:.2}"), FontId::monospace(13.0), GOLD);
}

fn paint_residual(ui: &Ui, rect: Rect, x: f64, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(24.0);
    if inner.width() < 40.0 {
        return;
    }
    let exact = heat_residual(K_HEAT, x, t, |xx, tt| heat_exact(K_HEAT, xx, tt)).abs();
    let lie = heat_residual(K_HEAT, x, t, |xx, tt| Net::biased().eval(xx, tt)).abs();
    thermometer(&painter, left_half(inner), exact, 1.0e-4, GREEN, "sinus ciepła");
    thermometer(&painter, right_half(inner), lie, 8.0, BRICK, "sieć biased");
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

fn thermometer(
    painter: &egui::Painter,
    rect: Rect,
    value: f64,
    scale: f64,
    color: Color32,
    title: &str,
) {
    painter.text(
        Pos2::new(rect.center().x, rect.top() + 6.0),
        egui::Align2::CENTER_TOP,
        title,
        FontId::proportional(12.0),
        LABEL,
    );
    let stem = Rect::from_center_size(
        Pos2::new(rect.center().x, rect.center().y + 8.0),
        Vec2::new(22.0, rect.height() * 0.7),
    );
    painter.rect_stroke(stem, 4.0, Stroke::new(1.0, GRID), egui::StrokeKind::Middle);
    let fill = ((value / scale).clamp(0.0, 1.0) as f32) * stem.height();
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(stem.left() + 3.0, stem.bottom() - fill),
            Pos2::new(stem.right() - 3.0, stem.bottom()),
        ),
        3.0,
        color,
    );
    painter.text(
        Pos2::new(rect.center().x, rect.bottom() - 6.0),
        egui::Align2::CENTER_BOTTOM,
        format!("|R| = {value:.2e}"),
        FontId::monospace(12.0),
        NUM,
    );
}

fn paint_pdes(ui: &Ui, rect: Rect, t: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.height() < 40.0 {
        return;
    }
    let top = Rect::from_min_max(inner.left_top(), Pos2::new(inner.right(), inner.center().y - 6.0));
    let bot = Rect::from_min_max(Pos2::new(inner.left(), inner.center().y + 6.0), inner.right_bottom());
    strip(&painter, top, t, true);
    strip(&painter, bot, t, false);
}

fn strip(painter: &egui::Painter, rect: Rect, t: f64, heat: bool) {
    let n = 64;
    let mut prev = None;
    for i in 0..=n {
        let x = f64::from(i) / f64::from(n);
        let u = if heat {
            heat_exact(K_HEAT, x, t)
        } else {
            wave_exact(C_WAVE, x, t)
        };
        let px = rect.left() + rect.width() * x as f32;
        let py = rect.center().y - (u as f32) * rect.height() * 0.38;
        let p = Pos2::new(px, py);
        if let Some(q) = prev {
            painter.line_segment([q, p], Stroke::new(2.0, if heat { GOLD } else { TEAL }));
        }
        prev = Some(p);
    }
    painter.text(
        Pos2::new(rect.left() + 8.0, rect.top() + 6.0),
        egui::Align2::LEFT_TOP,
        if heat { "ciepło  sin(πx) e^{−π² k t}" } else { "fala  sin(πx) cos(c π t)" },
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
    strip(&painter, left, t, true);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_heat_residual_on_the_probe_vanishes() {
        let r = heat_residual(K_HEAT, 0.5, 0.05, |x, t| heat_exact(K_HEAT, x, t));
        assert!(r.abs() < 2e-6, "R={r}");
    }

    #[test]
    fn analytic_wave_residual_on_the_probe_vanishes() {
        let r = wave_residual(C_WAVE, 0.4, 0.1, |x, t| wave_exact(C_WAVE, x, t));
        assert!(r.abs() < 2e-4, "R={r}");
    }

    #[test]
    fn biased_net_does_not_solve_heat() {
        let r = heat_residual(K_HEAT, 0.5, 0.05, |x, t| Net::biased().eval(x, t));
        assert!(r.abs() > 1e-3, "biased miał kłamać, R={r}");
    }

    #[test]
    fn pde_time_stays_in_the_film() {
        assert!((pde_time(0.0) - 0.0).abs() < 1e-15);
        assert!(pde_time(TAU) < 1e-6);
        assert!(pde_time(TAU * 0.5) > 0.1);
    }

    #[test]
    fn pinn_lessons_paint_without_a_window() {
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
                assert!((clamp_x(playback.probe) - playback.probe).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }
}
