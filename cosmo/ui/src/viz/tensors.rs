//! Ruchomy obraz ścieżki tensory: kartka, maszyna, η, linijka, przekładnia.
//!
//! Liczby biorą się z [`bone_core::gr::tensor`] i [`bone_core::gr::metric`].
//! UI tylko obraca osie i podpisuje kratki. Riemann i PINN tu nie wchodzą.

use std::f32::consts::TAU;

use bone_core::gr::{Schwarzschild, Tensor02, Tensor11, Vector};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Shape, Stroke, Ui, Vec2};

use super::{MASS_DEFAULT, MASS_MAX};
use crate::lesson::Playback;

/// Strzałka przyklejona do pokoju: stała w świecie, kostium tańczy na kartce.
pub const WORLD: Vector = Vector {
    t: 0.0,
    x: 1.40,
    y: 0.50,
    z: 0.0,
};
/// Druga strzałka do skrzynki η(u, v).
pub const OTHER: Vector = Vector {
    t: 0.0,
    x: 0.30,
    y: 1.10,
    z: 0.0,
};

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(28, 34, 44);
const AXIS: Color32 = Color32::from_rgb(90, 100, 118);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const PAPER: Color32 = Color32::from_rgb(190, 150, 220);
const ARROW: Color32 = Color32::from_rgb(120, 180, 220);
const OTHER_C: Color32 = Color32::from_rgb(160, 200, 140);
const GOLD: Color32 = Color32::from_rgb(220, 180, 90);
const NUM: Color32 = Color32::from_rgb(230, 230, 240);

pub fn clamp_angle(spin: f32) -> f32 {
    if !spin.is_finite() {
        0.85
    } else {
        spin.rem_euclid(TAU)
    }
}

/// Składowe stałej strzałki na osiach obróconych o θ — wzór z lekcji 1.
pub fn axis_components(v: Vector, theta: f64) -> (f64, f64) {
    let c = theta.cos();
    let s = theta.sin();
    (v.x * c + v.y * s, -v.x * s + v.y * c)
}

pub fn length_xy(v: Vector) -> f64 {
    v.x.hypot(v.y)
}

/// Obrót w płaszczyźnie xy: maszyna (1,1) z lekcji 2.
pub fn rotation_xy(theta: f64) -> Tensor11 {
    let c = theta.cos();
    let s = theta.sin();
    let mut m = Tensor11::identity().components();
    m[1][1] = c;
    m[1][2] = -s;
    m[2][1] = s;
    m[2][2] = c;
    Tensor11::from_components(m)
}

pub fn probe_r(mass: f64, probe: f64) -> f64 {
    let m = if mass.is_finite() {
        mass.clamp(0.0, MASS_MAX)
    } else {
        MASS_DEFAULT
    };
    let floor = if m <= 1e-9 { 0.8 } else { 2.0 * m + 0.35 * m.max(0.2) };
    let raw = if probe.is_finite() { probe } else { 6.0 * m.max(1.0) };
    raw.clamp(floor, 16.0)
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    playback.spin = clamp_angle(playback.spin);
    playback.mass = if playback.mass.is_finite() {
        playback.mass.clamp(0.0, MASS_MAX)
    } else {
        MASS_DEFAULT
    };
    playback.probe = probe_r(playback.mass, playback.probe);
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        match lesson {
            1..=3 => {
                ui.add(
                    egui::Slider::new(&mut playback.spin, 0.0..=TAU)
                        .text("θ")
                        .fixed_decimals(2),
                );
                playback.spin = clamp_angle(playback.spin);
            }
            4 => {
                ui.add(
                    egui::Slider::new(&mut playback.mass, 0.0..=MASS_MAX)
                        .text("M")
                        .fixed_decimals(2),
                );
                ui.add(
                    egui::Slider::new(&mut playback.probe, 2.2..=14.0)
                        .text("r")
                        .fixed_decimals(2),
                );
                playback.probe = probe_r(playback.mass, playback.probe);
            }
            _ => {
                if playback.probe.abs() > 2.0 {
                    playback.probe = 0.80;
                }
                ui.add(
                    egui::Slider::new(&mut playback.probe, -1.6..=1.6)
                        .text("v^t")
                        .fixed_decimals(2),
                );
            }
        }
        ui.label(RichText::new(hud(lesson, playback)).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    match lesson {
        1 => paint_paper(ui, rect, f64::from(playback.spin), playback.t),
        2 => paint_machine(ui, rect, f64::from(playback.spin), playback.t),
        3 => paint_slots(ui, rect, f64::from(playback.spin), playback.t),
        4 => paint_metric(ui, rect, playback.mass, playback.probe, playback.t),
        _ => paint_raise_lower(ui, rect, playback.probe, playback.t),
    }
}

fn hud(lesson: u8, playback: &Playback) -> String {
    match lesson {
        1 => {
            let (vx, vy) = axis_components(WORLD, f64::from(playback.spin));
            format!("|v| = {:.3}   v' = ({:.2}, {:.2})", length_xy(WORLD), vx, vy)
        }
        2 => {
            let w = rotation_xy(f64::from(playback.spin)).apply(WORLD);
            format!("|w| = {:.3}  = |v|", length_xy(w))
        }
        3 => {
            let n = Tensor02::minkowski().on(WORLD, OTHER);
            format!("η(u, v) = {n:.3}")
        }
        4 => match Schwarzschild::new(playback.mass) {
            Ok(bh) => match (bh.g_tt(playback.probe), bh.g_rr(playback.probe)) {
                (Ok(tt), Ok(rr)) => format!("g_tt = {tt:.3}   g_rr = {rr:.3}"),
                _ => "r na horyzoncie".into(),
            },
            Err(_) => "M".into(),
        },
        _ => {
            let v = Vector::new(playback.probe, 0.80, 0.0, 0.0);
            let w = v.lower();
            format!("v^t = {:.2}   v_t = {:.2}", v.t, w.t)
        }
    }
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "liczba jedzie z kartką  ·  strzałka zostaje przy ścianie, składowe tańczą",
        2 => "automat (1,1)  ·  strzałka wjeżdża, obrócona wyjeżdża, długość stoi",
        3 => "dwa sloty  ·  η(u, v) jest liczbą, nie strzałką",
        4 => "ta sama skrzynka  ·  eta daleko, g_tt i g_rr puchną przy studni",
        _ => "przekładnia g  ·  góra = dokąd, dół = ile warte, czas zmienia znak",
    }
}

fn paint_paper(ui: &Ui, rect: Rect, theta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let Some(plot) = Plot::new(rect.shrink(28.0), 2.2) else {
        return;
    };
    let clip = ui.painter_at(plot.inner);
    paint_room_grid(&clip, &plot);
    let pulse = 0.92 + 0.08 * phase.sin();
    draw_paper(&painter, &plot, theta, pulse);
    let origin = plot.px(0.0, 0.0);
    arrow(&painter, origin, plot.px(WORLD.x, WORLD.y), ARROW);
    painter.text(
        plot.px(WORLD.x, WORLD.y) + Vec2::new(8.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        "ściana",
        FontId::proportional(11.0),
        ARROW,
    );
    let (vx, vy) = axis_components(WORLD, theta);
    painter.text(
        Pos2::new(plot.inner.left() + 8.0, plot.inner.top() + 8.0),
        egui::Align2::LEFT_TOP,
        format!("T = 20\nvx' = {vx:.2}\nvy' = {vy:.2}"),
        FontId::monospace(12.0),
        NUM,
    );
}

fn paint_machine(ui: &Ui, rect: Rect, theta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(20.0);
    if inner.width() < 40.0 {
        return;
    }
    let box_w = inner.width() * 0.28;
    let machine = Rect::from_center_size(inner.center(), Vec2::new(box_w, inner.height() * 0.55));
    painter.rect_stroke(machine, 4.0, Stroke::new(1.5, PAPER), egui::StrokeKind::Middle);
    let m = rotation_xy(theta).components();
    painter.text(
        machine.center(),
        egui::Align2::CENTER_CENTER,
        format!("[{:.2}  {:.2}]\n[{:.2}  {:.2}]", m[1][1], m[1][2], m[2][1], m[2][2]),
        FontId::monospace(13.0),
        GOLD,
    );
    let y = inner.center().y;
    let left = Pos2::new(inner.left() + 24.0, y);
    let right = Pos2::new(inner.right() - 24.0, y);
    let vin = Vec2::new(36.0, -18.0 * phase.sin());
    arrow(&painter, left, left + vin, ARROW);
    let w = rotation_xy(theta).apply(WORLD);
    let vout = Vec2::new(36.0 * w.x as f32 / 1.5, -36.0 * w.y as f32 / 1.5);
    arrow(&painter, right, right + vout, OTHER_C);
    painter.text(
        left + Vec2::new(0.0, 28.0),
        egui::Align2::CENTER_TOP,
        "v",
        FontId::proportional(12.0),
        ARROW,
    );
    painter.text(
        right + Vec2::new(0.0, 28.0),
        egui::Align2::CENTER_TOP,
        "w = A v",
        FontId::proportional(12.0),
        OTHER_C,
    );
}

fn paint_slots(ui: &Ui, rect: Rect, theta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let Some(plot) = Plot::new(rect.shrink(28.0), 2.4) else {
        return;
    };
    paint_room_grid(&ui.painter_at(plot.inner), &plot);
    let origin = plot.px(0.0, 0.0);
    let rot = rotation_xy(theta);
    let u = rot.apply(WORLD);
    let wobble = f64::from(phase.sin()) * 0.15;
    let v = rotation_xy(theta + wobble).apply(OTHER);
    arrow(&painter, origin, plot.px(u.x, u.y), ARROW);
    arrow(&painter, origin, plot.px(v.x, v.y), OTHER_C);
    let n = Tensor02::minkowski().on(u, v);
    let world_n = Tensor02::minkowski().on(WORLD, OTHER);
    painter.text(
        Pos2::new(plot.inner.center().x, plot.inner.top() + 10.0),
        egui::Align2::CENTER_TOP,
        format!("η(u, v) = {n:.3}   (stałe η(świat) = {world_n:.3})"),
        FontId::monospace(13.0),
        GOLD,
    );
}

fn paint_metric(ui: &Ui, rect: Rect, mass: f64, r_probe: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(24.0);
    if inner.width() < 40.0 {
        return;
    }
    let Ok(bh) = Schwarzschild::new(mass.clamp(0.0, MASS_MAX)) else {
        return;
    };
    let r_min = probe_r(mass, 0.0);
    let r_max = 16.0;
    let mut prev_tt = None;
    let mut prev_rr = None;
    let samples = 80;
    for i in 0..=samples {
        let r = r_min + (r_max - r_min) * f64::from(i) / f64::from(samples);
        let x = inner.left() + inner.width() * (r - r_min) as f32 / (r_max - r_min) as f32;
        if let (Ok(tt), Ok(rr)) = (bh.g_tt(r), bh.g_rr(r)) {
            let y_tt = map_g(inner, tt, -2.2, 0.2);
            let y_rr = map_g(inner, rr, -0.2, 6.0);
            let p_tt = Pos2::new(x, y_tt);
            let p_rr = Pos2::new(x, y_rr);
            if let Some(q) = prev_tt {
                painter.line_segment([q, p_tt], Stroke::new(2.0, GOLD));
            }
            if let Some(q) = prev_rr {
                painter.line_segment([q, p_rr], Stroke::new(2.0, ARROW));
            }
            prev_tt = Some(p_tt);
            prev_rr = Some(p_rr);
        }
    }
    let eta_y = map_g(inner, -1.0, -2.2, 0.2);
    painter.line_segment(
        [
            Pos2::new(inner.left(), eta_y),
            Pos2::new(inner.right(), eta_y),
        ],
        Stroke::new(1.0, PAPER),
    );
    let x = inner.left() + inner.width() * (r_probe - r_min) as f32 / (r_max - r_min) as f32;
    let glow = 5.0 + 2.0 * phase.sin();
    painter.line_segment(
        [
            Pos2::new(x, inner.top()),
            Pos2::new(x, inner.bottom()),
        ],
        Stroke::new(1.5, NUM),
    );
    painter.circle_filled(Pos2::new(x, eta_y), glow, NUM);
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 8.0),
        egui::Align2::LEFT_TOP,
        "g_tt  złoty   g_rr  błękit   η_tt = −1  fiolet",
        FontId::proportional(11.0),
        LABEL,
    );
}

fn paint_raise_lower(ui: &Ui, rect: Rect, vt: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(28.0);
    if inner.width() < 40.0 {
        return;
    }
    let v = Vector::new(vt, 0.80, 0.0, 0.0);
    let down = v.lower();
    let back = down.raise();
    let mid = inner.center();
    let left = Pos2::new(inner.left() + inner.width() * 0.22, mid.y);
    let right = Pos2::new(inner.right() - inner.width() * 0.22, mid.y);
    let gear = Rect::from_center_size(mid, Vec2::splat(72.0 + 6.0 * phase.sin()));
    painter.rect_stroke(gear, 8.0, Stroke::new(2.0, GOLD), egui::StrokeKind::Middle);
    painter.text(mid, egui::Align2::CENTER_CENTER, "η", FontId::proportional(22.0), GOLD);
    painter.text(
        left,
        egui::Align2::CENTER_CENTER,
        format!("v^μ\n({:.2}, {:.2}, 0, 0)", v.t, v.x),
        FontId::monospace(13.0),
        ARROW,
    );
    painter.text(
        right,
        egui::Align2::CENTER_CENTER,
        format!("v_μ\n({:.2}, {:.2}, 0, 0)", down.t, down.x),
        FontId::monospace(13.0),
        OTHER_C,
    );
    painter.text(
        Pos2::new(mid.x, inner.bottom() - 10.0),
        egui::Align2::CENTER_BOTTOM,
        format!("opuść, podnieś → v^t = {:.2}  (start {:.2})", back.t, v.t),
        FontId::monospace(12.0),
        NUM,
    );
}

fn map_g(inner: Rect, g: f64, lo: f64, hi: f64) -> f32 {
    let t = ((g - lo) / (hi - lo)).clamp(0.0, 1.0) as f32;
    inner.bottom() - t * inner.height()
}

fn draw_paper(painter: &egui::Painter, plot: &Plot, theta: f64, pulse: f32) {
    let c = theta.cos() as f32;
    let s = theta.sin() as f32;
    let axes = [(c, s, "x'"), (-s, c, "y'")];
    for (dx, dy, name) in axes {
        let tip = plot.px(f64::from(dx) * 1.7, f64::from(dy) * 1.7);
        arrow(painter, plot.px(0.0, 0.0), tip, PAPER);
        painter.text(
            tip + Vec2::new(4.0, 0.0),
            egui::Align2::LEFT_CENTER,
            name,
            FontId::proportional(11.0),
            PAPER,
        );
    }
    let badge = plot.px(0.55 * f64::from(c), 0.55 * f64::from(s));
    painter.circle_filled(badge, 16.0 * pulse, Color32::from_rgb(40, 28, 48));
    painter.text(
        badge,
        egui::Align2::CENTER_CENTER,
        "20",
        FontId::proportional(13.0),
        PAPER,
    );
}

fn paint_room_grid(painter: &egui::Painter, plot: &Plot) {
    for i in -2..=2 {
        let a = f64::from(i);
        painter.line_segment(
            [plot.px(a, -2.2), plot.px(a, 2.2)],
            Stroke::new(1.0, GRID),
        );
        painter.line_segment(
            [plot.px(-2.2, a), plot.px(2.2, a)],
            Stroke::new(1.0, GRID),
        );
    }
    painter.line_segment(
        [plot.px(-2.2, 0.0), plot.px(2.2, 0.0)],
        Stroke::new(1.5, AXIS),
    );
    painter.line_segment(
        [plot.px(0.0, -2.2), plot.px(0.0, 2.2)],
        Stroke::new(1.5, AXIS),
    );
}

fn arrow(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    painter.line_segment([from, to], Stroke::new(2.4, color));
    let v = to - from;
    let len = v.length();
    if len < 6.0 {
        return;
    }
    let dir = v / len;
    let n = Vec2::new(-dir.y, dir.x);
    let base = to - dir * 11.0;
    painter.add(Shape::convex_polygon(
        vec![to, base + n * 5.0, base - n * 5.0],
        color,
        Stroke::NONE,
    ));
}

struct Plot {
    inner: Rect,
    origin: Pos2,
    scale: f32,
}

impl Plot {
    fn new(inner: Rect, span: f64) -> Option<Self> {
        if inner.width() < 16.0 || inner.height() < 16.0 || span <= 0.0 {
            return None;
        }
        let side = inner.width().min(inner.height());
        let boxr = Rect::from_center_size(inner.center(), Vec2::splat(side));
        Some(Self {
            inner: boxr,
            origin: boxr.center(),
            scale: side * 0.5 / span as f32,
        })
    }

    fn px(&self, x: f64, y: f64) -> Pos2 {
        Pos2::new(
            self.origin.x + x as f32 * self.scale,
            self.origin.y - y as f32 * self.scale,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::gr::Tensor20;

    #[test]
    fn axis_rotation_keeps_the_arrow_length() {
        let len = length_xy(WORLD);
        for theta in [0.0, 0.4, 1.2, std::f64::consts::FRAC_PI_2] {
            let (vx, vy) = axis_components(WORLD, theta);
            assert!((vx.hypot(vy) - len).abs() < 1e-12, "θ={theta}");
        }
        let (vx, vy) = axis_components(WORLD, 0.0);
        assert!((vx - WORLD.x).abs() < 1e-15);
        assert!((vy - WORLD.y).abs() < 1e-15);
    }

    #[test]
    fn rotation_machine_preserves_xy_length() {
        let a = rotation_xy(0.7);
        let w = a.apply(WORLD);
        assert!((length_xy(w) - length_xy(WORLD)).abs() < 1e-12);
        assert!((a.compose(rotation_xy(-0.7)).apply(WORLD).x - WORLD.x).abs() < 1e-12);
    }

    #[test]
    fn minkowski_on_two_arrows_is_the_spatial_dot() {
        let n = Tensor02::minkowski().on(WORLD, OTHER);
        assert!((n - (WORLD.x * OTHER.x + WORLD.y * OTHER.y)).abs() < 1e-15);
        let rot = rotation_xy(0.9);
        let n_rot = Tensor02::minkowski().on(rot.apply(WORLD), rot.apply(OTHER));
        assert!((n_rot - n).abs() < 1e-12);
    }

    #[test]
    fn eta_inverse_is_identity() {
        let product = Tensor20::minkowski().contract_weight(Tensor02::minkowski());
        assert_eq!(product, Tensor11::identity());
    }

    #[test]
    fn raise_after_lower_returns_the_vector() {
        let v = Vector::new(0.6, 0.8, 0.0, 0.0);
        assert_eq!(v.lower().raise(), v);
        assert!((v.lower().t + v.t).abs() < 1e-15);
    }

    #[test]
    fn probe_stays_outside_the_horizon() {
        assert!(probe_r(1.0, 6.0) > 2.0);
        assert!(probe_r(1.0, 0.1) > 2.0);
        assert!(probe_r(0.0, 1.0) > 0.0);
    }

    #[test]
    fn tensor_lessons_paint_without_a_window() {
        for index in 1..=5 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.1,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
            });
            let _ = ctx.end_pass();
        }
    }
}
