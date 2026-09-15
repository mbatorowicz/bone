//! Film RK4 kontra Euler: to samo koło harmoniczne, dwa kroki, liczba błędu.
//!
//! Prawa strona `y' = (v, −x)` jest w [`bone_core::gr::rk4`]. Euler stoi
//! tylko tu — silnik go nie eksportuje, bo kurs ma pokazać, czemu go nie ma
//! w geodezyjnej. Play przesuwa haust; suwak `h` psuje albo ratuje spiralę.

use std::f32::consts::TAU;

use bone_core::gr::rk4::{self, Scratch};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui};

use crate::lesson::Playback;

/// Domyślny haust: dość duży, żeby Euler wypadł z koła, a RK4 zostało.
pub const H_DEFAULT: f64 = 0.35;
pub const H_MIN: f64 = 0.08;
pub const H_MAX: f64 = 0.55;
/// Długość filmu: dwa okrążenia dokładnego `x = cos t`.
pub const T_END: f64 = 4.0 * std::f64::consts::PI;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const AXIS: Color32 = Color32::from_rgb(90, 100, 118);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const EXACT: Color32 = Color32::from_rgb(70, 82, 100);
const RK: Color32 = Color32::from_rgb(120, 180, 220);
const EU: Color32 = Color32::from_rgb(220, 150, 110);

pub fn clamp_h(h: f64) -> f64 {
    if !h.is_finite() {
        H_DEFAULT
    } else {
        h.clamp(H_MIN, H_MAX)
    }
}

pub fn harmonic(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = y[1];
    dy[1] = -y[0];
}

/// Jeden krok Eulera na `y' = (v, −x)`.
pub fn euler_step(y: &mut [f64; 2], h: f64) {
    let x = y[0];
    let v = y[1];
    y[0] = x + h * v;
    y[1] = v + h * (-x);
}

pub fn integrate_euler(h: f64, n: usize) -> [f64; 2] {
    let mut y = [1.0, 0.0];
    for _ in 0..n {
        euler_step(&mut y, h);
    }
    y
}

pub fn integrate_rk4(h: f64, n: usize) -> [f64; 2] {
    let y = rk4::integrate(0.0, &[1.0, 0.0], h, n, harmonic);
    [y[0], y[1]]
}

pub fn radius(y: [f64; 2]) -> f64 {
    (y[0] * y[0] + y[1] * y[1]).sqrt()
}

pub fn radius_error(y: [f64; 2]) -> f64 {
    (radius(y) - 1.0).abs()
}

pub fn steps_for(h: f64, t: f64) -> usize {
    let h = clamp_h(h);
    if h <= 0.0 || t <= 0.0 {
        0
    } else {
        (t / h).floor() as usize
    }
}

pub fn film_time(phase: f32) -> f64 {
    f64::from(phase.rem_euclid(TAU) / TAU) * T_END
}

fn trail(h: f64, n: usize, rk: bool) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(n + 1);
    if rk {
        let mut y = vec![1.0, 0.0];
        let mut scratch = Scratch::with_len(2);
        let mut t = 0.0;
        out.push([y[0], y[1]]);
        for _ in 0..n {
            rk4::step(t, &mut y, h, &mut scratch, harmonic);
            t += h;
            out.push([y[0], y[1]]);
        }
    } else {
        let mut y = [1.0, 0.0];
        out.push(y);
        for _ in 0..n {
            euler_step(&mut y, h);
            out.push(y);
        }
    }
    out
}

pub fn draw(ui: &mut Ui, playback: &mut Playback) {
    let id = ui.id().with("geo-rk4-h");
    let mut h = ui.ctx().data(|d| d.get_temp::<f64>(id)).unwrap_or(H_DEFAULT);
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut h, H_MIN..=H_MAX)
                .text("h")
                .fixed_decimals(2),
        );
        h = clamp_h(h);
        let n = steps_for(h, film_time(playback.t));
        let eu = integrate_euler(h, n);
        let rk = integrate_rk4(h, n);
        ui.label(
            RichText::new(format!(
                "|Δr|_Euler = {:.3}   |Δr|_RK4 = {:.4}",
                radius_error(eu),
                radius_error(rk)
            ))
            .monospace(),
        );
    });
    ui.ctx().data_mut(|d| d.insert_temp(id, h));
    ui.label(
        RichText::new("to samo y' = (v, −x)  ·  Euler jeden raz, RK4 cztery k-i  ·  koło zdradza błąd")
            .small()
            .weak(),
    );
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    paint(ui, rect, h, playback.t);
}

fn paint(ui: &Ui, rect: Rect, h: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = rect.shrink(18.0);
    if inner.width() < 16.0 || inner.height() < 16.0 {
        return;
    }
    let c = inner.center();
    let span = 2.4_f32;
    let scale = inner.width().min(inner.height()) * 0.42 / span;
    let map = |x: f64, y: f64| {
        Pos2::new(
            c.x + scale * x as f32,
            c.y - scale * y as f32,
        )
    };
    painter.line_segment(
        [Pos2::new(inner.left(), c.y), Pos2::new(inner.right(), c.y)],
        Stroke::new(1.0, AXIS),
    );
    painter.line_segment(
        [Pos2::new(c.x, inner.top()), Pos2::new(c.x, inner.bottom())],
        Stroke::new(1.0, AXIS),
    );
    painter.circle_stroke(c, scale, Stroke::new(1.0, EXACT));
    painter.text(
        Pos2::new(inner.right() - 4.0, c.y - 4.0),
        egui::Align2::RIGHT_BOTTOM,
        "x",
        FontId::monospace(11.0),
        LABEL,
    );
    painter.text(
        Pos2::new(c.x + 6.0, inner.top() + 2.0),
        egui::Align2::LEFT_TOP,
        "v",
        FontId::monospace(11.0),
        LABEL,
    );

    let n = steps_for(h, film_time(phase));
    let eu = trail(h, n, false);
    let rk = trail(h, n, true);
    stroke_trail(&painter, &map, &eu, EU);
    stroke_trail(&painter, &map, &rk, RK);
    if let Some(&last) = eu.last() {
        painter.circle_filled(map(last[0], last[1]), 5.5, EU);
    }
    if let Some(&last) = rk.last() {
        painter.circle_filled(map(last[0], last[1]), 5.5, RK);
    }

    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 10.0),
        egui::Align2::LEFT_TOP,
        "RK4",
        FontId::proportional(12.0),
        RK,
    );
    painter.text(
        Pos2::new(inner.left() + 8.0, inner.top() + 26.0),
        egui::Align2::LEFT_TOP,
        "Euler",
        FontId::proportional(12.0),
        EU,
    );
}

fn stroke_trail(
    painter: &egui::Painter,
    map: &dyn Fn(f64, f64) -> Pos2,
    pts: &[[f64; 2]],
    color: Color32,
) {
    for pair in pts.windows(2) {
        let a = map(pair[0][0], pair[0][1]);
        let b = map(pair[1][0], pair[1][1]);
        painter.line_segment([a, b], Stroke::new(1.8, color));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h_slider_stays_in_range() {
        assert_eq!(clamp_h(H_DEFAULT), H_DEFAULT);
        assert_eq!(clamp_h(0.0), H_MIN);
        assert_eq!(clamp_h(4.0), H_MAX);
        assert_eq!(clamp_h(f64::NAN), H_DEFAULT);
    }

    #[test]
    fn rk4_matches_the_core_stepper() {
        let h = 0.1;
        let n = 10;
        let from_viz = integrate_rk4(h, n);
        let from_core = rk4::integrate(0.0, &[1.0, 0.0], h, n, harmonic);
        assert!((from_viz[0] - from_core[0]).abs() < 1e-15);
        assert!((from_viz[1] - from_core[1]).abs() < 1e-15);
    }

    #[test]
    fn euler_spirals_out_and_rk4_stays_closer() {
        let h = 0.3;
        let n = 40;
        let eu = integrate_euler(h, n);
        let rk = integrate_rk4(h, n);
        assert!(
            radius(eu) > 1.05,
            "Euler miał wyjść z koła, r = {}",
            radius(eu)
        );
        assert!(
            radius_error(rk) < radius_error(eu),
            "RK4 |Δr|={} Euler |Δr|={}",
            radius_error(rk),
            radius_error(eu)
        );
    }

    #[test]
    fn textbook_exp_stepper_still_beats_euler() {
        let h = 0.1;
        let n = 10;
        let rk = rk4::integrate(0.0, &[1.0], h, n, |_t, y, dy| dy[0] = y[0])[0];
        let mut eu = 1.0;
        for _ in 0..n {
            eu += h * eu;
        }
        let want = std::f64::consts::E;
        assert!((rk - want).abs() < (eu - want).abs());
    }

    #[test]
    fn film_time_covers_two_turns() {
        assert!((film_time(0.0)).abs() < 1e-15);
        assert!((film_time(TAU) - 0.0).abs() < 1e-5 || (film_time(TAU) - T_END).abs() < 1e-5);
        assert!((film_time(std::f32::consts::PI) - 0.5 * T_END).abs() < 1e-5);
        assert_eq!(steps_for(0.2, 0.0), 0);
        assert_eq!(steps_for(0.2, 1.0), 5);
    }

    #[test]
    fn geo_lesson_5_paints_without_a_window() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            let mut playback = Playback {
                playing: false,
                t: 1.2,
                ..Default::default()
            };
            draw(ui, &mut playback);
        });
        let _ = ctx.end_pass();
    }
}
