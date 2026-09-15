//! Zegary, linijka i 4-wektor: lekcje STW 4–6 oraz drzwi do N-ciał.
//!
//! Liczby biorą się z [`bone_core::gr::lorentz`] (`dilated_time`, `contracted_length`,
//! `Event`). Ten plik tylko składa je w tarcze, kreskę i punkt `(ct, x)`. Lekcja 7
//! nie liczy nowej fizyki — otwiera laboratorium, które już istnieje.

use std::f32::consts::{PI, TAU};

use bone_core::gr::{boost_x, contracted_length, dilated_time, Event};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, StrokeKind, Ui, Vec2};

use super::minkowski::{self, gamma_of};
use super::BETA_MAX;
use crate::lesson::Playback;
use crate::screen::LabId;

/// Czas własny jednego tyknięcia podręcznikowego — Δτ = 1, Δt = γ.
pub const PROPER_TICK: f64 = 1.0;
/// Długość spoczynkowa linijki. Po kontrakcji `L = L₀ / γ`.
pub const ROD_REST: f64 = 1.6;
/// Wysokość linijki: kierunek prostopadły do ruchu, bez kontrakcji.
pub const ROD_HEIGHT: f64 = 0.45;
/// Drugie wydarzenie na lekcji 6: stały znacznik, żeby był interwał, nie jedna kropka.
pub fn marker() -> Event {
    Event::on_axis(0.4, 1.1)
}

const SPAN: f64 = 2.2;
const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(28, 34, 44);
const AXIS: Color32 = Color32::from_rgb(90, 100, 118);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const LIGHT: Color32 = Color32::from_rgb(220, 180, 90);
const ANNA: Color32 = Color32::from_rgb(120, 180, 220);
const BARTEK: Color32 = Color32::from_rgb(160, 200, 140);
const EVENT: Color32 = Color32::from_rgb(230, 230, 240);

/// Δt = γ Δτ po przycięciu suwaka. Panika tylko gdy `BETA_MAX` przestanie być `< 1`.
pub fn lab_time(proper: f64, beta: f64) -> f64 {
    dilated_time(proper, minkowski::clamp_beta(beta)).expect("suwak trzyma |β| < 1")
}

/// Δτ = Δt / γ — odczyt zegara Bartka, gdy Anna odmierzyła `lab`.
pub fn proper_of(lab: f64, beta: f64) -> f64 {
    lab / gamma_of(beta)
}

/// L = L₀ / γ wzdłuż lotu.
pub fn rod_lab_length(beta: f64) -> f64 {
    contracted_length(ROD_REST, minkowski::clamp_beta(beta)).expect("suwak trzyma |β| < 1")
}

/// Wydarzenie na światolinii Bartka: w jego układzie `x' = 0`, czas własny `τ`.
///
/// `s² = τ²`: to tyknięcie zegara, nie linijka i nie światło.
pub fn bartek_event(proper: f64, beta: f64) -> Event {
    let b = minkowski::clamp_beta(beta);
    let ct = lab_time(proper, b);
    Event::on_axis(ct, b * ct)
}

/// Końce linijki w jednej chwili Anny: ten sam `ct`, Δx = L.
pub fn rod_ends_anna_now(ct: f64, beta: f64) -> (Event, Event) {
    let half = rod_lab_length(beta) * 0.5;
    (
        Event::on_axis(ct, -half),
        Event::on_axis(ct, half),
    )
}

pub fn draw(ui: &mut Ui, lesson: u8, playback: &mut Playback) {
    ui.horizontal(|ui| {
        let label = if playback.playing { "Pauza" } else { "Play" };
        if ui.button(label).clicked() {
            playback.playing = !playback.playing;
        }
        ui.add(
            egui::Slider::new(&mut playback.beta, 0.0..=BETA_MAX)
                .text("β")
                .fixed_decimals(2),
        );
        playback.beta = playback.beta.clamp(0.0, BETA_MAX);
        ui.label(RichText::new(format!("γ = {:.3}", gamma_of(playback.beta))).monospace());
    });
    ui.label(RichText::new(caption(lesson)).small().weak());
    ui.add_space(4.0);

    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    match lesson {
        4 => paint_clocks(ui, rect, playback.beta, playback.t),
        5 => paint_rod(ui, rect, playback.beta, playback.t),
        _ => paint_four_vector(ui, rect, playback.beta, playback.t),
    }
}

/// Drzwi z lekcji 7: bez nowej fizyki, z jawnym kłamstwem N-ciał.
pub fn draw_nbody_door(ui: &mut Ui) -> bool {
    let mut open = false;
    ui.add_space(12.0);
    ui.label(
        RichText::new("Drzwi do chmury, którą Bone już umie liczyć")
            .size(16.0)
            .strong(),
    );
    ui.label(
        RichText::new(
            "Cząstki dostają kinematykę Einsteina. Siła zostaje wzorem Newtona:\n\
             natychmiastowa, 1/r², ze zmiękczeniem. Nie ma zakrzywionej sceny.",
        )
        .small()
        .weak(),
    );
    ui.add_space(10.0);
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(28, 34, 22))
        .stroke(Stroke::new(1.0, Color32::from_rgb(160, 200, 140)))
        .inner_margin(10.0)
        .show(ui, |ui| {
            ui.label(
                RichText::new(crate::simulation::NBODY_NOT_GR)
                    .size(15.0)
                    .italics()
                    .color(Color32::from_rgb(200, 230, 170)),
            );
        });
    ui.add_space(12.0);
    ui.horizontal(|ui| {
        formula_tile(ui, "Newton", "v = p / m", ANNA);
        formula_tile(ui, "SR", "v = p c² / E", BARTEK);
    });
    ui.add_space(8.0);
    ui.label(
        RichText::new("Przełącznik w laboratorium zostaje. Grawitacja w obu trybach ta sama.")
            .small()
            .weak(),
    );
    ui.add_space(16.0);
    if ui
        .add_sized(
            [280.0, 36.0],
            egui::Button::new("Otwórz laboratorium N-ciała"),
        )
        .clicked()
    {
        open = true;
    }
    ui.add_space(6.0);
    ui.label(
        RichText::new(format!("wejście → {} · preset relatywistyczny", LabId::Nbody.label()))
            .small()
            .weak()
            .monospace(),
    );
    open
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        4 => "dwa zegary · Anna tyka z planszą · Bartek zwalnia z γ",
        5 => "linijka wzdłuż x · wysokość bez zmian · końce jednoczesne dla Anny",
        _ => "punkt (ct, x) · to wydarzenie, nie piłka · s² nie zależy od β",
    }
}

fn formula_tile(ui: &mut Ui, title: &str, body: &str, color: Color32) {
    ui.group(|ui| {
        ui.set_width(160.0);
        ui.label(RichText::new(title).strong().color(color));
        ui.label(RichText::new(body).monospace());
    });
}

fn paint_clocks(ui: &Ui, rect: Rect, beta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let g = gamma_of(beta);
    let lab = f64::from(phase.rem_euclid(TAU) / TAU) * 4.0 * PROPER_TICK;
    let proper = proper_of(lab, beta);
    let split = rect.center().y;
    let top = Rect::from_min_max(rect.min, Pos2::new(rect.right(), split - 4.0));
    let bot = Rect::from_min_max(Pos2::new(rect.left(), split + 4.0), rect.max);

    let faces = top.shrink(16.0);
    if faces.width() > 40.0 && faces.height() > 40.0 {
        let r = faces.height().min(faces.width() * 0.22).clamp(28.0, 70.0);
        let cy = faces.center().y;
        let gap = faces.width() * 0.28;
        analog(
            &painter,
            Pos2::new(faces.center().x - gap, cy),
            r,
            lab,
            ANNA,
            "Anna",
            format!("Δt = {lab:.2}"),
        );
        analog(
            &painter,
            Pos2::new(faces.center().x + gap, cy),
            r,
            proper,
            BARTEK,
            "Bartek",
            format!("Δτ = {proper:.2}"),
        );
    }

    paint_light_clocks(&painter, bot, beta, proper);
    painter.text(
        Pos2::new(rect.left() + 10.0, rect.bottom() - 8.0),
        egui::Align2::LEFT_BOTTOM,
        format!("Δt = γ Δτ    γ = {g:.3}    β = {beta:.2}"),
        FontId::monospace(12.0),
        LABEL,
    );
}

fn analog(
    painter: &egui::Painter,
    c: Pos2,
    r: f32,
    time: f64,
    color: Color32,
    name: &str,
    readout: String,
) {
    painter.circle_stroke(c, r, Stroke::new(2.0, color));
    for i in 0..12 {
        let a = i as f32 / 12.0 * TAU - PI / 2.0;
        let inner = r * 0.82;
        let outer = r * 0.94;
        let dir = Vec2::new(a.cos(), a.sin());
        painter.line_segment(
            [c + dir * inner, c + dir * outer],
            Stroke::new(1.0, AXIS),
        );
    }
    let turns = (time / PROPER_TICK) as f32;
    let ang = turns * TAU - PI / 2.0;
    let hand = Vec2::new(ang.cos(), ang.sin());
    painter.line_segment(
        [c, c + hand * (r * 0.72)],
        Stroke::new(2.4, color),
    );
    painter.circle_filled(c, 3.0, color);
    painter.text(
        c + Vec2::new(0.0, r + 4.0),
        egui::Align2::CENTER_TOP,
        name,
        FontId::proportional(12.0),
        color,
    );
    painter.text(
        c + Vec2::new(0.0, r + 18.0),
        egui::Align2::CENTER_TOP,
        readout,
        FontId::monospace(11.0),
        LABEL,
    );
}

fn paint_light_clocks(painter: &egui::Painter, rect: Rect, beta: f64, proper: f64) {
    let inner = rect.shrink2(Vec2::new(24.0, 28.0));
    if inner.height() < 40.0 || inner.width() < 80.0 {
        return;
    }
    let mid = inner.center().x;
    let h = inner.height() * 0.72;
    let floor = inner.bottom() - 8.0;
    let ceil = floor - h;
    let anna_x = inner.left() + inner.width() * 0.22;
    let bartek_x = mid + inner.width() * 0.12;
    let period = 2.0 * f64::from(h);
    let s = (proper * f64::from(h) / PROPER_TICK).rem_euclid(period);
    let y_rest = if s < f64::from(h) {
        s as f32
    } else {
        2.0 * h - s as f32
    };

    painter.line_segment(
        [Pos2::new(anna_x, floor), Pos2::new(anna_x, ceil)],
        Stroke::new(1.5, ANNA),
    );
    painter.circle_filled(Pos2::new(anna_x, floor - y_rest), 4.5, LIGHT);
    painter.text(
        Pos2::new(anna_x, inner.top()),
        egui::Align2::CENTER_TOP,
        "zegar Anny: pion",
        FontId::proportional(10.0),
        ANNA,
    );

    let zig = (minkowski::clamp_beta(beta) as f32) * h * 0.55;
    let going_up = s < f64::from(h);
    let t = if going_up {
        (s as f32) / h
    } else {
        (s as f32 - h) / h
    };
    let (left, right) = (bartek_x, bartek_x + zig);
    painter.line_segment(
        [Pos2::new(left, floor), Pos2::new(right, ceil)],
        Stroke::new(1.2, BARTEK),
    );
    painter.line_segment(
        [Pos2::new(right, ceil), Pos2::new(left + zig * 2.0, floor)],
        Stroke::new(1.2, BARTEK),
    );
    let photon = if going_up {
        Pos2::new(left + zig * t, floor - h * t)
    } else {
        Pos2::new(right + zig * t, ceil + h * t)
    };
    painter.circle_filled(photon, 4.5, LIGHT);
    painter.text(
        Pos2::new(bartek_x + zig, inner.top()),
        egui::Align2::CENTER_TOP,
        "ten sam foton z peronu: zygzak",
        FontId::proportional(10.0),
        BARTEK,
    );
}

fn paint_rod(ui: &Ui, rect: Rect, beta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let split = rect.top() + rect.height() * 0.38;
    let wagon = Rect::from_min_max(rect.min, Pos2::new(rect.right(), split));
    let diagram = Rect::from_min_max(Pos2::new(rect.left(), split), rect.max);
    paint_wagon(&painter, wagon, beta);
    paint_rod_diagram(&painter, diagram, beta, phase);
}

fn paint_wagon(painter: &egui::Painter, rect: Rect, beta: f64) {
    let inner = rect.shrink2(Vec2::new(28.0, 18.0));
    if inner.width() < 40.0 {
        return;
    }
    let l0 = inner.width() * 0.72;
    let l = l0 / gamma_of(beta) as f32;
    let h = inner.height() * 0.42;
    let rest = Rect::from_center_size(inner.center(), Vec2::new(l0, h));
    let moving = Rect::from_center_size(inner.center(), Vec2::new(l, h));
    painter.rect_stroke(rest, 3.0, Stroke::new(1.0, AXIS), StrokeKind::Middle);
    painter.rect_filled(moving, 3.0, Color32::from_rgba_unmultiplied(160, 200, 140, 70));
    painter.rect_stroke(moving, 3.0, Stroke::new(2.0, BARTEK), StrokeKind::Middle);
    let y = moving.center().y;
    painter.circle_filled(Pos2::new(moving.left(), y), 4.0, ANNA);
    painter.circle_filled(Pos2::new(moving.right(), y), 4.0, ANNA);
    let g = gamma_of(beta);
    painter.text(
        Pos2::new(inner.left(), inner.top()),
        egui::Align2::LEFT_TOP,
        format!(
            "L₀ = {ROD_REST:.2}    L = {l:.2}    wysokość = {h_rest:.2} (bez zmian)",
            l = rod_lab_length(beta),
            h_rest = ROD_HEIGHT
        ),
        FontId::monospace(11.0),
        LABEL,
    );
    painter.text(
        Pos2::new(inner.right(), inner.bottom()),
        egui::Align2::RIGHT_BOTTOM,
        format!("L = L₀ / γ    γ = {g:.3}"),
        FontId::monospace(11.0),
        LABEL,
    );
}

fn paint_rod_diagram(painter: &egui::Painter, rect: Rect, beta: f64, phase: f32) {
    let inner = square_in(rect, 18.0);
    let Some(plot) = Plot::new(inner, SPAN) else {
        return;
    };
    paint_grid(painter, &plot);
    paint_axes(painter, &plot);
    let axes = minkowski::primed_axes(beta);
    painter.line_segment(
        plot.seg(
            -SPAN * axes.ct_ct,
            -SPAN * axes.ct_x,
            SPAN * axes.ct_ct,
            SPAN * axes.ct_x,
        ),
        Stroke::new(1.4, BARTEK),
    );
    painter.line_segment(
        plot.seg(
            -SPAN * axes.x_ct,
            -SPAN * axes.x_x,
            SPAN * axes.x_ct,
            SPAN * axes.x_x,
        ),
        Stroke::new(1.4, BARTEK),
    );
    let ct = minkowski::phase_time(phase, SPAN * 0.45);
    let (left, right) = rod_ends_anna_now(ct, beta);
    dashed(
        painter,
        plot.seg(ct, -SPAN, ct, SPAN),
        Stroke::new(1.0, ANNA),
    );
    painter.line_segment(
        plot.seg(left.ct, left.x, right.ct, right.x),
        Stroke::new(3.0, BARTEK),
    );
    painter.circle_filled(plot.px(left.ct, left.x), 4.0, ANNA);
    painter.circle_filled(plot.px(right.ct, right.x), 4.0, ANNA);
    painter.text(
        plot.px(ct, 0.0) + Vec2::new(8.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        "teraz Anny",
        FontId::proportional(10.0),
        ANNA,
    );
}

fn paint_four_vector(ui: &Ui, rect: Rect, beta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = square_in(rect, 24.0);
    let Some(plot) = Plot::new(inner, SPAN) else {
        return;
    };
    paint_grid(&painter, &plot);
    paint_axes(&painter, &plot);
    painter.line_segment(plot.seg(-SPAN, -SPAN, SPAN, SPAN), Stroke::new(1.2, LIGHT));
    painter.line_segment(plot.seg(-SPAN, SPAN, SPAN, -SPAN), Stroke::new(1.2, LIGHT));

    let tau = minkowski::phase_time(phase, 1.4);
    let p = bartek_event(tau, beta);
    let q = marker();
    painter.line_segment(plot.seg(0.0, 0.0, SPAN, beta * SPAN), Stroke::new(1.6, BARTEK));
    painter.line_segment(
        plot.seg(p.ct, p.x, q.ct, q.x),
        Stroke::new(1.2, Color32::from_rgb(180, 140, 150)),
    );
    painter.circle_filled(plot.px(0.0, 0.0), 3.5, AXIS);
    painter.circle_filled(plot.px(p.ct, p.x), 5.5, EVENT);
    painter.circle_filled(plot.px(q.ct, q.x), 4.5, LIGHT);
    painter.text(
        plot.px(p.ct, p.x) + Vec2::new(8.0, -4.0),
        egui::Align2::LEFT_BOTTOM,
        format!("(ct, x) = ({:.2}, {:.2})", p.ct, p.x),
        FontId::monospace(12.0),
        EVENT,
    );
    painter.text(
        plot.px(q.ct, q.x) + Vec2::new(6.0, 10.0),
        egui::Align2::LEFT_TOP,
        "Q",
        FontId::monospace(12.0),
        LIGHT,
    );
    painter.text(
        plot.px(p.ct, p.x) + Vec2::new(8.0, 12.0),
        egui::Align2::LEFT_TOP,
        "to nie piłka — to wydarzenie",
        FontId::proportional(11.0),
        BARTEK,
    );

    let s2 = p.interval_sq_to(q);
    let primed = boost_x(p, minkowski::clamp_beta(beta));
    let q_p = boost_x(q, minkowski::clamp_beta(beta));
    let s2_p = match (primed, q_p) {
        (Ok(a), Ok(b)) => a.interval_sq_to(b),
        _ => s2,
    };
    painter.text(
        plot.px(-SPAN, -SPAN) + Vec2::new(4.0, 16.0),
        egui::Align2::LEFT_TOP,
        format!("s² = (ct)² − x² = {s2:.3}    po boostcie s² = {s2_p:.3}"),
        FontId::monospace(11.0),
        LABEL,
    );
}

fn square_in(rect: Rect, pad: f32) -> Rect {
    let inner = rect.shrink(pad);
    let side = inner.width().min(inner.height());
    Rect::from_center_size(inner.center(), Vec2::splat(side))
}

struct Plot {
    origin: Pos2,
    scale: f32,
}

impl Plot {
    fn new(inner: Rect, span: f64) -> Option<Self> {
        if inner.width() < 8.0 || span <= 0.0 {
            return None;
        }
        Some(Self {
            origin: inner.center(),
            scale: inner.width() * 0.5 / span as f32,
        })
    }

    fn px(&self, ct: f64, x: f64) -> Pos2 {
        Pos2::new(
            self.origin.x + x as f32 * self.scale,
            self.origin.y - ct as f32 * self.scale,
        )
    }

    fn seg(&self, ct0: f64, x0: f64, ct1: f64, x1: f64) -> [Pos2; 2] {
        [self.px(ct0, x0), self.px(ct1, x1)]
    }
}

fn paint_grid(painter: &egui::Painter, plot: &Plot) {
    let stroke = Stroke::new(1.0, GRID);
    let n = SPAN.floor() as i32;
    for i in -n..=n {
        if i == 0 {
            continue;
        }
        let v = f64::from(i);
        painter.line_segment(plot.seg(-SPAN, v, SPAN, v), stroke);
        painter.line_segment(plot.seg(v, -SPAN, v, SPAN), stroke);
    }
}

fn paint_axes(painter: &egui::Painter, plot: &Plot) {
    let stroke = Stroke::new(1.4, AXIS);
    painter.line_segment(plot.seg(0.0, -SPAN, 0.0, SPAN), stroke);
    painter.line_segment(plot.seg(-SPAN, 0.0, SPAN, 0.0), stroke);
    painter.text(
        plot.px(0.0, SPAN) + Vec2::new(-4.0, -2.0),
        egui::Align2::RIGHT_BOTTOM,
        "x",
        FontId::monospace(12.0),
        LABEL,
    );
    painter.text(
        plot.px(SPAN, 0.0) + Vec2::new(6.0, 2.0),
        egui::Align2::LEFT_TOP,
        "ct",
        FontId::monospace(12.0),
        LABEL,
    );
}

fn dashed(painter: &egui::Painter, ends: [Pos2; 2], stroke: Stroke) {
    let d = ends[1] - ends[0];
    let len = d.length();
    if len < 1.0 {
        return;
    }
    let u = d / len;
    let mut s = 0.0;
    while s < len {
        let e = (s + 6.0).min(len);
        painter.line_segment([ends[0] + u * s, ends[0] + u * e], stroke);
        s += 10.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::BETA_DEFAULT;
    use bone_core::gr::{contract_rod, gamma};
    use bone_core::vec3::vec3;

    #[test]
    fn dilation_and_contraction_match_lorentz_at_the_textbook_beta() {
        let beta = BETA_DEFAULT;
        assert!((lab_time(PROPER_TICK, beta) - 1.25).abs() < 1e-15);
        assert!((lab_time(PROPER_TICK, beta) / PROPER_TICK - gamma_of(beta)).abs() < 1e-15);
        assert!((rod_lab_length(beta) * gamma_of(beta) - ROD_REST).abs() < 1e-15);
        assert!(rod_lab_length(beta) < ROD_REST);
        assert!((lab_time(1.0, 0.0) - 1.0).abs() < 1e-15);
        assert!((rod_lab_length(0.0) - ROD_REST).abs() < 1e-15);
        assert_eq!(gamma_of(beta), gamma(beta).expect("0.6 jest poniżej c"));
    }

    #[test]
    fn rod_shrinks_only_along_the_boost() {
        let proper = vec3(ROD_REST, ROD_HEIGHT, 0.2);
        let moved = contract_rod(proper, vec3(BETA_DEFAULT, 0.0, 0.0)).unwrap();
        assert!((moved.x - rod_lab_length(BETA_DEFAULT)).abs() < 1e-15);
        assert!((moved.y - ROD_HEIGHT).abs() < 1e-15);
        assert!((moved.z - 0.2).abs() < 1e-15);
    }

    #[test]
    fn bartek_worldline_event_is_a_clock_tick() {
        let tau = 0.8;
        let p = bartek_event(tau, BETA_DEFAULT);
        assert!((p.interval_sq() - tau * tau).abs() < 1e-12);
        assert!((p.x - BETA_DEFAULT * p.ct).abs() < 1e-12);
        let rest = bartek_event(tau, 0.0);
        assert!((rest.ct - tau).abs() < 1e-15 && rest.x.abs() < 1e-15);
    }

    #[test]
    fn anna_photographs_the_rod_at_one_ct() {
        let (left, right) = rod_ends_anna_now(0.3, BETA_DEFAULT);
        assert!((left.ct - right.ct).abs() < 1e-15);
        assert!((right.x - left.x - rod_lab_length(BETA_DEFAULT)).abs() < 1e-15);
        let left_p = boost_x(left, BETA_DEFAULT).unwrap();
        let right_p = boost_x(right, BETA_DEFAULT).unwrap();
        assert!(
            (left_p.ct - right_p.ct).abs() > 0.4,
            "końce miały stracić wspólną chwilę Bartka"
        );
    }

    #[test]
    fn four_vector_interval_survives_the_boost() {
        let p = bartek_event(0.7, BETA_DEFAULT);
        let s2 = p.interval_sq_to(marker());
        let p2 = boost_x(p, BETA_DEFAULT).unwrap();
        let q2 = boost_x(marker(), BETA_DEFAULT).unwrap();
        assert!((p2.interval_sq_to(q2) - s2).abs() < 1e-12);
        assert_ne!(format!("({:.2}, {:.2})", p.ct, p.x), "(0.00, 0.00)");
    }

    #[test]
    fn stw_lessons_4_to_6_paint_without_a_window() {
        for index in 4..=6 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.2,
                    beta: BETA_DEFAULT,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
                assert!((gamma_of(playback.beta) - 1.25).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }

    #[test]
    fn nbody_door_paints_without_opening() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            assert!(!draw_nbody_door(ui));
        });
        let _ = ctx.end_pass();
    }
}
