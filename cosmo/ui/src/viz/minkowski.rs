//! Diagram ct–x: światło pod 45°, światolinie, suwak β przechyla osie.
//!
//! Współrzędne i boost biorą się z [`bone_core::gr::lorentz`]. UI tylko
//! rysuje i przycina β, żeby suwak nigdy nie wjechał w światło — silnik
//! nadal odrzuca `|β| ≥ 1`, gdy ktoś wywoła go wprost.

use std::f32::consts::TAU;

use bone_core::gr::{boost_x, gamma, Event, Superluminal};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, StrokeKind, Ui, Vec2};

use super::BETA_MAX;
use crate::lesson::Playback;

/// Pół długości peronu: pioruny w `x = ±1`, Anna w środku.
pub const LIGHTNING_HALF: f64 = 1.0;
/// Zasięg osi na diagramie. Przy β = 0.6 widać oba spotkania Bartka.
const SPAN: f64 = 2.6;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const GRID: Color32 = Color32::from_rgb(28, 34, 44);
const AXIS: Color32 = Color32::from_rgb(90, 100, 118);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const LIGHT: Color32 = Color32::from_rgb(220, 180, 90);
const ANNA: Color32 = Color32::from_rgb(120, 180, 220);
const BARTEK: Color32 = Color32::from_rgb(160, 200, 140);
const FLASH: Color32 = Color32::from_rgb(240, 220, 140);

/// β z suwaka: zawsze legalny boost.
pub fn clamp_beta(beta: f64) -> f64 {
    beta.clamp(0.0, BETA_MAX)
}

/// `γ` po przycięciu suwaka. Panika tylko gdy `BETA_MAX` przestanie być `< 1`.
pub fn gamma_of(beta: f64) -> f64 {
    gamma(clamp_beta(beta)).expect("suwak trzyma |β| < 1")
}

/// Kierunki osi primowanych w płaszczyźnie laboratoryjnej `(x, ct)`.
///
/// Oś `ct'`: światolinia początku układu Bartka, `x = β ct`.
/// Oś `x'`: jego „teraz”, `ct = β x`. Światło (`x = ct`) zostaje.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrimedAxes {
    pub ct_x: f64,
    pub ct_ct: f64,
    pub x_x: f64,
    pub x_ct: f64,
}

pub fn primed_axes(beta: f64) -> PrimedAxes {
    let b = clamp_beta(beta);
    PrimedAxes {
        ct_x: b,
        ct_ct: 1.0,
        x_x: 1.0,
        x_ct: b,
    }
}

/// Kąt między osią `ct'` a linią światła `x = ct`, w radianach.
///
/// Maleje, gdy rośnie β: osie Bartka przytulają się do stożka, światło stoi.
pub fn tilt_from_light(beta: f64) -> f64 {
    std::f64::consts::FRAC_PI_4 - clamp_beta(beta).atan()
}

/// Dwa pioruny na peronie Anny: ten sam `ct`, różne `x`.
pub fn lightning_strikes(half: f64) -> (Event, Event) {
    (Event::on_axis(0.0, -half), Event::on_axis(0.0, half))
}

/// Scena Einsteina: pioruny, spotkanie u Anny, dwa spotkania u Bartka.
#[derive(Clone, Copy, Debug)]
pub struct LightningScene {
    pub left: Event,
    pub right: Event,
    pub anna: Event,
    /// Bartek jedzie w `+x`, więc prawy piorun jest z przodu.
    pub front: Event,
    pub back: Event,
}

impl LightningScene {
    /// `Err`, gdy `|β| ≥ 1` — tak samo jak [`boost_x`], bez cichego przycinania.
    pub fn new(beta: f64, half: f64) -> Result<Self, Superluminal> {
        let _g = gamma(beta)?;
        let (left, right) = lightning_strikes(half);
        let anna = Event::on_axis(half, 0.0);
        let front_ct = half / (1.0 + beta);
        let back_ct = half / (1.0 - beta);
        Ok(Self {
            left,
            right,
            anna,
            front: Event::on_axis(front_ct, beta * front_ct),
            back: Event::on_axis(back_ct, beta * back_ct),
        })
    }
}

/// Chwila na filmie: faza pętli `0..τ` na odcinek `[0, max]`.
pub fn phase_time(phase: f32, max: f64) -> f64 {
    f64::from(phase.rem_euclid(TAU) / TAU) * max
}

fn sample_event() -> Event {
    Event::on_axis(1.0, 0.4)
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

    let beta = playback.beta;
    if lesson == 1 {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(ui.available_width(), 88.0), egui::Sense::hover());
        paint_stage(ui, rect, beta, phase_time(playback.t, stage_ct_max(beta)));
        ui.add_space(4.0);
    }

    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    paint_diagram(ui, rect, lesson, beta, playback.t);
}

fn caption(lesson: u8) -> &'static str {
    match lesson {
        1 => "dwa pioruny · Anna w spoczynku · Bartek z β",
        2 => "stożek świetlny · światło zawsze 45°",
        _ => "osie Bartka przechylają się ku światłu",
    }
}

fn stage_ct_max(beta: f64) -> f64 {
    let later = LIGHTNING_HALF / (1.0 - clamp_beta(beta)).max(0.12);
    later.min(SPAN)
}

fn paint_stage(ui: &Ui, rect: Rect, beta: f64, ct: f64) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(12, 14, 20));
    let pad = 16.0;
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + pad, rect.top() + 8.0),
        Pos2::new(rect.right() - pad, rect.bottom() - 8.0),
    );
    if inner.width() < 8.0 {
        return;
    }
    let x_span = 1.7;
    let map = |x: f64| {
        let u = ((x / x_span) as f32 + 1.0) * 0.5;
        inner.left() + u * inner.width()
    };
    let y = inner.center().y + 10.0;
    let scale = inner.width() / (2.0 * x_span as f32);
    painter.line_segment(
        [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
        Stroke::new(1.0, AXIS),
    );

    let left_x = map(-LIGHTNING_HALF);
    let right_x = map(LIGHTNING_HALF);
    let lamp = Stroke::new(2.0, FLASH);
    painter.line_segment([Pos2::new(left_x, y), Pos2::new(left_x, y - 18.0)], lamp);
    painter.line_segment([Pos2::new(right_x, y), Pos2::new(right_x, y - 18.0)], lamp);
    if ct < 0.18 {
        let glow = (1.0 - (ct / 0.18) as f32).clamp(0.0, 1.0);
        let r = 5.0 + 10.0 * glow;
        painter.circle_filled(
            Pos2::new(left_x, y - 18.0),
            r,
            Color32::from_rgba_unmultiplied(240, 220, 120, (180.0 * glow) as u8),
        );
        painter.circle_filled(
            Pos2::new(right_x, y - 18.0),
            r,
            Color32::from_rgba_unmultiplied(240, 220, 120, (180.0 * glow) as u8),
        );
    }
    if ct > 0.0 {
        let radius = (ct as f32) * scale;
        painter.circle_stroke(Pos2::new(left_x, y), radius, Stroke::new(1.2, LIGHT));
        painter.circle_stroke(Pos2::new(right_x, y), radius, Stroke::new(1.2, LIGHT));
    }

    let anna_x = map(0.0);
    painter.circle_filled(Pos2::new(anna_x, y - 6.0), 5.0, ANNA);
    painter.text(
        Pos2::new(anna_x, y + 6.0),
        egui::Align2::CENTER_TOP,
        "Anna",
        FontId::proportional(10.0),
        ANNA,
    );
    let bartek_x = map(beta * ct);
    painter.circle_filled(Pos2::new(bartek_x, y - 6.0), 5.0, BARTEK);
    painter.rect_stroke(
        Rect::from_center_size(Pos2::new(bartek_x, y - 14.0), Vec2::new(28.0, 12.0)),
        2.0,
        Stroke::new(1.0, BARTEK),
        StrokeKind::Middle,
    );
    painter.text(
        Pos2::new(bartek_x, inner.top() + 2.0),
        egui::Align2::CENTER_TOP,
        "Bartek",
        FontId::proportional(10.0),
        BARTEK,
    );
}

fn paint_diagram(ui: &Ui, rect: Rect, lesson: u8, beta: f64, phase: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, BG);
    let inner = square_in(rect, 28.0);
    if inner.width() < 16.0 {
        return;
    }
    let Some(plot) = Plot::new(inner, SPAN) else {
        return;
    };
    let clip = ui.painter_at(inner);

    paint_grid(&clip, &plot);
    if lesson == 2 {
        paint_cone_fill(&clip, &plot);
    }
    paint_lab_axes(&painter, &plot);
    paint_light(&clip, &plot);

    match lesson {
        1 => paint_lightning(&clip, &painter, &plot, beta, phase),
        2 => paint_cone_worldlines(&clip, &painter, &plot, beta, phase),
        _ => paint_boost(&clip, &painter, &plot, beta, phase),
    }
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

fn paint_lab_axes(painter: &egui::Painter, plot: &Plot) {
    let stroke = Stroke::new(1.4, AXIS);
    arrow(painter, plot.px(0.0, -SPAN), plot.px(0.0, SPAN), stroke);
    arrow(painter, plot.px(-SPAN, 0.0), plot.px(SPAN, 0.0), stroke);
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

fn paint_light(painter: &egui::Painter, plot: &Plot) {
    let stroke = Stroke::new(1.3, LIGHT);
    painter.line_segment(plot.seg(-SPAN, -SPAN, SPAN, SPAN), stroke);
    painter.line_segment(plot.seg(-SPAN, SPAN, SPAN, -SPAN), stroke);
}

fn paint_cone_fill(painter: &egui::Painter, plot: &Plot) {
    let o = plot.px(0.0, 0.0);
    let tr = plot.px(SPAN, SPAN);
    let tl = plot.px(SPAN, -SPAN);
    let br = plot.px(-SPAN, SPAN);
    let bl = plot.px(-SPAN, -SPAN);
    let future = Color32::from_rgba_unmultiplied(40, 70, 90, 50);
    let past = Color32::from_rgba_unmultiplied(40, 50, 70, 40);
    let elsewhere = Color32::from_rgba_unmultiplied(70, 50, 60, 40);
    painter.add(egui::Shape::convex_polygon(
        vec![o, tr, tl],
        future,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![o, br, bl],
        past,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![o, tr, br],
        elsewhere,
        Stroke::NONE,
    ));
    painter.add(egui::Shape::convex_polygon(
        vec![o, tl, bl],
        elsewhere,
        Stroke::NONE,
    ));
}

fn paint_lightning(
    clip: &egui::Painter,
    labels: &egui::Painter,
    plot: &Plot,
    beta: f64,
    phase: f32,
) {
    let Ok(scene) = LightningScene::new(beta, LIGHTNING_HALF) else {
        return;
    };
    let ct_now = phase_time(phase, stage_ct_max(beta));
    let g = gamma_of(beta);

    clip.line_segment(plot.seg(-SPAN, 0.0, SPAN, 0.0), Stroke::new(1.6, ANNA));
    let axes = primed_axes(beta);
    arrow(
        clip,
        plot.px(-SPAN * axes.ct_ct, -SPAN * axes.ct_x),
        plot.px(SPAN * axes.ct_ct, SPAN * axes.ct_x),
        Stroke::new(1.8, BARTEK),
    );

    dashed(
        clip,
        plot.seg(ct_now, -SPAN, ct_now, SPAN),
        Stroke::new(1.0, ANNA),
    );
    let ct_bartek_now = ct_now * (1.0 - beta * beta);
    dashed(
        clip,
        plot.seg(
            ct_bartek_now - beta * SPAN,
            -SPAN,
            ct_bartek_now + beta * SPAN,
            SPAN,
        ),
        Stroke::new(1.0, BARTEK),
    );

    for (src, toward_right) in [(scene.left, true), (scene.right, false)] {
        let dir = if toward_right { 1.0 } else { -1.0 };
        clip.line_segment(
            plot.seg(0.0, src.x, SPAN, src.x + dir * SPAN),
            Stroke::new(1.0, LIGHT),
        );
    }

    dot(
        clip,
        labels,
        plot.px(scene.left.ct, scene.left.x),
        FLASH,
        "lewy",
        true,
    );
    dot(
        clip,
        labels,
        plot.px(scene.right.ct, scene.right.x),
        FLASH,
        "prawy",
        false,
    );
    dot(
        clip,
        labels,
        plot.px(scene.anna.ct, scene.anna.x),
        ANNA,
        "Anna",
        false,
    );
    if beta > 0.04 {
        dot(
            clip,
            labels,
            plot.px(scene.front.ct, scene.front.x),
            BARTEK,
            "przód",
            false,
        );
        if scene.back.ct <= SPAN {
            dot(
                clip,
                labels,
                plot.px(scene.back.ct, scene.back.x),
                BARTEK,
                "tył",
                false,
            );
        }
    }

    if let (Ok(left_p), Ok(right_p)) = (boost_x(scene.left, beta), boost_x(scene.right, beta)) {
        let hud = format!(
            "Anna: Δct = 0    Bartek: ct' lewy = {:+.2}  prawy = {:+.2}    γ = {g:.3}",
            left_p.ct, right_p.ct
        );
        labels.text(
            plot.px(-SPAN, -SPAN) + Vec2::new(0.0, 18.0),
            egui::Align2::LEFT_TOP,
            hud,
            FontId::monospace(11.0),
            LABEL,
        );
    }

    let now = plot.px(ct_now, 0.0);
    clip.circle_filled(now, 4.0, ANNA);
}

fn paint_cone_worldlines(
    clip: &egui::Painter,
    labels: &egui::Painter,
    plot: &Plot,
    beta: f64,
    phase: f32,
) {
    let ct = phase_time(phase, SPAN * 0.85);
    let x = beta * ct;
    clip.line_segment(plot.seg(-SPAN, 0.0, SPAN, 0.0), Stroke::new(1.5, ANNA));
    let axes = primed_axes(beta);
    clip.line_segment(
        plot.seg(
            -SPAN * axes.ct_ct,
            -SPAN * axes.ct_x,
            SPAN * axes.ct_ct,
            SPAN * axes.ct_x,
        ),
        Stroke::new(1.8, BARTEK),
    );

    let you = Event::on_axis(ct, x);
    let reach = SPAN * 1.4;
    clip.line_segment(
        plot.seg(you.ct - reach, you.x - reach, you.ct + reach, you.x + reach),
        Stroke::new(1.1, LIGHT),
    );
    clip.line_segment(
        plot.seg(you.ct - reach, you.x + reach, you.ct + reach, you.x - reach),
        Stroke::new(1.1, LIGHT),
    );
    clip.circle_filled(
        plot.px(you.ct, you.x),
        5.0,
        Color32::from_rgb(230, 230, 240),
    );

    let s2 = you.interval_sq();
    labels.text(
        plot.px(SPAN, -SPAN) + Vec2::new(4.0, 16.0),
        egui::Align2::LEFT_TOP,
        format!("s² = (ct)² − x² = {s2:.2}    przyszłość / przeszłość / gdzie indziej"),
        FontId::monospace(11.0),
        LABEL,
    );
    labels.text(
        plot.px(SPAN * 0.72, 0.12),
        egui::Align2::LEFT_CENTER,
        "przyszłość",
        FontId::proportional(11.0),
        ANNA,
    );
    labels.text(
        plot.px(-SPAN * 0.55, SPAN * 0.72),
        egui::Align2::CENTER_CENTER,
        "gdzie indziej",
        FontId::proportional(11.0),
        Color32::from_rgb(180, 140, 150),
    );
}

fn paint_boost(clip: &egui::Painter, labels: &egui::Painter, plot: &Plot, beta: f64, phase: f32) {
    let axes = primed_axes(beta);
    arrow(
        clip,
        plot.px(-SPAN * axes.ct_ct, -SPAN * axes.ct_x),
        plot.px(SPAN * axes.ct_ct, SPAN * axes.ct_x),
        Stroke::new(1.8, BARTEK),
    );
    arrow(
        clip,
        plot.px(-SPAN * axes.x_ct, -SPAN * axes.x_x),
        plot.px(SPAN * axes.x_ct, SPAN * axes.x_x),
        Stroke::new(1.8, BARTEK),
    );
    labels.text(
        plot.px(SPAN * 0.85 * axes.ct_ct, SPAN * 0.85 * axes.ct_x) + Vec2::new(6.0, 0.0),
        egui::Align2::LEFT_CENTER,
        "ct'",
        FontId::monospace(12.0),
        BARTEK,
    );
    labels.text(
        plot.px(SPAN * 0.85 * axes.x_ct, SPAN * 0.85 * axes.x_x) + Vec2::new(0.0, 10.0),
        egui::Align2::CENTER_TOP,
        "x'",
        FontId::monospace(12.0),
        BARTEK,
    );

    let p = sample_event();
    let Ok(primed) = boost_x(p, beta) else {
        return;
    };
    let Ok(on_ct) = boost_x(Event::on_axis(primed.ct, 0.0), -beta) else {
        return;
    };
    let Ok(on_x) = boost_x(Event::on_axis(0.0, primed.x), -beta) else {
        return;
    };
    dashed(
        clip,
        plot.seg(p.ct, p.x, on_ct.ct, on_ct.x),
        Stroke::new(1.0, BARTEK),
    );
    dashed(
        clip,
        plot.seg(p.ct, p.x, on_x.ct, on_x.x),
        Stroke::new(1.0, BARTEK),
    );
    clip.circle_filled(plot.px(p.ct, p.x), 5.0, Color32::WHITE);
    labels.text(
        plot.px(p.ct, p.x) + Vec2::new(8.0, -6.0),
        egui::Align2::LEFT_BOTTOM,
        "P",
        FontId::monospace(12.0),
        Color32::WHITE,
    );

    let ct_you = phase_time(phase, SPAN * 0.8);
    clip.circle_filled(plot.px(ct_you, beta * ct_you), 4.0, BARTEK);

    let g = gamma_of(beta);
    let hud = format!(
        "β = {beta:.2}   γ = {g:.3}   P = (ct, x) = ({:.2}, {:.2})   P' = (ct', x') = ({:.2}, {:.2})",
        p.ct, p.x, primed.ct, primed.x
    );
    labels.text(
        plot.px(-SPAN, -SPAN) + Vec2::new(0.0, 18.0),
        egui::Align2::LEFT_TOP,
        hud,
        FontId::monospace(11.0),
        LABEL,
    );
}

fn arrow(painter: &egui::Painter, from: Pos2, to: Pos2, stroke: Stroke) {
    painter.line_segment([from, to], stroke);
    let dir = to - from;
    let len = dir.length();
    if len < 12.0 {
        return;
    }
    let u = dir / len;
    let n = Vec2::new(-u.y, u.x);
    let back = to - u * 8.0;
    painter.line_segment([to, back + n * 4.0], stroke);
    painter.line_segment([to, back - n * 4.0], stroke);
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

fn dot(
    clip: &egui::Painter,
    labels: &egui::Painter,
    p: Pos2,
    color: Color32,
    text: &str,
    left: bool,
) {
    clip.circle_filled(p, 4.5, color);
    let align = if left {
        egui::Align2::RIGHT_BOTTOM
    } else {
        egui::Align2::LEFT_BOTTOM
    };
    let off = if left {
        Vec2::new(-6.0, -3.0)
    } else {
        Vec2::new(6.0, -3.0)
    };
    labels.text(p + off, align, text, FontId::proportional(10.0), color);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_on_the_slider_matches_lorentz_textbook_values() {
        assert!((gamma_of(0.0) - 1.0).abs() < 1e-15);
        assert!((gamma_of(0.6) - 1.25).abs() < 1e-15);
        assert!((gamma_of(0.8) - 5.0 / 3.0).abs() < 1e-15);
        assert_eq!(gamma_of(0.6), gamma(0.6).expect("0.6 jest poniżej c"));
        assert_eq!(gamma_of(crate::viz::BETA_DEFAULT), 1.25);
    }

    #[test]
    fn slider_never_reaches_lightspeed() {
        assert!(clamp_beta(1.0) < 1.0);
        assert!(clamp_beta(4.0) < 1.0);
        assert!(clamp_beta(-0.3) >= 0.0);
        assert!(gamma(BETA_MAX).is_ok());
        assert!(gamma(clamp_beta(1.0)).is_ok());
    }

    #[test]
    fn primed_axes_tilt_toward_the_light() {
        let rest = tilt_from_light(0.0);
        let slow = tilt_from_light(0.4);
        let fast = tilt_from_light(0.8);
        assert!((rest - std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        assert!(fast < slow && slow < rest);
        assert!(tilt_from_light(0.95) < 0.1);
        let axes = primed_axes(0.6);
        assert!((axes.ct_x - 0.6).abs() < 1e-15 && (axes.ct_ct - 1.0).abs() < 1e-15);
        assert!((axes.x_ct - 0.6).abs() < 1e-15 && (axes.x_x - 1.0).abs() < 1e-15);
    }

    #[test]
    fn lightning_is_simultaneous_for_anna_and_splits_after_boost() {
        let (left, right) = lightning_strikes(LIGHTNING_HALF);
        assert_eq!(left.ct, right.ct);
        assert!(left.x < 0.0 && right.x > 0.0);
        let beta = 0.6;
        let left_p = boost_x(left, beta).unwrap();
        let right_p = boost_x(right, beta).unwrap();
        assert!(
            (left_p.ct - right_p.ct).abs() > 1.0,
            "pioruny miały stracić wspólną chwilę"
        );
        let g = gamma(beta).unwrap();
        assert!((left_p.ct - g * beta).abs() < 1e-15);
        assert!((right_p.ct + g * beta).abs() < 1e-15);
    }

    #[test]
    fn bartek_meets_the_forward_flash_first() {
        let scene = LightningScene::new(0.6, LIGHTNING_HALF).unwrap();
        assert!(scene.front.ct < scene.anna.ct);
        assert!(scene.anna.ct < scene.back.ct);
        assert!((scene.front.x - 0.6 * scene.front.ct).abs() < 1e-12);
        assert!((scene.back.x - 0.6 * scene.back.ct).abs() < 1e-12);
        assert!(scene.left.interval_sq_to(scene.back).abs() < 1e-12);
        assert!(scene.right.interval_sq_to(scene.front).abs() < 1e-12);
        assert!(scene.left.interval_sq_to(scene.anna).abs() < 1e-12);
        assert!(scene.right.interval_sq_to(scene.anna).abs() < 1e-12);
    }

    #[test]
    fn rest_bartek_is_anna() {
        let scene = LightningScene::new(0.0, LIGHTNING_HALF).unwrap();
        assert!((scene.front.ct - scene.anna.ct).abs() < 1e-15);
        assert!((scene.back.ct - scene.anna.ct).abs() < 1e-15);
        assert!(LightningScene::new(1.0, LIGHTNING_HALF).is_err());
    }

    #[test]
    fn sample_event_boost_matches_lorentz() {
        let p = sample_event();
        let beta = 0.6;
        let primed = boost_x(p, beta).unwrap();
        let g = 1.25;
        assert!((primed.ct - g * (p.ct - beta * p.x)).abs() < 1e-15);
        assert!((primed.x - g * (p.x - beta * p.ct)).abs() < 1e-15);
        let back = boost_x(primed, -beta).unwrap();
        assert!((back.ct - p.ct).abs() < 1e-12 && (back.x - p.x).abs() < 1e-12);
        assert!(p.interval_sq() > 0.0);
        assert!((primed.interval_sq() - p.interval_sq()).abs() < 1e-12);
    }

    #[test]
    fn stw_lessons_1_to_3_paint_without_a_window() {
        for index in 1..=3 {
            let ctx = egui::Context::default();
            ctx.begin_pass(egui::RawInput::default());
            egui::CentralPanel::default().show(&ctx, |ui| {
                let mut playback = Playback {
                    playing: false,
                    t: 1.2,
                    beta: crate::viz::BETA_DEFAULT,
                    ..Default::default()
                };
                draw(ui, index, &mut playback);
                assert!((gamma_of(playback.beta) - 1.25).abs() < 1e-15);
            });
            let _ = ctx.end_pass();
        }
    }
}
