//! Laboratorium zrzucania: foton albo cząstka w równiku Schwarzschilda.
//!
//! Liczby biorą się z [`bone_core::gr::geodesic`]. Ten plik składa zrzut w widok
//! z góry: horyzont, pierścienie `3M` / `6M`, kilka krzywych i suwak rozmachu.
//! Raytracera pikseli tu nie ma — tylko równik i tor.

use std::f64::consts::{FRAC_PI_2, TAU};

use bone_core::gr::geodesic::{self, GeodesicState, STATE_LEN};
use bone_core::gr::rk4::Scratch;
use bone_core::gr::{horizon_radius, isco_radius, photon_sphere_radius, Schwarzschild};
use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui};

use super::MASS_DEFAULT;
use crate::screen::LabId;

/// Masa stołu: te same `2M` / `3M` / `6M` co testy metryki i geodezyjnej.
pub const MASS: f64 = MASS_DEFAULT;
/// Zrzut z łąki: dość daleko, żeby parametr zderzenia miał miejsce na periapsis.
pub const R_START: f64 = 20.0;
/// Haust RK4 zrzutu. Koło bierze gęstszy krok od okresu.
pub const DROP_H: f64 = 0.05;
pub const B_MIN: f64 = 2.5;
pub const B_MAX: f64 = 9.0;
pub const B_CAPTURE: f64 = 4.0;
pub const B_ESCAPE: f64 = 7.0;
pub const L_MIN: f64 = 2.0;
pub const L_MAX: f64 = 6.2;
pub const L_CAPTURE: f64 = 2.5;
pub const L_ESCAPE: f64 = 6.0;
/// Energia zrzutu z nieskończoności: `E = 1`.
pub const DROP_ENERGY: f64 = 1.0;

const BG: Color32 = Color32::from_rgb(8, 10, 16);
const LABEL: Color32 = Color32::from_rgb(140, 150, 170);
const HORIZON_FILL: Color32 = Color32::from_rgb(12, 14, 18);
const RING_H: Color32 = Color32::from_rgb(90, 90, 100);
const RING_PH: Color32 = Color32::from_rgb(220, 180, 90);
const RING_ISCO: Color32 = Color32::from_rgb(120, 180, 220);
const CAP: Color32 = Color32::from_rgb(220, 120, 100);
const ORB: Color32 = Color32::from_rgb(160, 200, 140);
const ESC: Color32 = Color32::from_rgb(120, 180, 220);
const NOW: Color32 = Color32::from_rgb(230, 230, 240);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Null,
    Timelike,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Seat {
    Drop,
    Circle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fate {
    Capture,
    Orbit,
    Escape,
}

impl Fate {
    pub fn label(self) -> &'static str {
        match self {
            Self::Capture => "wychwyt",
            Self::Orbit => "orbita",
            Self::Escape => "ucieczka",
        }
    }

    fn color(self) -> Color32 {
        match self {
            Self::Capture => CAP,
            Self::Orbit => ORB,
            Self::Escape => ESC,
        }
    }
}

/// Co puszczamy: zrzut z łąki albo gotowe koło z silnika.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Photon { b: f64 },
    PhotonSphere,
    Particle { ell: f64 },
    Isco,
}

impl Kind {
    pub fn species(self) -> Species {
        match self {
            Self::Photon { .. } | Self::PhotonSphere => Species::Null,
            Self::Particle { .. } | Self::Isco => Species::Timelike,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trace {
    pub points: Vec<[f64; 2]>,
    pub radii: Vec<f64>,
    pub fate: Fate,
    pub energy: f64,
    pub angular_momentum: f64,
}

#[derive(Clone, Debug)]
struct Cache {
    key: (Species, Seat, i32),
    current: Trace,
    capture: Trace,
    orbit: Trace,
    escape: Trace,
}

/// Suwaki i play stołu zrzucania.
#[derive(Clone, Debug)]
pub struct Lab {
    pub playing: bool,
    pub t: f32,
    pub species: Species,
    pub seat: Seat,
    pub impact: f64,
    pub momentum: f64,
    cache: Option<Cache>,
}

impl Default for Lab {
    fn default() -> Self {
        Self {
            playing: true,
            t: 0.0,
            species: Species::Null,
            seat: Seat::Drop,
            impact: B_ESCAPE,
            momentum: L_CAPTURE,
            cache: None,
        }
    }
}

impl Lab {
    pub fn tick(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        self.t = (self.t + dt * 0.28).rem_euclid(1.0);
    }

    pub fn kind(&self) -> Kind {
        match (self.species, self.seat) {
            (Species::Null, Seat::Drop) => Kind::Photon {
                b: clamp_b(self.impact),
            },
            (Species::Null, Seat::Circle) => Kind::PhotonSphere,
            (Species::Timelike, Seat::Drop) => Kind::Particle {
                ell: clamp_l(self.momentum),
            },
            (Species::Timelike, Seat::Circle) => Kind::Isco,
        }
    }

    fn traces(&mut self) -> &Cache {
        let key = (
            self.species,
            self.seat,
            param_key(self.species, self.seat, self.impact, self.momentum),
        );
        let stale = match &self.cache {
            Some(c) => c.key != key,
            None => true,
        };
        if stale {
            self.cache = Some(Cache {
                key,
                current: trace(self.kind()),
                capture: match self.species {
                    Species::Null => trace(Kind::Photon { b: B_CAPTURE }),
                    Species::Timelike => trace(Kind::Particle { ell: L_CAPTURE }),
                },
                orbit: match self.species {
                    Species::Null => trace(Kind::PhotonSphere),
                    Species::Timelike => trace(Kind::Isco),
                },
                escape: match self.species {
                    Species::Null => trace(Kind::Photon { b: B_ESCAPE }),
                    Species::Timelike => trace(Kind::Particle { ell: L_ESCAPE }),
                },
            });
        }
        self.cache.as_ref().expect("cache po zapisie")
    }
}

fn param_key(species: Species, seat: Seat, impact: f64, momentum: f64) -> i32 {
    match (species, seat) {
        (_, Seat::Circle) => -1,
        (Species::Null, Seat::Drop) => (clamp_b(impact) * 1000.0).round() as i32,
        (Species::Timelike, Seat::Drop) => (clamp_l(momentum) * 1000.0).round() as i32,
    }
}

pub fn clamp_b(b: f64) -> f64 {
    if !b.is_finite() {
        B_ESCAPE
    } else {
        b.clamp(B_MIN, B_MAX)
    }
}

pub fn clamp_l(ell: f64) -> f64 {
    if !ell.is_finite() {
        L_CAPTURE
    } else {
        ell.clamp(L_MIN, L_MAX)
    }
}

pub fn mass() -> Schwarzschild {
    Schwarzschild::new(MASS).expect("M = 1")
}

/// `b_c = 3√3 M`. Krawędź wychwytu fotonu, ta sama co `circular_photon`.
pub fn photon_b_crit(mass: f64) -> f64 {
    3.0 * 3.0_f64.sqrt() * mass
}

/// `L` koła ISCO z silnika, nie ze wzoru na suwaku.
pub fn isco_ell(mass: f64) -> f64 {
    let bh = Schwarzschild::new(mass).expect("M ≥ 0");
    let s = GeodesicState::circular_timelike(bh, isco_radius(mass)).expect("ISCO");
    s.angular_momentum(bh).expect("L poza horyzontem")
}

pub fn launch(kind: Kind) -> GeodesicState {
    let bh = mass();
    match kind {
        Kind::PhotonSphere => GeodesicState::circular_photon(bh).expect("sfera fotonowa"),
        Kind::Isco => GeodesicState::circular_timelike(bh, isco_radius(MASS)).expect("ISCO"),
        Kind::Photon { b } => launch_null(clamp_b(b)).expect("zrzut fotonu"),
        Kind::Particle { ell } => launch_timelike(clamp_l(ell)).expect("zrzut cząstki"),
    }
}

fn launch_null(b: f64) -> Option<GeodesicState> {
    equatorial_inward(R_START, DROP_ENERGY, b, true)
}

fn launch_timelike(ell: f64) -> Option<GeodesicState> {
    equatorial_inward(R_START, DROP_ENERGY, ell, false)
}

/// Z łąki, w równiku, do środka. `null` ustawia `κ = 0`, inaczej `κ = −1`.
fn equatorial_inward(r: f64, energy: f64, ell: f64, null: bool) -> Option<GeodesicState> {
    let bh = mass();
    let f = bh.f(r).ok()?;
    if f <= 0.0 {
        return None;
    }
    let u_t = energy / f;
    let u_phi = ell / (r * r);
    let ur2 = if null {
        energy * energy - f * ell * ell / (r * r)
    } else {
        energy * energy - f * (1.0 + ell * ell / (r * r))
    };
    if ur2 < -1e-12 {
        return None;
    }
    Some(GeodesicState {
        t: 0.0,
        r,
        theta: FRAC_PI_2,
        phi: 0.0,
        u_t,
        u_r: -ur2.max(0.0).sqrt(),
        u_theta: 0.0,
        u_phi,
    })
}

fn xy(s: GeodesicState) -> [f64; 2] {
    let st = s.theta.sin();
    [s.r * st * s.phi.cos(), s.r * st * s.phi.sin()]
}

pub fn trace(kind: Kind) -> Trace {
    let bh = mass();
    let start = launch(kind);
    let energy = start.energy(bh).unwrap_or(0.0);
    let angular_momentum = start.angular_momentum(bh).unwrap_or(0.0);
    let circular = matches!(kind, Kind::PhotonSphere | Kind::Isco);
    let (h, n) = if circular {
        let period = TAU / start.u_phi.abs().max(1e-9);
        let n = 2000;
        (period / n as f64, n)
    } else {
        (DROP_H, 5000)
    };
    let mut y = start.to_array();
    let mut scratch = Scratch::with_len(STATE_LEN);
    let mut points = Vec::with_capacity(n + 1);
    let mut radii = Vec::with_capacity(n + 1);
    points.push(xy(start));
    radii.push(start.r);
    let mut last_ur = start.u_r;
    let mut hit_horizon = false;
    let r_h = horizon_radius(MASS);
    for _ in 0..n {
        if geodesic::step(bh, 0.0, &mut y, h, &mut scratch).is_err() {
            hit_horizon = true;
            break;
        }
        let s = match GeodesicState::from_slice(&y) {
            Ok(s) => s,
            Err(_) => {
                hit_horizon = true;
                break;
            }
        };
        if !s.r.is_finite() || s.r <= r_h + 0.04 * MASS {
            hit_horizon = true;
            points.push(xy(s));
            radii.push(s.r.max(r_h));
            break;
        }
        last_ur = s.u_r;
        points.push(xy(s));
        radii.push(s.r);
        if !circular && s.r >= R_START + 0.4 && s.u_r > 0.0 {
            break;
        }
    }
    let fate = classify(&radii, last_ur, circular, hit_horizon);
    Trace {
        points,
        radii,
        fate,
        energy,
        angular_momentum,
    }
}

fn classify(radii: &[f64], last_ur: f64, circular: bool, hit_horizon: bool) -> Fate {
    if circular {
        return Fate::Orbit;
    }
    if hit_horizon {
        return Fate::Capture;
    }
    let last = radii.last().copied().unwrap_or(0.0);
    let start = radii.first().copied().unwrap_or(last);
    let r_h = horizon_radius(MASS);
    if last <= r_h + 0.12 * MASS {
        return Fate::Capture;
    }
    if last >= start - 0.2 && last_ur > 0.0 {
        return Fate::Escape;
    }
    if last + 1.0 < start {
        Fate::Capture
    } else {
        Fate::Escape
    }
}

/// Drzwi z lekcji B6: bez nowej fizyki, z jawnym stołem zrzucania.
pub fn draw_drop_door(ui: &mut Ui) -> bool {
    let mut open = false;
    ui.add_space(12.0);
    ui.label(
        RichText::new("Drzwi do stołu, który Bone już umie liczyć")
            .size(16.0)
            .strong(),
    );
    ui.label(
        RichText::new(
            "Puszczasz foton albo cząstkę w równiku. Nie dodajesz siły w locie.\n\
             Mały rozmach tonie. Duży wraca na łąkę. Koło to 3M albo 6M.",
        )
        .small()
        .weak(),
    );
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        fate_tile(ui, Fate::Capture, "mały b, małe L");
        fate_tile(ui, Fate::Orbit, "3M · 6M");
        fate_tile(ui, Fate::Escape, "duży b, duże L");
    });
    ui.add_space(8.0);
    ui.label(
        RichText::new("Raytracera pikseli tu nie ma. Najpierw równik i kilka krzywych.")
            .small()
            .weak(),
    );
    ui.add_space(16.0);
    if ui
        .add_sized(
            [280.0, 36.0],
            egui::Button::new("Otwórz laboratorium geodezyjnych"),
        )
        .clicked()
    {
        open = true;
    }
    ui.add_space(6.0);
    ui.label(
        RichText::new(format!(
            "wejście → {} · równik Schwarzschilda",
            LabId::Geodesics.label()
        ))
        .small()
        .weak()
        .monospace(),
    );
    open
}

fn fate_tile(ui: &mut Ui, fate: Fate, hint: &str) {
    ui.group(|ui| {
        ui.set_width(90.0);
        ui.label(RichText::new(fate.label()).strong().color(fate.color()));
        ui.label(RichText::new(hint).small().weak());
    });
}

/// Cały stół: suwaki z lewej, równik w środku.
pub fn draw_lab(ctx: &egui::Context, lab: &mut Lab) {
    egui::SidePanel::left("geo-drop-panel")
        .resizable(false)
        .min_width(300.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| controls(ui, lab));
        });
    egui::CentralPanel::default().show(ctx, |ui| paint_equator(ui, lab));
}

fn controls(ui: &mut Ui, lab: &mut Lab) {
    ui.add_space(8.0);
    ui.label(RichText::new("Zrzut w równiku").strong());
    ui.label(
        RichText::new("Schwarzschild · G = c = 1 · M = 1. Widok z góry na θ = π/2.")
            .small()
            .weak(),
    );
    ui.add_space(10.0);
    ui.horizontal(|ui| {
        if ui
            .selectable_label(lab.species == Species::Null, "Foton")
            .clicked()
        {
            lab.species = Species::Null;
        }
        if ui
            .selectable_label(lab.species == Species::Timelike, "Cząstka")
            .clicked()
        {
            lab.species = Species::Timelike;
        }
    });
    ui.horizontal(|ui| {
        if ui
            .selectable_label(lab.seat == Seat::Drop, "Zrzut")
            .clicked()
        {
            lab.seat = Seat::Drop;
        }
        let circle = match lab.species {
            Species::Null => "Koło 3M",
            Species::Timelike => "Koło 6M",
        };
        if ui
            .selectable_label(lab.seat == Seat::Circle, circle)
            .clicked()
        {
            lab.seat = Seat::Circle;
        }
    });
    ui.add_space(8.0);
    match lab.species {
        Species::Null => {
            ui.add_enabled_ui(lab.seat == Seat::Drop, |ui| {
                ui.add(
                    egui::Slider::new(&mut lab.impact, B_MIN..=B_MAX)
                        .text("b / M")
                        .fixed_decimals(2),
                );
            });
            lab.impact = clamp_b(lab.impact);
            ui.label(
                RichText::new(format!(
                    "krawędź wychwytu  b/M = 3√3 ≈ {:.3}",
                    photon_b_crit(MASS)
                ))
                .small()
                .weak()
                .monospace(),
            );
        }
        Species::Timelike => {
            ui.add_enabled_ui(lab.seat == Seat::Drop, |ui| {
                ui.add(
                    egui::Slider::new(&mut lab.momentum, L_MIN..=L_MAX)
                        .text("L / M")
                        .fixed_decimals(2),
                );
            });
            lab.momentum = clamp_l(lab.momentum);
            ui.label(
                RichText::new(format!(
                    "ISCO  L/M = √12 ≈ {:.3} · zrzut E = 1  krawędź L/M = 4",
                    isco_ell(MASS)
                ))
                .small()
                .weak()
                .monospace(),
            );
        }
    }
    ui.add_space(8.0);
    let (fate, energy, ell) = {
        let c = lab.traces();
        (c.current.fate, c.current.energy, c.current.angular_momentum)
    };
    ui.label(
        RichText::new(format!("koniec: {}", fate.label()))
            .strong()
            .color(fate.color()),
    );
    ui.label(
        RichText::new(format!("E = {energy:.4}    L = {ell:.4}"))
            .small()
            .monospace(),
    );
    ui.add_space(8.0);
    let label = if lab.playing { "Pauza" } else { "Play" };
    if ui.button(label).clicked() {
        lab.playing = !lab.playing;
    }
    ui.add_space(12.0);
    ui.label(
        RichText::new("Cieńkie krzywe to trzy końce: wychwyt, koło, ucieczka. Gruba — Twój zrzut.")
            .small()
            .weak(),
    );
    ui.add_space(8.0);
    egui::Frame::group(ui.style())
        .fill(Color32::from_rgb(22, 28, 36))
        .inner_margin(8.0)
        .show(ui, |ui| {
            ui.label(
                RichText::new("siła nie ciągnie — idziesz prosto według linijki")
                    .italics()
                    .color(Color32::from_rgb(200, 230, 170)),
            );
        });
}

fn paint_equator(ui: &mut Ui, lab: &mut Lab) {
    let playing = lab.playing;
    let phase = lab.t;
    let species = lab.species;
    let cache = lab.traces();
    let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, BG);
    let r_view = R_START + 2.0;
    let map = |x: f64, y: f64| to_screen(rect, x, y, r_view);
    let origin = map(0.0, 0.0);
    let scale = 0.46 * rect.width().min(rect.height()) / r_view as f32;
    ring_fill(&painter, origin, scale, horizon_radius(MASS), HORIZON_FILL);
    ring_stroke(&painter, origin, scale, horizon_radius(MASS), RING_H, 2.0);
    ring_stroke(
        &painter,
        origin,
        scale,
        photon_sphere_radius(MASS),
        RING_PH,
        1.4,
    );
    ring_stroke(&painter, origin, scale, isco_radius(MASS), RING_ISCO, 1.4);
    caption(&painter, map, horizon_radius(MASS), 0.35, "2M");
    caption(&painter, map, photon_sphere_radius(MASS), 0.55, "3M");
    caption(&painter, map, isco_radius(MASS), 0.85, "6M");
    stroke_trace(&painter, map, &cache.capture, 1.2, CAP.gamma_multiply(0.55));
    stroke_trace(&painter, map, &cache.escape, 1.2, ESC.gamma_multiply(0.55));
    stroke_trace(&painter, map, &cache.orbit, 1.4, ORB.gamma_multiply(0.70));
    stroke_trace(
        &painter,
        map,
        &cache.current,
        2.4,
        cache.current.fate.color(),
    );
    if let Some(p) = along(&cache.current.points, phase) {
        painter.circle_filled(map(p[0], p[1]), 5.0, NOW);
    }
    let fate = cache.current.fate;
    painter.text(
        rect.left_top() + egui::vec2(12.0, 10.0),
        egui::Align2::LEFT_TOP,
        format!(
            "{}{} · {}",
            if playing { "" } else { "pauza · " },
            match species {
                Species::Null => "foton",
                Species::Timelike => "cząstka",
            },
            fate.label()
        ),
        FontId::proportional(14.0),
        fate.color(),
    );
}

fn to_screen(rect: Rect, x: f64, y: f64, r_view: f64) -> Pos2 {
    let s = 0.46 * rect.width().min(rect.height()) / r_view as f32;
    let c = rect.center();
    Pos2::new(c.x + x as f32 * s, c.y - y as f32 * s)
}

fn ring_fill(painter: &egui::Painter, origin: Pos2, scale: f32, r: f64, color: Color32) {
    painter.circle_filled(origin, (r as f32) * scale, color);
}

fn ring_stroke(
    painter: &egui::Painter,
    origin: Pos2,
    scale: f32,
    r: f64,
    color: Color32,
    width: f32,
) {
    painter.circle_stroke(origin, (r as f32) * scale, Stroke::new(width, color));
}

fn caption(painter: &egui::Painter, map: impl Fn(f64, f64) -> Pos2, r: f64, ang: f64, text: &str) {
    let p = map(r * ang.cos(), r * ang.sin());
    painter.text(
        p,
        egui::Align2::LEFT_CENTER,
        text,
        FontId::monospace(11.0),
        LABEL,
    );
}

fn stroke_trace(
    painter: &egui::Painter,
    map: impl Fn(f64, f64) -> Pos2,
    trace: &Trace,
    width: f32,
    color: Color32,
) {
    let pts = &trace.points;
    if pts.len() < 2 {
        return;
    }
    let stride = (pts.len() / 1200).max(1);
    let mut prev = map(pts[0][0], pts[0][1]);
    for p in pts.iter().step_by(stride).skip(1) {
        let next = map(p[0], p[1]);
        painter.line_segment([prev, next], Stroke::new(width, color));
        prev = next;
    }
    let last = pts[pts.len() - 1];
    let end = map(last[0], last[1]);
    if prev != end {
        painter.line_segment([prev, end], Stroke::new(width, color));
    }
}

fn along(points: &[[f64; 2]], t: f32) -> Option<[f64; 2]> {
    if points.is_empty() {
        return None;
    }
    let n = points.len();
    if n == 1 {
        return Some(points[0]);
    }
    let x = (t.rem_euclid(1.0) as f64) * (n - 1) as f64;
    let i = x.floor() as usize;
    let i = i.min(n - 2);
    let a = points[i];
    let b = points[i + 1];
    let f = x - i as f64;
    Some([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliders_stay_in_range() {
        assert_eq!(clamp_b(B_ESCAPE), B_ESCAPE);
        assert_eq!(clamp_b(0.0), B_MIN);
        assert_eq!(clamp_b(40.0), B_MAX);
        assert_eq!(clamp_b(f64::NAN), B_ESCAPE);
        assert_eq!(clamp_l(L_CAPTURE), L_CAPTURE);
        assert_eq!(clamp_l(-2.0), L_MIN);
        assert_eq!(clamp_l(99.0), L_MAX);
        assert_eq!(clamp_l(f64::NAN), L_CAPTURE);
    }

    #[test]
    fn photon_edge_and_isco_match_the_engine() {
        let bh = mass();
        let photon = GeodesicState::circular_photon(bh).unwrap();
        let b = photon.angular_momentum(bh).unwrap() / photon.energy(bh).unwrap();
        assert!((b - photon_b_crit(MASS)).abs() < 1e-15);
        assert!((photon.r - photon_sphere_radius(MASS)).abs() < 1e-15);
        let isco = GeodesicState::circular_timelike(bh, isco_radius(MASS)).unwrap();
        assert!((isco.r - 6.0).abs() < 1e-15);
        let want_e = 4.0 / 18.0_f64.sqrt();
        let want_l = 12.0_f64.sqrt();
        assert!((isco.energy(bh).unwrap() - want_e).abs() < 1e-15);
        assert!((isco.angular_momentum(bh).unwrap() - want_l).abs() < 1e-15);
        assert!((isco_ell(MASS) - want_l).abs() < 1e-15);
    }

    #[test]
    fn small_impact_is_capture_large_is_escape() {
        let small = trace(Kind::Photon { b: B_CAPTURE });
        let large = trace(Kind::Photon { b: B_ESCAPE });
        assert_eq!(small.fate, Fate::Capture, "b = {B_CAPTURE} miało tonąć");
        assert_eq!(large.fate, Fate::Escape, "b = {B_ESCAPE} miało uciec");
        assert!(small.radii.iter().copied().fold(f64::INFINITY, f64::min) < 3.0);
        assert!(large.radii.iter().copied().fold(f64::INFINITY, f64::min) > 3.0);
    }

    #[test]
    fn small_momentum_is_capture_large_is_escape() {
        let small = trace(Kind::Particle { ell: L_CAPTURE });
        let large = trace(Kind::Particle { ell: L_ESCAPE });
        assert_eq!(small.fate, Fate::Capture, "L = {L_CAPTURE} miało tonąć");
        assert_eq!(large.fate, Fate::Escape, "L = {L_ESCAPE} miało uciec");
        assert!(small.radii.iter().copied().fold(f64::INFINITY, f64::min) < 6.0);
        assert!(large.radii.iter().copied().fold(f64::INFINITY, f64::min) > 4.0);
    }

    #[test]
    fn circles_stay_at_three_m_and_six_m() {
        let photon = trace(Kind::PhotonSphere);
        let isco = trace(Kind::Isco);
        assert_eq!(photon.fate, Fate::Orbit);
        assert_eq!(isco.fate, Fate::Orbit);
        let ph_dr = photon
            .radii
            .iter()
            .map(|r| (r - 3.0).abs())
            .fold(0.0_f64, f64::max);
        let is_dr = isco
            .radii
            .iter()
            .map(|r| (r - 6.0).abs())
            .fold(0.0_f64, f64::max);
        assert!(ph_dr < 1e-6, "sfera fotonowa |Δr|_max = {ph_dr}");
        assert!(is_dr < 1e-7, "ISCO |Δr|_max = {is_dr}");
        assert!((photon.energy - 1.0).abs() < 1e-12);
        let want_e = 4.0 / 18.0_f64.sqrt();
        assert!((isco.energy - want_e).abs() < 1e-12);
    }

    #[test]
    fn null_and_timelike_keep_kappa() {
        let bh = mass();
        let photon = launch(Kind::Photon { b: B_ESCAPE });
        let particle = launch(Kind::Particle { ell: L_ESCAPE });
        assert!(photon.u_sq(bh).unwrap().abs() < 1e-12);
        assert!((particle.u_sq(bh).unwrap() + 1.0).abs() < 1e-12);
        assert!(photon.u_r < 0.0 && particle.u_r < 0.0);
    }

    #[test]
    fn lab_kind_follows_the_toggles() {
        let mut lab = Lab::default();
        assert_eq!(lab.kind(), Kind::Photon { b: B_ESCAPE });
        lab.seat = Seat::Circle;
        assert_eq!(lab.kind(), Kind::PhotonSphere);
        lab.species = Species::Timelike;
        assert_eq!(lab.kind(), Kind::Isco);
        lab.seat = Seat::Drop;
        lab.momentum = L_CAPTURE;
        assert_eq!(lab.kind(), Kind::Particle { ell: L_CAPTURE });
    }

    #[test]
    fn drop_door_and_lab_paint_without_a_window() {
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        egui::CentralPanel::default().show(&ctx, |ui| {
            assert!(!draw_drop_door(ui));
        });
        let _ = ctx.end_pass();
        let ctx = egui::Context::default();
        ctx.begin_pass(egui::RawInput::default());
        let mut lab = Lab {
            playing: false,
            ..Lab::default()
        };
        draw_lab(&ctx, &mut lab);
        let _ = ctx.end_pass();
    }
}
