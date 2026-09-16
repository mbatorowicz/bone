//! Ekran lekcji: tekst z Markdown po prawej, obraz po lewej.
//!
//! Parser zna nagłówki, akapity, `> callout` i fenced code —
//! tyle, ile stuby i późniejsze teksty kursu naprawdę użyją. Nieznany znacznik
//! spada do akapitu zamiast wywalić okno. Układ 60/40 i play/pauza są wspólne.
//! STW 1–6 rysuje [`crate::viz`]; lekcja 7 otwiera laboratorium N-ciał.
//! Geodezyjna 1–5 też: kula, siatka z suwakiem M, film RK4 vs Euler.
//! Lekcja 6 otwiera stół zrzucania. Czarna dziura 1–2: pierścienie i pęk
//! w 2D. C3 i C4 otwierają laboratorium raytracera (klatka w tle).
//! Tensory / Einstein / PINN: algebra, pole i residual — obraz z `gr`, nie stub.
//! Kerr: wleczenie, ergo, pęk pierścieni; lekcja 4 otwiera raytracer (spin = 0).
//! Siatka PDE: stub i placeholder; ostatnia lekcja wraca na mapę, bez labu.

use std::f32::consts::TAU;

use eframe::egui::{self, Color32, FontId, Pos2, Rect, RichText, Stroke, Ui};

use crate::screen::{LabId, LessonId, Track};

macro_rules! lesson_md {
    ($path:expr) => {
        include_str!(concat!("../../lessons/", $path))
    };
}

/// Co przyciski lekcji chcą od okna.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Map,
    Lesson(LessonId),
    Lab(LabId),
}

/// Play/pauza, faza pętli, β Minkowskiego, masa Schwarzschilda, obrót kuli
/// i nachylenie dysku (ścieżka C, ten sam `spin`). Na ścieżce Kerr `spin`
/// to `a/M`. `probe` to r albo x na ścieżkach Einstein / PINN / tensory;
/// `density` to ρ pyłu.
///
/// Reset przy zmianie lekcji, żeby obraz nie skakał ze środka poprzedniej
/// strony, a γ wracało do podręcznikowego 5/4 i M do jedynki z testów metryki.
#[derive(Clone, Copy, Debug)]
pub struct Playback {
    pub playing: bool,
    pub t: f32,
    pub beta: f64,
    pub mass: f64,
    pub spin: f32,
    pub probe: f64,
    pub density: f64,
}

impl Default for Playback {
    fn default() -> Self {
        Self {
            playing: true,
            t: 0.0,
            beta: crate::viz::BETA_DEFAULT,
            mass: crate::viz::MASS_DEFAULT,
            spin: crate::viz::SPIN_DEFAULT,
            probe: crate::viz::PROBE_DEFAULT,
            density: 0.0,
        }
    }
}

impl Playback {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn tick(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        self.t = (self.t + dt).rem_euclid(TAU);
    }
}

/// Wstecz / Dalej po ścieżce. A7 otwiera N-ciała; B6 — geodezyjne; C4 i Kerr 4 — raytracer.
pub fn step(id: LessonId, forward: bool) -> Action {
    if forward {
        if let Some(lab) = id.opens_lab() {
            return Action::Lab(lab);
        }
        id.next().map(Action::Lesson).unwrap_or(Action::Map)
    } else {
        id.prev().map(Action::Lesson).unwrap_or(Action::Map)
    }
}

pub fn source(id: LessonId) -> &'static str {
    match (id.track, id.index) {
        (Track::Stw, 1) => lesson_md!("stw/01.md"),
        (Track::Stw, 2) => lesson_md!("stw/02.md"),
        (Track::Stw, 3) => lesson_md!("stw/03.md"),
        (Track::Stw, 4) => lesson_md!("stw/04.md"),
        (Track::Stw, 5) => lesson_md!("stw/05.md"),
        (Track::Stw, 6) => lesson_md!("stw/06.md"),
        (Track::Stw, 7) => lesson_md!("stw/07.md"),
        (Track::Geo, 1) => lesson_md!("geo/01.md"),
        (Track::Geo, 2) => lesson_md!("geo/02.md"),
        (Track::Geo, 3) => lesson_md!("geo/03.md"),
        (Track::Geo, 4) => lesson_md!("geo/04.md"),
        (Track::Geo, 5) => lesson_md!("geo/05.md"),
        (Track::Geo, 6) => lesson_md!("geo/06.md"),
        (Track::Bh, 1) => lesson_md!("bh/01.md"),
        (Track::Bh, 2) => lesson_md!("bh/02.md"),
        (Track::Bh, 3) => lesson_md!("bh/03.md"),
        (Track::Bh, 4) => lesson_md!("bh/04.md"),
        (Track::Tensor, 1) => lesson_md!("ten/01.md"),
        (Track::Tensor, 2) => lesson_md!("ten/02.md"),
        (Track::Tensor, 3) => lesson_md!("ten/03.md"),
        (Track::Tensor, 4) => lesson_md!("ten/04.md"),
        (Track::Tensor, 5) => lesson_md!("ten/05.md"),
        (Track::Einstein, 1) => lesson_md!("ein/01.md"),
        (Track::Einstein, 2) => lesson_md!("ein/02.md"),
        (Track::Einstein, 3) => lesson_md!("ein/03.md"),
        (Track::Einstein, 4) => lesson_md!("ein/04.md"),
        (Track::Pinn, 1) => lesson_md!("pinn/01.md"),
        (Track::Pinn, 2) => lesson_md!("pinn/02.md"),
        (Track::Pinn, 3) => lesson_md!("pinn/03.md"),
        (Track::Pinn, 4) => lesson_md!("pinn/04.md"),
        (Track::Kerr, 1) => lesson_md!("kerr/01.md"),
        (Track::Kerr, 2) => lesson_md!("kerr/02.md"),
        (Track::Kerr, 3) => lesson_md!("kerr/03.md"),
        (Track::Kerr, 4) => lesson_md!("kerr/04.md"),
        (Track::Pde, 1) => lesson_md!("pde/01.md"),
        (Track::Pde, 2) => lesson_md!("pde/02.md"),
        (Track::Pde, 3) => lesson_md!("pde/03.md"),
        (Track::Pde, 4) => lesson_md!("pde/04.md"),
        _ => "",
    }
}

/// Blok, który umiemy narysować. Reszta Markdowna nie istnieje w tym parserze.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Block {
    Heading { level: u8, text: String },
    Paragraph(String),
    Callout(String),
    Code { lang: String, body: String },
}

pub fn parse(src: &str) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut lines = src.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim().is_empty() {
            continue;
        }
        if let Some(lang) = fence_lang(line) {
            let mut body = String::new();
            for inner in lines.by_ref() {
                if fence_lang(inner).is_some() {
                    break;
                }
                if !body.is_empty() {
                    body.push('\n');
                }
                body.push_str(inner);
            }
            blocks.push(Block::Code { lang, body });
            continue;
        }
        if let Some((level, text)) = heading(line) {
            blocks.push(Block::Heading { level, text });
            continue;
        }
        if let Some(first) = callout_line(line) {
            let mut parts = vec![first];
            while let Some(next) = lines.peek().copied().and_then(callout_line) {
                lines.next();
                if !next.is_empty() {
                    parts.push(next);
                }
            }
            let text = parts.join(" ");
            if !text.is_empty() {
                blocks.push(Block::Callout(text));
            }
            continue;
        }
        let mut parts = vec![line.trim().to_string()];
        while let Some(next) = lines.peek().copied() {
            if next.trim().is_empty()
                || fence_lang(next).is_some()
                || heading(next).is_some()
                || callout_line(next).is_some()
            {
                break;
            }
            lines.next();
            parts.push(next.trim().to_string());
        }
        let text = parts.join(" ");
        if !text.is_empty() {
            blocks.push(Block::Paragraph(text));
        }
    }
    blocks
}

fn fence_lang(line: &str) -> Option<String> {
    let trimmed = line.trim();
    trimmed
        .strip_prefix("```")
        .map(|rest| rest.trim().to_string())
}

fn heading(line: &str) -> Option<(u8, String)> {
    let stripped = line.trim();
    let hashes = stripped.bytes().take_while(|&b| b == b'#').count();
    if !(1..=6).contains(&hashes) {
        return None;
    }
    let rest = &stripped[hashes..];
    if !rest.is_empty() && !rest.starts_with(char::is_whitespace) {
        return None;
    }
    Some((hashes as u8, rest.trim().to_string()))
}

fn callout_line(line: &str) -> Option<String> {
    line.trim_start()
        .strip_prefix('>')
        .map(|rest| rest.trim().to_string())
}

pub fn draw(ui: &mut Ui, id: LessonId, playback: &mut Playback) -> Action {
    let mut action = Action::None;
    let text_w = (ui.available_width() * 0.4).clamp(280.0, 520.0);
    egui::SidePanel::right("lesson-text")
        .resizable(false)
        .exact_width(text_w)
        .show_inside(ui, |ui| {
            action = draw_text(ui, id);
        });
    egui::CentralPanel::default().show_inside(ui, |ui| match crate::viz::draw(ui, id, playback) {
        crate::viz::Outcome::Placeholder => draw_placeholder(ui, playback),
        crate::viz::Outcome::Drawn => {}
        crate::viz::Outcome::OpenLab(lab) => {
            if action == Action::None {
                action = Action::Lab(lab);
            }
        }
    });
    action
}

fn draw_text(ui: &mut Ui, id: LessonId) -> Action {
    let mut action = Action::None;
    egui::TopBottomPanel::bottom("lesson-nav")
        .show_separator_line(true)
        .show_inside(ui, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let back = if id.prev().is_some() {
                    "Wstecz"
                } else {
                    "Mapa"
                };
                if ui.button(back).clicked() {
                    action = step(id, false);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let forth = if id.next().is_some() {
                        "Dalej"
                    } else if let Some(lab) = id.opens_lab() {
                        lab.course_button()
                    } else {
                        "Mapa"
                    };
                    if ui.button(forth).clicked() {
                        action = step(id, true);
                    }
                });
            });
            ui.add_space(4.0);
            ui.label(RichText::new(id.slug()).small().weak().monospace());
        });

    egui::ScrollArea::vertical()
        .id_salt("lesson-scroll")
        .show(ui, |ui| {
            ui.add_space(8.0);
            ui.label(RichText::new(id.title()).size(18.0).strong());
            ui.add_space(10.0);
            for block in parse(source(id)) {
                show_block(ui, &block);
                ui.add_space(8.0);
            }
            ui.add_space(12.0);
        });
    action
}

fn show_block(ui: &mut Ui, block: &Block) {
    match block {
        Block::Heading { level, text } => {
            let size = match level {
                1 => 20.0,
                2 => 17.0,
                _ => 15.0,
            };
            ui.label(RichText::new(text).size(size).strong());
        }
        Block::Paragraph(text) => {
            ui.label(text);
        }
        Block::Callout(text) => {
            egui::Frame::group(ui.style())
                .fill(Color32::from_rgb(28, 34, 22))
                .stroke(Stroke::new(1.0, Color32::from_rgb(160, 200, 140)))
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.label(RichText::new(text).italics());
                });
        }
        Block::Code { lang, body } => {
            if !lang.is_empty() {
                ui.label(RichText::new(lang).small().weak().monospace());
            }
            egui::Frame::NONE
                .fill(Color32::from_rgb(12, 16, 24))
                .inner_margin(8.0)
                .show(ui, |ui| {
                    ui.monospace(body);
                });
        }
    }
}

fn draw_placeholder(ui: &mut Ui, playback: &mut Playback) {
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let label = if playback.playing { "Pauza" } else { "Play" };
            if ui.button(label).clicked() {
                playback.playing = !playback.playing;
            }
            ui.label(RichText::new("placeholder — nie fizyka").small().weak());
        });
        ui.add_space(4.0);
        let (rect, _) = ui.allocate_exact_size(ui.available_size(), egui::Sense::hover());
        paint_orbit(ui, rect, playback.t);
    });
}

fn paint_orbit(ui: &Ui, rect: Rect, t: f32) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(8, 10, 16));
    let pad = 28.0;
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + pad, rect.top() + pad),
        Pos2::new(rect.right() - pad, rect.bottom() - pad),
    );
    if inner.width() < 8.0 || inner.height() < 8.0 {
        return;
    }
    let c = inner.center();
    let r = inner.width().min(inner.height()) * 0.38;
    let axis = Color32::from_rgb(50, 58, 72);
    painter.line_segment(
        [Pos2::new(inner.left(), c.y), Pos2::new(inner.right(), c.y)],
        Stroke::new(1.0, axis),
    );
    painter.line_segment(
        [Pos2::new(c.x, inner.top()), Pos2::new(c.x, inner.bottom())],
        Stroke::new(1.0, axis),
    );
    painter.text(
        Pos2::new(inner.right() - 4.0, c.y - 4.0),
        egui::Align2::RIGHT_BOTTOM,
        "x",
        FontId::monospace(11.0),
        Color32::from_rgb(140, 150, 170),
    );
    painter.text(
        Pos2::new(c.x + 6.0, inner.top() + 2.0),
        egui::Align2::LEFT_TOP,
        "ct",
        FontId::monospace(11.0),
        Color32::from_rgb(140, 150, 170),
    );
    painter.circle_stroke(c, r, Stroke::new(1.0, Color32::from_rgb(40, 56, 80)));
    let x = c.x + r * t.cos();
    let y = c.y - r * t.sin();
    painter.circle_filled(Pos2::new(x, y), 6.0, Color32::from_rgb(120, 180, 220));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_reads_headings_paragraphs_callouts_and_code() {
        let src = "\
# Tytuł

Akapit jeden
ciąg dalszy.

> Callout
> druga linia.

```rust
let x = 1;
```
";
        let blocks = parse(src);
        assert_eq!(
            blocks,
            vec![
                Block::Heading {
                    level: 1,
                    text: "Tytuł".into(),
                },
                Block::Paragraph("Akapit jeden ciąg dalszy.".into()),
                Block::Callout("Callout druga linia.".into()),
                Block::Code {
                    lang: "rust".into(),
                    body: "let x = 1;".into(),
                },
            ]
        );
    }

    #[test]
    fn parser_does_not_panic_on_garbage_or_unclosed_fence() {
        let blocks = parse("```\nnie zamknięte\n# niby nagłówek w kodzie\ntekst\n> x\n###### ");
        assert!(!blocks.is_empty());
        parse("");
        parse("###");
        parse("```");
        parse("#notheading\nzwykły");
    }

    #[test]
    fn every_stub_parses_and_has_a_heading() {
        let mut seen = 0;
        for track in Track::ALL {
            for id in track.lessons() {
                let src = source(id);
                assert!(!src.trim().is_empty(), "pusty stub {}", id.slug());
                let blocks = parse(src);
                assert!(
                    blocks.iter().any(|b| matches!(b, Block::Heading { .. })),
                    "brak nagłówka {}",
                    id.slug()
                );
                assert!(
                    blocks.iter().any(|b| matches!(
                        b,
                        Block::Paragraph(_) | Block::Callout(_) | Block::Code { .. }
                    )),
                    "brak treści {}",
                    id.slug()
                );
                seen += 1;
            }
        }
        assert_eq!(seen, 7 + 6 + 4 + 5 + 4 + 4 + 4 + 4);
        assert!(parse(source(Track::Stw.lessons().nth(5).expect("stw/06")))
            .iter()
            .any(|b| matches!(b, Block::Code { .. })));
        assert!(parse(source(Track::Stw.first()))
            .iter()
            .any(|b| matches!(b, Block::Callout(_))));
    }

    #[test]
    fn stw_lessons_are_complete_lay_pages() {
        for id in Track::Stw.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Code { .. })),
                "brak wzoru {}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn geo_lessons_are_complete_lay_pages() {
        for id in Track::Geo.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn stw_next_walks_to_the_seventh_lesson_then_nbody() {
        let mut id = Track::Stw.first();
        let mut hops = 0;
        loop {
            match step(id, true) {
                Action::Lesson(next) => {
                    id = next;
                    hops += 1;
                }
                Action::Lab(lab) => {
                    assert_eq!(lab, LabId::Nbody);
                    break;
                }
                Action::Map => panic!("A7 miało otworzyć N-ciała, nie mapę"),
                Action::None => panic!("krok nie może być pusty"),
            }
        }
        assert_eq!(hops, 6);
        assert_eq!(id.index, 7);
        assert_eq!(step(id, true), Action::Lab(LabId::Nbody));
        assert_eq!(step(Track::Stw.first(), false), Action::Map);
    }

    #[test]
    fn bh_lessons_are_complete_lay_pages() {
        for id in Track::Bh.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn pinn_lessons_are_complete_lay_pages() {
        for id in Track::Pinn.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn ein_lessons_are_complete_lay_pages() {
        for id in Track::Einstein.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn ten_lessons_are_complete_lay_pages() {
        for id in Track::Tensor.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
        }
    }

    #[test]
    fn kerr_lessons_are_complete_lay_pages() {
        for id in Track::Kerr.lessons() {
            let src = source(id);
            let blocks = parse(src);
            let headings: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Heading { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            for need in ["Analogia", "Co widać", "Wzór", "W kodzie"] {
                assert!(
                    headings.contains(&need),
                    "{}: brak sekcji {}",
                    id.slug(),
                    need
                );
            }
            assert!(
                blocks.iter().any(|b| matches!(b, Block::Callout(_))),
                "brak calloutu {}",
                id.slug()
            );
            let n_code = blocks
                .iter()
                .filter(|b| matches!(b, Block::Code { .. }))
                .count();
            assert_eq!(
                n_code,
                1,
                "{}: oczekiwany jeden wzór, jest {n_code}",
                id.slug()
            );
            assert!(
                src.chars().count() > 1400,
                "za krótka strona {} ({})",
                id.slug(),
                src.chars().count()
            );
            assert!(
                !src.to_lowercase().contains("cauchy"),
                "krok 27 bez Cauchy horizon {}",
                id.slug()
            );
            assert!(
                !src.contains("placeholder"),
                "krok 31: animacja zamiast placeholdera {}",
                id.slug()
            );
        }
    }

    #[test]
    fn bh_next_walks_to_the_fourth_lesson_then_raytracer() {
        let mut id = Track::Bh.first();
        let mut hops = 0;
        loop {
            match step(id, true) {
                Action::Lesson(next) => {
                    id = next;
                    hops += 1;
                }
                Action::Lab(lab) => {
                    assert_eq!(lab, LabId::BlackHole);
                    break;
                }
                Action::Map => panic!("C4 miało otworzyć raytracer, nie mapę"),
                Action::None => panic!("krok nie może być pusty"),
            }
        }
        assert_eq!(hops, 3);
        assert_eq!(id.index, 4);
        assert_eq!(step(id, true), Action::Lab(LabId::BlackHole));
        assert_eq!(step(Track::Bh.first(), false), Action::Map);
        assert_eq!(id.opens_lab(), Some(LabId::BlackHole));
        assert_eq!(
            Track::Bh.lessons().nth(2).unwrap().opens_lab(),
            None,
            "C3 otwiera lab tylko przyciskiem w obrazie"
        );
    }

    #[test]
    fn geo_next_walks_to_the_sixth_lesson_then_geodesics() {
        let mut id = Track::Geo.first();
        let mut hops = 0;
        loop {
            match step(id, true) {
                Action::Lesson(next) => {
                    id = next;
                    hops += 1;
                }
                Action::Lab(lab) => {
                    assert_eq!(lab, LabId::Geodesics);
                    break;
                }
                Action::Map => panic!("B6 miało otworzyć geodezyjne, nie mapę"),
                Action::None => panic!("krok nie może być pusty"),
            }
        }
        assert_eq!(hops, 5);
        assert_eq!(id.index, 6);
        assert_eq!(step(id, true), Action::Lab(LabId::Geodesics));
        assert_eq!(step(Track::Geo.first(), false), Action::Map);
    }

    #[test]
    fn next_tracks_walk_to_the_map_without_a_lab() {
        for track in Track::NEXT
            .iter()
            .copied()
            .chain(std::iter::once(Track::Pde))
        {
            let mut id = track.first();
            let mut hops = 0u8;
            loop {
                match step(id, true) {
                    Action::Lesson(next) => {
                        id = next;
                        hops += 1;
                    }
                    Action::Map => break,
                    Action::Lab(lab) => panic!("{} nie otwiera labu {:?}", id.slug(), lab),
                    Action::None => panic!("krok nie może być pusty"),
                }
            }
            assert_eq!(hops, track.lesson_count() - 1);
            assert_eq!(id.index, track.lesson_count());
            assert_eq!(step(id, true), Action::Map);
            assert_eq!(step(track.first(), false), Action::Map);
        }
    }

    #[test]
    fn kerr_next_walks_to_the_fourth_lesson_then_raytracer() {
        let mut id = Track::Kerr.first();
        let mut hops = 0;
        loop {
            match step(id, true) {
                Action::Lesson(next) => {
                    id = next;
                    hops += 1;
                }
                Action::Lab(lab) => {
                    assert_eq!(lab, LabId::BlackHole);
                    break;
                }
                Action::Map => panic!("Kerr 4 miało otworzyć raytracer, nie mapę"),
                Action::None => panic!("krok nie może być pusty"),
            }
        }
        assert_eq!(hops, 3);
        assert_eq!(id.index, 4);
        assert_eq!(step(id, true), Action::Lab(LabId::BlackHole));
        assert_eq!(step(Track::Kerr.first(), false), Action::Map);
        assert_eq!(id.opens_lab(), Some(LabId::BlackHole));
        assert_eq!(
            Track::Kerr.lessons().nth(2).unwrap().opens_lab(),
            None,
            "Kerr 3 otwiera lab tylko z lekcji 4"
        );
    }

    #[test]
    fn playback_advances_only_while_playing() {
        let mut p = Playback::default();
        assert!(p.playing);
        p.tick(0.5);
        assert!(p.t > 0.0);
        let frozen = p.t;
        p.playing = false;
        p.tick(1.0);
        assert_eq!(p.t, frozen);
        p.reset();
        assert!(p.playing);
        assert_eq!(p.t, 0.0);
        assert_eq!(p.beta, crate::viz::BETA_DEFAULT);
        assert_eq!(p.mass, crate::viz::MASS_DEFAULT);
        assert_eq!(p.spin, crate::viz::SPIN_DEFAULT);
        assert_eq!(p.probe, crate::viz::PROBE_DEFAULT);
        assert_eq!(p.density, 0.0);
        assert!((crate::viz::minkowski::gamma_of(p.beta) - 1.25).abs() < 1e-15);
    }

    #[test]
    fn lesson_layout_renders_stubs_without_a_window() {
        for track in Track::ALL {
            for id in track.lessons() {
                let ctx = egui::Context::default();
                ctx.begin_pass(egui::RawInput::default());
                egui::CentralPanel::default().show(&ctx, |ui| {
                    let mut playback = Playback::default();
                    let _ = draw(ui, id, &mut playback);
                });
                let _ = ctx.end_pass();
            }
        }
    }
}
