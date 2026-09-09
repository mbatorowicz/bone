//! Wykresy funkcji falowych i poziomów energii — czysta geometria egui.
//!
//! Panel nie liczy fizyki. Dostaje gotowe serie z `bone_core::qm::plot` i rysuje
//! je. Dzięki temu ten sam zestaw punktów można sprawdzić testem bez okna.

use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Ui};

use bone_core::qm::plot::{Charts, Level, Series};

pub fn atom_charts(ui: &mut Ui, charts: &Charts) {
    ui.add_space(6.0);
    ui.label(egui::RichText::new("Funkcje").small().strong());
    ui.label(egui::RichText::new(charts.formula).small().weak().italics());
    ui.add_space(4.0);
    ui.label(egui::RichText::new("P(r) = r² R²").small());
    line_plot(ui, &charts.probability, 72.0, Color32::from_rgb(120, 180, 220));
    ui.label(egui::RichText::new("Rₙₗ(r)").small());
    line_plot(ui, &charts.radial, 56.0, Color32::from_rgb(220, 180, 90));
    ui.label(egui::RichText::new("|Yₗₘ(θ)|²  w płaszczyźnie xz").small());
    line_plot(ui, &charts.angular, 48.0, Color32::from_rgb(160, 200, 140));
    if !charts.levels.is_empty() {
        ui.label(egui::RichText::new("poziomy Eₙ").small());
        energy_ladder(ui, &charts.levels, 80.0);
    }
    if !charts.transitions.is_empty() {
        ui.add_space(2.0);
        for t in charts.transitions.iter().take(6) {
            ui.label(
                egui::RichText::new(format!(
                    "{}  n={}→{}  {:.1} nm  ({:.2} eV)",
                    t.series, t.from_n, t.to_n, t.wavelength_nm, t.delta_ev
                ))
                .small()
                .weak()
                .monospace(),
            );
        }
    }
}

fn line_plot(ui: &mut Ui, series: &Series, height: f32, color: Color32) {
    if series.xs.len() < 2 || series.xs.len() != series.ys.len() {
        return;
    }
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(12, 16, 24));
    let (x0, x1) = min_max(&series.xs);
    let (y0, y1) = min_max(&series.ys);
    let dx = (x1 - x0).max(1e-12);
    let dy = (y1 - y0).max(1e-12);
    let pad = 4.0;
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + pad, rect.top() + pad),
        Pos2::new(rect.right() - pad, rect.bottom() - pad),
    );
    if y0 < 0.0 && y1 > 0.0 {
        let t = ((0.0 - y0) / dy) as f32;
        let y = inner.bottom() - t * inner.height();
        painter.line_segment(
            [Pos2::new(inner.left(), y), Pos2::new(inner.right(), y)],
            Stroke::new(1.0, Color32::from_rgb(40, 48, 60)),
        );
    }
    let mut pts: Vec<Pos2> = Vec::with_capacity(series.xs.len());
    for (x, y) in series.xs.iter().zip(series.ys.iter()) {
        let u = ((*x - x0) / dx) as f32;
        let v = ((*y - y0) / dy) as f32;
        pts.push(Pos2::new(
            inner.left() + u * inner.width(),
            inner.bottom() - v * inner.height(),
        ));
    }
    painter.add(egui::Shape::line(pts, Stroke::new(1.4, color)));
}

fn energy_ladder(ui: &mut Ui, levels: &[Level], height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), height), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(12, 16, 24));
    let energies: Vec<f64> = levels.iter().map(|l| l.e_ev).collect();
    let (e0, e1) = min_max(&energies);
    let de = (e1 - e0).abs().max(1e-6);
    let pad = 8.0;
    let inner = Rect::from_min_max(
        Pos2::new(rect.left() + pad, rect.top() + pad),
        Pos2::new(rect.right() - pad, rect.bottom() - pad),
    );
    for level in levels {
        let t = ((level.e_ev - e0) / de) as f32;
        let y = inner.bottom() - t * inner.height();
        let occupied = level.occupied > 0;
        let color = if occupied {
            Color32::from_rgb(228, 176, 60)
        } else {
            Color32::from_rgb(80, 100, 140)
        };
        painter.line_segment(
            [Pos2::new(inner.left() + 28.0, y), Pos2::new(inner.right() - 4.0, y)],
            Stroke::new(if occupied { 1.8 } else { 1.0 }, color),
        );
        painter.text(
            Pos2::new(inner.left(), y),
            egui::Align2::LEFT_CENTER,
            format!("n={}", level.n),
            egui::FontId::monospace(9.0),
            color,
        );
    }
}

fn min_max(values: &[f64]) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for v in values {
        if v.is_finite() {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
    }
    if !lo.is_finite() {
        (0.0, 1.0)
    } else if (hi - lo).abs() < 1e-18 {
        (lo - 1.0, hi + 1.0)
    } else {
        (lo, hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_handles_a_spike_and_a_flatline() {
        assert_eq!(min_max(&[1.0, 3.0, 2.0]), (1.0, 3.0));
        let (lo, hi) = min_max(&[5.0, 5.0]);
        assert!(hi > lo);
        let (lo, hi) = min_max(&[]);
        assert_eq!((lo, hi), (0.0, 1.0));
    }
}
