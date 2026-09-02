//! Rysowanie chmury punktów: akumulacja poświaty i mapa kolorów.
//!
//! Cząstki nie są rysowane jako punkty o własnym kolorze, a jako WKŁADY do bufora
//! jasności, który dopiero na końcu przechodzi przez mapę kolorów. Różnica jest
//! zasadnicza dla tego, co widać: przy 250 tys. cząstek na milionie pikseli
//! rysowanie punktów daje szum, w którym gęste obszary są nierozróżnialne od
//! rzadkich, bo każdy piksel i tak jest zamalowany. Akumulacja pokazuje gęstość
//! rzutowaną — czyli to, co w tych symulacjach jest treścią.
//!
//! Jasność przechodzi przez `asinh`, a nie przez logarytm ani skalę liniową.
//! Powód: w pustce wkład jest bliski zeru, a logarytm rozdmuchuje tam szum do
//! pełnej jasności. `asinh` jest liniowy przy zerze i logarytmiczny dla dużych
//! wartości, więc pustka zostaje ciemna, a halo nie zostaje przepalone.

use eframe::egui::{Color32, ColorImage};

use crate::camera::{Camera, View};
use bone_core::vec3::Vec3;

/// Co renderer musi wiedzieć o symulacji — i nic więcej.
///
/// Trait, a nie konkretny typ, bo rysowane są dwa różne modele: chmura SR (odcień
/// to prędkość względem `c`) i próbka ΛCDM (odcień to kontrast gęstości). Renderer
/// nie ma powodu wiedzieć, który to który.
pub trait PointCloud {
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn position(&self, index: usize) -> Vec3;

    /// Wartość z przedziału [0, 1] sterująca jasnością i barwą.
    fn shade(&self, index: usize) -> f32;

    /// Środek i największa rozciągłość — kamera się do nich dopasowuje.
    fn center_span(&self) -> (Vec3, f64);
}

/// Powyżej tylu rysowanych punktów obraz i tak jest wysycony, więc dalsze
/// zagęszczanie kosztuje czas i nic nie wnosi.
const MAX_DRAWN_POINTS: usize = 400_000;

const BACKGROUND: Color32 = Color32::from_rgb(6, 8, 14);

pub fn render(
    cloud: Option<&dyn PointCloud>,
    camera: &Camera,
    width: usize,
    height: usize,
) -> ColorImage {
    let width = width.clamp(64, 2560);
    let height = height.clamp(64, 1600);
    let Some(cloud) = cloud.filter(|c| !c.is_empty()) else {
        return ColorImage::new([width, height], BACKGROUND);
    };

    let (center, span) = cloud.center_span();
    let view = camera.view(center, span, width, height);
    let stride = (cloud.len() / MAX_DRAWN_POINTS).max(1);

    let mut glow = vec![0.0f32; width * height];
    for i in (0..cloud.len()).step_by(stride) {
        let Some((u, v)) = view.project(cloud.position(i)) else {
            continue;
        };
        let shade = cloud.shade(i);
        if !shade.is_finite() {
            continue;
        }
        let shade = shade.clamp(0.0, 1.0);
        // Wkład rośnie z kwadratem odcienia, więc jasne obszary wygrywają nad
        // rozlanym tłem także wtedy, gdy jest w nim więcej cząstek.
        splat(&mut glow, &view, u, v, 0.15 + 3.4 * shade * shade, shade > 0.28);
    }

    let pixels = glow.iter().map(|g| brightness_to_color(*g)).collect();
    ColorImage {
        size: [width, height],
        pixels,
    }
}

/// Wkład jednej cząstki: piksel, a dla jaśniejszych także czterej sąsiedzi.
///
/// Rozmycie tylko dla jasnych punktów, bo w pustce rozlewałoby szum na sąsiedztwo,
/// a właśnie w pustce ma być ciemno.
fn splat(glow: &mut [f32], view: &View, u: usize, v: usize, weight: f32, wide: bool) {
    let (width, _) = view.size();
    glow[v * width + u] += weight;
    if !wide {
        return;
    }
    // `project` odrzuca krawędź, więc wszyscy czterej sąsiedzi są w obrazie.
    for (du, dv) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
        let x = (u as i32 + du) as usize;
        let y = (v as i32 + dv) as usize;
        glow[y * width + x] += weight * 0.32;
    }
}

fn brightness_to_color(glow: f32) -> Color32 {
    if glow <= 1e-6 {
        return BACKGROUND;
    }
    let t = (glow * 0.22).asinh() / 2.4_f32.asinh();
    heat(t)
}

/// Mapa kolorów od granatu przez błękit i zieleń do bieli.
///
/// Cztery odcinki liniowe zamiast gotowej palety: nie potrzebuje zależności, a jej
/// kształt jest dobrany do tego obrazu — pustka schodzi do tła, a nie do czerni,
/// więc widać, że tam JEST przestrzeń, a nie brak danych.
fn heat(t: f32) -> Color32 {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [6.0, 8.0, 14.0]),
        (0.28, [28.0, 44.0, 92.0]),
        (0.58, [106.0, 140.0, 128.0]),
        (0.82, [228.0, 176.0, 60.0]),
        (1.00, [255.0, 244.0, 200.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    let mut rgb = STOPS[STOPS.len() - 1].1;
    for pair in STOPS.windows(2) {
        let (lo_t, lo) = pair[0];
        let (hi_t, hi) = pair[1];
        if t <= hi_t {
            let u = (t - lo_t) / (hi_t - lo_t);
            rgb = [
                lo[0] + (hi[0] - lo[0]) * u,
                lo[1] + (hi[1] - lo[1]) * u,
                lo[2] + (hi[2] - lo[2]) * u,
            ];
            break;
        }
    }
    Color32::from_rgb(
        rgb[0].round().clamp(0.0, 255.0) as u8,
        rgb[1].round().clamp(0.0, 255.0) as u8,
        rgb[2].round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::vec3::{vec3, ZERO};

    struct Cloud {
        points: Vec<Vec3>,
        shades: Vec<f32>,
    }

    impl Cloud {
        fn line(n: usize, shade: f32) -> Self {
            Self {
                points: (0..n)
                    .map(|i| vec3(i as f64 / n as f64 - 0.5, 0.0, 0.0))
                    .collect(),
                shades: vec![shade; n],
            }
        }
    }

    impl PointCloud for Cloud {
        fn len(&self) -> usize {
            self.points.len()
        }
        fn position(&self, index: usize) -> Vec3 {
            self.points[index]
        }
        fn shade(&self, index: usize) -> f32 {
            self.shades[index]
        }
        fn center_span(&self) -> (Vec3, f64) {
            bone_core::grid::center_span(&self.points)
        }
    }

    fn lit_pixels(image: &ColorImage) -> usize {
        image.pixels.iter().filter(|p| **p != BACKGROUND).count()
    }

    #[test]
    fn empty_scene_is_all_background() {
        let image = render(None, &Camera::default(), 64, 64);
        assert_eq!(image.size, [64, 64]);
        assert_eq!(lit_pixels(&image), 0);

        let empty = Cloud::line(0, 0.5);
        assert_eq!(lit_pixels(&render(Some(&empty), &Camera::default(), 64, 64)), 0);
    }

    #[test]
    fn particles_light_up_pixels() {
        let cloud = Cloud::line(500, 0.6);
        let image = render(Some(&cloud), &Camera::default(), 240, 160);
        assert!(lit_pixels(&image) > 10, "narysowano {}", lit_pixels(&image));
    }

    /// Jaśniejsza chmura musi dawać jaśniejszy obraz. Bez tego mapa jasności mogłaby
    /// być stała i nikt by tego nie zauważył, bo obrazek nadal coś pokazuje.
    #[test]
    fn brighter_shade_gives_a_brighter_image() {
        let camera = Camera::default();
        let dim = render(Some(&Cloud::line(200, 0.05)), &camera, 200, 200);
        let bright = render(Some(&Cloud::line(200, 0.95)), &camera, 200, 200);
        let sum = |img: &ColorImage| -> u32 {
            img.pixels
                .iter()
                .map(|p| p.r() as u32 + p.g() as u32 + p.b() as u32)
                .sum()
        };
        assert!(sum(&bright) > sum(&dim), "{} vs {}", sum(&bright), sum(&dim));
    }

    #[test]
    fn palette_is_monotone_in_brightness() {
        let mut previous = 0u32;
        for i in 0..=20 {
            let c = heat(i as f32 / 20.0);
            let luma = c.r() as u32 + c.g() as u32 + c.b() as u32;
            assert!(luma >= previous, "t={} luma spadła", i as f32 / 20.0);
            previous = luma;
        }
    }

    #[test]
    fn palette_ends_are_clamped() {
        assert_eq!(heat(-5.0), heat(0.0));
        assert_eq!(heat(5.0), heat(1.0));
    }

    #[test]
    fn zero_brightness_is_background() {
        assert_eq!(brightness_to_color(0.0), BACKGROUND);
        assert_ne!(brightness_to_color(1.0), BACKGROUND);
    }

    /// Renderer nie może się wywrócić na absurdalnym rozmiarze okna ani na chmurze
    /// wypchniętej w nieskończoność — jedno bywa stanem okna, drugie rozbiegnięciem.
    #[test]
    fn survives_degenerate_input() {
        let camera = Camera::default();
        let image = render(Some(&Cloud::line(10, 0.5)), &camera, 0, 0);
        assert_eq!(image.size, [64, 64]);

        let broken = Cloud {
            points: vec![vec3(f64::NAN, 0.0, 0.0), ZERO],
            shades: vec![f32::NAN, 0.5],
        };
        render(Some(&broken), &camera, 80, 80);
    }

    /// Duża chmura jest przerzedzana, ale nie pomijana. Gdyby krok liczył się źle,
    /// zniknęłaby całkiem albo rysowałaby się w całości i zwieszała panel.
    #[test]
    fn large_clouds_are_thinned_not_dropped() {
        let cloud = Cloud::line(MAX_DRAWN_POINTS * 3, 0.7);
        let image = render(Some(&cloud), &Camera::default(), 200, 200);
        assert!(lit_pixels(&image) > 10);
    }
}
