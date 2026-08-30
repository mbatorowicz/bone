//! Okno desktop: panel + rzut chmury na GPU (wgpu przez eframe).

use std::time::Instant;

use eframe::egui::{self, Color32, ColorImage, TextureHandle, TextureOptions};
use egui::{RichText, Sense};

use crate::cosmology::Cosmology;
use crate::engine::{adaptive_dlna, Engine, RunConfig};

pub struct CosmoApp {
    engine: Option<Engine>,
    running: bool,
    preset: Preset,
    n_grid: usize,
    box_size: f64,
    dlna: f64,
    seed: u64,
    speed: u32,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    drag: Option<egui::Pos2>,
    texture: Option<TextureHandle>,
    last_frame: Instant,
    ms_step: f64,
    status: String,
    error: String,
    slow_hold: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Preset {
    Linear,
    Small,
    Medium,
}

impl Preset {
    fn label(self) -> &'static str {
        match self {
            Self::Linear => "wzrost liniowy",
            Self::Small => "formacja struktur",
            Self::Medium => "formacja 64³",
        }
    }

    fn config(self) -> RunConfig {
        match self {
            Self::Linear => RunConfig::linear(),
            Self::Small => RunConfig::planck18_small(),
            Self::Medium => RunConfig::planck18_64(),
        }
    }
}

impl Default for CosmoApp {
    fn default() -> Self {
        Self {
            engine: None,
            running: false,
            preset: Preset::Small,
            n_grid: 48,
            box_size: 32.0,
            dlna: 0.0005,
            seed: 42,
            speed: 1,
            yaw: 0.7,
            pitch: 0.35,
            zoom: 1.0,
            drag: None,
            texture: None,
            last_frame: Instant::now(),
            ms_step: 0.0,
            status: "Gotowy. Wybierz preset i uruchom — liczy karta / CPU tego komputera."
                .into(),
            error: String::new(),
            slow_hold: 0,
        }
    }
}

impl CosmoApp {
    fn spawn(&mut self) {
        self.error.clear();
        let mut cfg = self.preset.config();
        cfg.n_grid = self.n_grid;
        cfg.pm_grid = self.n_grid.min(64);
        cfg.box_size = self.box_size;
        cfg.dlna = self.dlna;
        cfg.seed = self.seed;
        self.status = format!("Składanie IC {}³…", cfg.n_grid);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Engine::new(Cosmology::planck18(), cfg)
        })) {
            Ok(eng) => {
                self.status = format!(
                    "Start z={:.1}  wiek={:.2} Gyr  N={}  próbka {} Mpc/h  przestrzeń otwarta",
                    eng.redshift(),
                    eng.cosmology.age_gyr(eng.a),
                    eng.n(),
                    eng.box_size
                );
                self.engine = Some(eng);
                self.running = true;
            }
            Err(_) => {
                self.error = "Nie udało się złożyć warunków początkowych (za mało RAM?).".into();
                self.running = false;
            }
        }
    }

    fn tick(&mut self) {
        let Some(eng) = self.engine.as_mut() else {
            return;
        };
        if !self.running {
            return;
        }
        eng.cfg.dlna = self.dlna;
        let n = if self.speed == 0 {
            self.slow_hold += 1;
            if self.slow_hold < 8 {
                return;
            }
            self.slow_hold = 0;
            1
        } else {
            self.slow_hold = 0;
            self.speed
        };
        let t0 = Instant::now();
        for _ in 0..n {
            eng.step();
        }
        self.ms_step = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let z = eng.redshift();
        self.status = format!(
            "z={:.3}  wiek={:.2} Gyr  a={:.4}  krok={}  {}×  Δln a={:.4}  Δt={:.0} ms  LI={:+.2e}",
            z,
            eng.cosmology.age_gyr(eng.a),
            eng.a,
            eng.step,
            n,
            adaptive_dlna(z, self.dlna),
            self.ms_step,
            eng.layzer_irvine()
        );
    }
}

impl eframe::App for CosmoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.running {
            self.tick();
            ctx.request_repaint();
        }

        egui::SidePanel::left("panel")
            .resizable(false)
            .min_width(280.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.label(RichText::new("BONE COSMO").size(22.0).strong());
                ui.label("ΛCDM · Planck 2018 · PM izolowany (Hockney)");
                ui.label(RichText::new("Model standardowy kosmologii").small().weak());
                ui.separator();

                ui.label("Preset");
                ui.horizontal_wrapped(|ui| {
                    for p in [Preset::Linear, Preset::Small, Preset::Medium] {
                        if ui.selectable_label(self.preset == p, p.label()).clicked() {
                            self.preset = p;
                            let c = p.config();
                            self.n_grid = c.n_grid;
                            self.box_size = c.box_size;
                            self.dlna = c.dlna;
                            self.seed = c.seed;
                        }
                    }
                });

                ui.add_space(6.0);
                ui.add(egui::Slider::new(&mut self.n_grid, 16..=64).text("siatka N³"));
                ui.add(
                    egui::Slider::new(&mut self.box_size, 20.0..=400.0).text("próbka [Mpc/h]"),
                );
                ui.add(egui::Slider::new(&mut self.dlna, 0.0002..=0.01).text("Δln a"));
                ui.add(
                    egui::Slider::new(&mut self.speed, 0..=16)
                        .text("szybkość")
                        .suffix("×"),
                );
                ui.label(
                    RichText::new(
                        "Przestrzeń otwarta — bez ścian i zawijania. Widok śledzi chmurę. \
                         Tempo zależy od z: szybciej przy z>20, wolniej przy z<5. Zero na suwaku = jeszcze wolniej.",
                    )
                    .small()
                    .weak(),
                );

                ui.horizontal(|ui| {
                    if ui
                        .add_sized([90.0, 28.0], egui::Button::new("Uruchom"))
                        .clicked()
                    {
                        self.spawn();
                    }
                    if ui
                        .button(if self.running { "Pauza" } else { "Wznów" })
                        .clicked()
                    {
                        if self.engine.is_some() {
                            self.running = !self.running;
                        }
                    }
                    if ui.button("Stop").clicked() {
                        self.running = false;
                        self.engine = None;
                        self.status = "Zatrzymano.".into();
                    }
                });

                ui.separator();
                if let Some(eng) = &self.engine {
                    let (t, u) = eng.energies();
                    ui.monospace(format!("z        {:>8.3}", eng.redshift()));
                    ui.monospace(format!("a        {:>8.4}", eng.a));
                    ui.monospace(format!("wiek     {:>6.2} Gyr", eng.cosmology.age_gyr(eng.a)));
                    ui.monospace(format!("N        {:>8}", eng.n()));
                    ui.monospace(format!("T        {:>10.3e}", t));
                    ui.monospace(format!("U        {:>10.3e}", u));
                    ui.monospace(format!("LI       {:>+10.2e}", eng.layzer_irvine()));
                    ui.monospace(format!(
                        "Ωm={:.3}  h={:.4}  σ₈={:.3}",
                        eng.cosmology.omega_m, eng.cosmology.h, eng.cosmology.sigma8
                    ));
                } else {
                    let c = Cosmology::planck18();
                    ui.label(format!(
                        "Planck18  Ωm={:.3}  Ωb={:.3}  ΩΛ={:.3}\nh={:.4}  ns={:.3}  σ₈={:.3}\nwiek dziś {:.2} Gyr",
                        c.omega_m,
                        c.omega_b,
                        c.omega_l,
                        c.h,
                        c.n_s,
                        c.sigma8,
                        c.age_gyr(1.0)
                    ));
                }

                ui.separator();
                ui.label(RichText::new(&self.status).small());
                if !self.error.is_empty() {
                    ui.colored_label(Color32::from_rgb(220, 80, 80), &self.error);
                }
                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "Silnik na tym komputerze (izolowany PM + FFT, bez periodycznego pudła). \
                         Widok śledzi chmurę. Nie liczy w chmurze.",
                    )
                    .small()
                    .weak(),
                );
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
            if response.dragged() {
                if let Some(prev) = self.drag {
                    let d = response.interact_pointer_pos().unwrap_or(prev) - prev;
                    self.yaw += d.x * 0.008;
                    self.pitch = (self.pitch + d.y * 0.008).clamp(-1.2, 1.2);
                }
                self.drag = response.interact_pointer_pos();
            } else {
                self.drag = None;
            }
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                self.zoom = (self.zoom * (1.0 - scroll * 0.001)).clamp(0.15, 8.0);
            }

            let image = render_cloud(
                self.engine.as_ref(),
                self.yaw,
                self.pitch,
                self.zoom,
                rect.width().max(8.0) as usize,
                rect.height().max(8.0) as usize,
            );
            let tex = self.texture.get_or_insert_with(|| {
                ui.ctx()
                    .load_texture("cloud", image.clone(), TextureOptions::LINEAR)
            });
            tex.set(image, TextureOptions::LINEAR);
            ui.put(rect, egui::Image::new(&*tex).fit_to_exact_size(rect.size()));
        });

        let _ = self.last_frame.elapsed();
        self.last_frame = Instant::now();
    }
}

fn render_cloud(
    engine: Option<&Engine>,
    yaw: f32,
    pitch: f32,
    zoom: f32,
    w: usize,
    h: usize,
) -> ColorImage {
    let w = w.clamp(64, 1400);
    let h = h.clamp(64, 900);
    let mut pixels = vec![Color32::from_rgb(6, 8, 14); w * h];
    let Some(eng) = engine else {
        return ColorImage::from_rgba_unmultiplied(
            [w, h],
            &flatten(&pixels),
        );
    };
    let (center, span) = eng.cloud_center_span();
    let half = (span * 0.5).max(1e-3);
    let cy = yaw.cos();
    let sy = yaw.sin();
    let cp = pitch.cos();
    let sp = pitch.sin();
    let stride = (eng.n() / 400_000).max(1);
    let scale = (w.min(h) as f32) * 0.42 * zoom;
    let cx = w as f32 * 0.5;
    let cy_pix = h as f32 * 0.5;
    let mut glow = vec![0.0f32; w * h];

    for (i, q) in eng.x.iter().enumerate().step_by(stride) {
        let px = (q[0] - center[0]) / half;
        let py = (q[1] - center[1]) / half;
        let pz = (q[2] - center[2]) / half;
        let x1 = px * cy + pz * sy;
        let z1 = -px * sy + pz * cy;
        let y1 = py * cp - z1 * sp;
        let u = (cx + x1 * scale) as i32;
        let v = (cy_pix - y1 * scale) as i32;
        if u < 1 || v < 1 || u >= w as i32 - 1 || v >= h as i32 - 1 {
            continue;
        }
        let s = eng.shade(i);
        let weight = 0.15 + 3.4 * s * s;
        splat_glow(&mut glow, w, h, u, v, weight, s > 0.28);
    }

    for (idx, g) in glow.iter().enumerate() {
        if *g <= 1e-6 {
            continue;
        }
        let t = (g * 0.22).asinh() / 2.4_f32.asinh();
        pixels[idx] = heat_color(t);
    }

    ColorImage::from_rgba_unmultiplied([w, h], &flatten(&pixels))
}

fn splat_glow(glow: &mut [f32], w: usize, h: usize, u: i32, v: i32, weight: f32, wide: bool) {
    let idx = v as usize * w + u as usize;
    glow[idx] += weight;
    if !wide {
        return;
    }
    for (dx, dy, k) in [(-1, 0, 0.32), (1, 0, 0.32), (0, -1, 0.32), (0, 1, 0.32)] {
        let x = u + dx;
        let y = v + dy;
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            glow[y as usize * w + x as usize] += weight * k;
        }
    }
}

fn heat_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 0.28 {
        let u = t / 0.28;
        (6.0 + 22.0 * u, 8.0 + 36.0 * u, 14.0 + 78.0 * u)
    } else if t < 0.58 {
        let u = (t - 0.28) / 0.30;
        (28.0 + 78.0 * u, 44.0 + 96.0 * u, 92.0 + 36.0 * u)
    } else if t < 0.82 {
        let u = (t - 0.58) / 0.24;
        (106.0 + 122.0 * u, 140.0 + 36.0 * u, 128.0 - 68.0 * u)
    } else {
        let u = (t - 0.82) / 0.18;
        (228.0 + 27.0 * u, 176.0 + 68.0 * u, 60.0 + 140.0 * u)
    };
    Color32::from_rgb(
        r.round().clamp(0.0, 255.0) as u8,
        g.round().clamp(0.0, 255.0) as u8,
        b.round().clamp(0.0, 255.0) as u8,
    )
}

fn flatten(px: &[Color32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len() * 4);
    for c in px {
        let [r, g, b, a] = c.to_array();
        out.extend_from_slice(&[r, g, b, a]);
    }
    out
}

pub fn run() -> eframe::Result<()> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([900.0, 600.0])
            .with_title("Bone Cosmo — ΛCDM"),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Bone Cosmo",
        opts,
        Box::new(|_cc| Ok(Box::new(CosmoApp::default()))),
    )
}
