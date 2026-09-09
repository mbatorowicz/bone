//! Panel boczny: wybór modelu, parametry startowe, przyciski i tabela pomiarów.
//!
//! Panel nie dotyka silnika. Zmienia [`Setup`] i zwraca [`Action`], a decyzję, co
//! z tym zrobić, podejmuje [`crate::App`]. Dzięki temu nie da się przypadkiem
//! przebudować symulacji w środku rysowania klatki — a to była najłatwiejsza droga
//! do zawieszenia panelu na kilka sekund bez żadnego komunikatu.

use eframe::egui::{self, Color32, RichText, Ui};

use bone_core::lcdm;
use bone_core::qm;
use bone_core::qm::config::Scene;
use bone_core::qm::elements;
use bone_core::qm::hydrogen::{orbital_label, spectroscopic};
use bone_core::sm;
use bone_core::sm::config::Shape;
use bone_core::sr;
use bone_core::sr::config::{BackendKind, Geometry};
use bone_core::io::checkpoint;
use crate::charts;
use crate::simulation::{Mode, View};

/// Nastawy formularza — to, co widzi użytkownik, zanim wciśnie „Uruchom".
pub struct Setup {
    pub mode: Mode,
    pub sr: sr::Config,
    pub sr_preset: &'static str,
    pub lcdm: lcdm::RunConfig,
    pub lcdm_preset: lcdm::Preset,
    pub sm: sm::Config,
    pub sm_preset: &'static str,
    pub qm: qm::Config,
    pub qm_preset: &'static str,
    /// Ile kroków symulacji na jedną klatkę; 0 znaczy „jeszcze wolniej".
    pub speed: u32,
    pub out_dir: String,
    pub record: bool,
}

impl Default for Setup {
    fn default() -> Self {
        Self {
            mode: Mode::Cosmological,
            sr: sr::presets::galaxy(),
            sr_preset: "galaxy",
            lcdm: lcdm::RunConfig::structure(),
            lcdm_preset: lcdm::Preset::Structure,
            sm: sm::presets::plazma(),
            sm_preset: "plazma",
            qm: qm::presets::wodor(),
            qm_preset: "wodor",
            speed: 1,
            out_dir: "runs/latest".to_string(),
            record: false,
        }
    }
}


/// Czego panel chce od aplikacji.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    None,
    Start,
    TogglePause,
    Stop,
    Resume,
    Replay,
}

pub fn side_panel(
    ui: &mut Ui,
    setup: &mut Setup,
    mut view: Option<&mut View>,
    running: bool,
    status: &str,
    error: &str,
    warnings: &[String],
) -> Action {
    ui.add_space(8.0);
    ui.label(RichText::new("BONE").size(22.0).strong());
    ui.label(setup.mode.subtitle());
    ui.separator();

    ui.label("Model");
    ui.horizontal_wrapped(|ui| {
        for mode in Mode::ALL {
            if ui
                .selectable_label(setup.mode == mode, mode.label())
                .clicked()
            {
                setup.mode = mode;
            }
        }
    });
    ui.add_space(6.0);
    model_card(ui, setup.mode);
    ui.add_space(6.0);

    match setup.mode {
        Mode::Relativistic => relativistic_form(ui, setup),
        Mode::Cosmological => {
            cosmological_form(ui, setup);
            let chart = view
                .as_ref()
                .and_then(|v| v.lcdm_growth())
                .unwrap_or_else(|| {
                    lcdm::growth::Chart::theoretical(lcdm::Cosmology::planck18(), setup.lcdm.z_start)
                });
            charts::cosmology_charts(ui, &chart);
        }
        Mode::Particles => particles_form(ui, setup),
        Mode::Atoms => atoms_form(ui, setup),
    }

    ui.add(
        egui::Slider::new(&mut setup.speed, 0..=16)
            .text("szybkość")
            .suffix("×"),
    );

    io_form(ui, setup);

    let has_checkpoint = checkpoint::exists(&setup.out_dir);
    let action = buttons(ui, view.as_ref().is_some(), running, has_checkpoint);

    if let Some(View::Replay(replay)) = view.as_mut() {
        let mut cursor = replay.cursor() as u32;
        let last = replay.n_frames().saturating_sub(1) as u32;
        if ui
            .add(egui::Slider::new(&mut cursor, 0..=last.max(1)).text("klatka"))
            .changed()
        {
            replay.seek(cursor as usize);
        }
    }

    ui.separator();
    if let Some(view) = view.as_ref() {
        egui::Grid::new("measurements")
            .num_columns(2)
            .spacing([12.0, 2.0])
            .show(ui, |ui| {
                for (name, value) in view.rows() {
                    ui.monospace(name);
                    ui.monospace(value);
                    ui.end_row();
                }
            });
    } else {
        ui.label(RichText::new(setup_summary(setup)).small().weak());
    }

    ui.separator();
    ui.label(RichText::new(status).small());
    if !error.is_empty() {
        ui.colored_label(Color32::from_rgb(220, 80, 80), error);
    }
    for warning in warnings {
        ui.colored_label(Color32::from_rgb(220, 180, 80), format!("⚠ {warning}"));
    }
    action
}

fn model_card(ui: &mut Ui, mode: Mode) {
    let card = mode.card();
    ui.group(|ui| {
        egui::Grid::new("model-card")
            .num_columns(2)
            .spacing([12.0, 2.0])
            .show(ui, |ui| {
                for (name, value) in [
                    ("Równanie", card.equation),
                    ("Zakres", card.scope),
                    ("Porównanie", card.comparison),
                    ("To nie jest", card.not_this),
                ] {
                    ui.label(RichText::new(name).small().weak());
                    ui.label(RichText::new(value).small());
                    ui.end_row();
                }
            });
    });
}

fn io_form(ui: &mut Ui, setup: &mut Setup) {
    ui.add_space(6.0);
    ui.label("Zapis");
    ui.add(egui::TextEdit::singleline(&mut setup.out_dir).desired_width(260.0));
    ui.checkbox(&mut setup.record, "Nagrywaj trajektorię i checkpoint");
}

fn buttons(ui: &mut Ui, has_view: bool, running: bool, has_checkpoint: bool) -> Action {
    let mut action = Action::None;
    ui.horizontal(|ui| {
        if ui
            .add_sized([90.0, 28.0], egui::Button::new("Uruchom"))
            .clicked()
        {
            action = Action::Start;
        }
        let pause_label = if running { "Pauza" } else { "Wznów" };
        if ui
            .add_enabled(has_view, egui::Button::new(pause_label))
            .clicked()
        {
            action = Action::TogglePause;
        }
        if ui.add_enabled(has_view, egui::Button::new("Stop")).clicked() {
            action = Action::Stop;
        }
    });
    ui.horizontal(|ui| {
        if ui
            .add_enabled(has_checkpoint, egui::Button::new("Wznów z pliku"))
            .clicked()
        {
            action = Action::Resume;
        }
        if ui.button("Odtwórz").clicked() {
            action = Action::Replay;
        }
    });
    action
}

fn relativistic_form(ui: &mut Ui, setup: &mut Setup) {
    ui.label("Zestaw nastaw");
    ui.horizontal_wrapped(|ui| {
        for id in sr::presets::ids() {
            if ui.selectable_label(setup.sr_preset == id, id).clicked() {
                if let Some(cfg) = sr::presets::preset(id) {
                    setup.sr = cfg;
                    setup.sr_preset = id;
                }
            }
        }
    });
    ui.add_space(6.0);

    egui::ComboBox::from_label("kształt")
        .selected_text(setup.sr.spawn.geometry.label())
        .show_ui(ui, |ui| {
            for geometry in Geometry::ALL {
                ui.selectable_value(&mut setup.sr.spawn.geometry, geometry, geometry.label());
            }
        });
    egui::ComboBox::from_label("solver")
        .selected_text(setup.sr.solver.backend.slug())
        .show_ui(ui, |ui| {
            for backend in BackendKind::ALL {
                ui.selectable_value(&mut setup.sr.solver.backend, backend, backend.slug());
            }
        });

    ui.add(
        egui::Slider::new(&mut setup.sr.spawn.n_particles, 100..=200_000)
            .logarithmic(true)
            .text("cząstek N"),
    );
    ui.add(egui::Slider::new(&mut setup.sr.spawn.radius, 0.5..=50.0).text("promień"));
    ui.add(
        egui::Slider::new(&mut setup.sr.spawn.total_mass, 1.0..=1e6)
            .logarithmic(true)
            .text("masa układu"),
    );
    ui.add(egui::Slider::new(&mut setup.sr.spawn.rotation, 0.0..=1.2).text("obrót"));
    if setup.sr.spawn.geometry.uses_thickness() {
        ui.add(egui::Slider::new(&mut setup.sr.spawn.thickness, 0.01..=0.5).text("grubość"));
    }
    ui.add(
        egui::Slider::new(&mut setup.sr.physics.cooling_rate, 0.0..=10.0)
            .text("chłodzenie 1/t"),
    );
}

fn cosmological_form(ui: &mut Ui, setup: &mut Setup) {
    ui.label("Zestaw nastaw");
    ui.horizontal_wrapped(|ui| {
        for preset in lcdm::Preset::ALL {
            if ui
                .selectable_label(setup.lcdm_preset == preset, preset.label())
                .clicked()
            {
                setup.lcdm_preset = preset;
                setup.lcdm = preset.config();
            }
        }
    });
    ui.add_space(6.0);

    ui.add(egui::Slider::new(&mut setup.lcdm.n_grid, 16..=64).text("siatka N³"));
    ui.add(
        egui::Slider::new(&mut setup.lcdm.box_size, 20.0..=lcdm::MAX_BOX_MPC_H)
            .text("próbka [Mpc/h]"),
    );
    ui.add(egui::Slider::new(&mut setup.lcdm.z_start, 20.0..=120.0).text("z startowe"));
    ui.add(
        egui::Slider::new(&mut setup.lcdm.dlna, 0.0002..=0.01)
            .logarithmic(true)
            .text("Δln a"),
    );
    // Siatka PM powyżej 64³ przestaje się opłacać: koszt periodycznego FFT
    // rośnie jak N³·log N, a rozdzielczość ogranicza i tak liczba cząstek.
    setup.lcdm.pm_grid = setup.lcdm.n_grid.min(64);
}

fn particles_form(ui: &mut Ui, setup: &mut Setup) {
    ui.label("Zestaw nastaw");
    ui.horizontal_wrapped(|ui| {
        for preset in sm::presets::PRESETS {
            if ui
                .selectable_label(setup.sm_preset == preset.id, preset.id)
                .clicked()
            {
                setup.sm = (preset.build)();
                setup.sm_preset = preset.id;
            }
        }
    });
    ui.add_space(6.0);

    egui::ComboBox::from_label("kształt")
        .selected_text(setup.sm.spawn.shape.label())
        .show_ui(ui, |ui| {
            for shape in Shape::ALL {
                ui.selectable_value(&mut setup.sm.spawn.shape, shape, shape.label());
            }
        });
    egui::ComboBox::from_label("solver")
        .selected_text(setup.sm.forces.backend.slug())
        .show_ui(ui, |ui| {
            for backend in BackendKind::ALL {
                ui.selectable_value(&mut setup.sm.forces.backend, backend, backend.slug());
            }
        });

    let mut n = setup.sm.spawn.total_count();
    if ui
        .add(
            egui::Slider::new(&mut n, 2..=20_000)
                .logarithmic(true)
                .text("cząstek N"),
        )
        .changed()
    {
        setup.sm.spawn.scale_to(n);
    }
    ui.add(
        egui::Slider::new(&mut setup.sm.spawn.radius, 0.1..=1.0e6)
            .logarithmic(true)
            .text("promień [fm]"),
    );
    match setup.sm.spawn.shape {
        Shape::Ball => {
            ui.add(
                egui::Slider::new(&mut setup.sm.spawn.temperature, 0.0..=10.0)
                    .text("T [MeV]"),
            );
        }
        Shape::Beam | Shape::Pair => {
            ui.add(
                egui::Slider::new(&mut setup.sm.spawn.beam_energy, 0.0..=1.0e5)
                    .text("energia wiązki [MeV]"),
            );
        }
    }

    ui.add_space(4.0);
    ui.checkbox(&mut setup.sm.decay.enabled, "rozpady");
    ui.checkbox(&mut setup.sm.decay.annihilation, "anihilacja");
    ui.checkbox(&mut setup.sm.run.adaptive, "krok adaptacyjny");

    ui.label(
        RichText::new(mixture_text(&setup.sm))
            .small()
            .weak(),
    );
}

fn atoms_form(ui: &mut Ui, setup: &mut Setup) {
    ui.label("Zestaw nastaw");
    ui.horizontal_wrapped(|ui| {
        for preset in qm::presets::PRESETS {
            if ui
                .selectable_label(setup.qm_preset == preset.id, preset.id)
                .clicked()
            {
                setup.qm = (preset.build)();
                setup.qm_preset = preset.id;
            }
        }
    });
    ui.add_space(6.0);

    ui.horizontal_wrapped(|ui| {
        let z = setup.qm.scene.z();
        if ui
            .selectable_label(matches!(setup.qm.scene, Scene::Orbital { .. }), "orbital")
            .clicked()
        {
            let (_, n, l, m) = setup.qm.scene.primary_orbital();
            setup.qm.scene = Scene::Orbital { z, n, l, m };
        }
        if ui
            .selectable_label(matches!(setup.qm.scene, Scene::Atom { .. }), "atom")
            .clicked()
        {
            setup.qm.scene = Scene::Atom { z };
        }
        if ui
            .selectable_label(
                matches!(setup.qm.scene, Scene::Superposition { .. }),
                "superpozycja",
            )
            .clicked()
        {
            setup.qm.scene = Scene::Superposition {
                z,
                terms: vec![
                    qm::Term::real(1, 0, 0, 1.0),
                    qm::Term::real(2, 1, 0, 1.0),
                ],
            };
        }
    });

    match &mut setup.qm.scene {
        Scene::Orbital { z, n, l, m } => {
            ui.add(egui::Slider::new(z, 1..=18).text("Z jądra"));
            ui.add(egui::Slider::new(n, 1..=8).text("n"));
            let n_now = *n;
            if *l >= n_now {
                *l = n_now.saturating_sub(1);
            }
            ui.add(egui::Slider::new(l, 0..=n_now.saturating_sub(1)).text("l"));
            let cap = *l as i32;
            *m = (*m).clamp(-cap, cap);
            ui.add(egui::Slider::new(m, -cap..=cap).text("m"));
            ui.label(
                RichText::new(format!(
                    "{}  ·  l={} ({})  ·  dokładny Schrödinger",
                    orbital_label(*n, *l, *m),
                    l,
                    spectroscopic(*l)
                ))
                .small()
                .weak(),
            );
        }
        Scene::Atom { z } => {
            ui.label("Pierwiastek");
            ui.horizontal_wrapped(|ui| {
                for el in elements::TABLE {
                    if ui.selectable_label(*z == el.z, el.symbol).clicked() {
                        *z = el.z;
                    }
                }
            });
            let el = elements::nearest(*z);
            ui.label(
                RichText::new(format!(
                    "{}  Z={}  {}  ·  IE NIST {:.2} eV",
                    el.name,
                    el.z,
                    el.configuration_text(),
                    el.ionization_ev
                ))
                .small()
                .weak(),
            );
        }
        Scene::Superposition { z, terms } => {
            ui.add(egui::Slider::new(z, 1..=8).text("Z jądra"));
            if terms.len() >= 2 {
                let text = format!(
                    "{} + {}  ·  gęstość bije, gdy n się różnią",
                    terms[0].label(),
                    terms[1].label()
                );
                ui.label(RichText::new(text).small().weak());
            }
        }
    }

    if matches!(setup.qm.scene, Scene::Superposition { .. }) {
        let mut mix = setup.qm.mix();
        if ui
            .add(egui::Slider::new(&mut mix, 0.0..=1.0).text("mieszanka 2. stanu"))
            .changed()
        {
            setup.qm.set_mix(mix);
        }
    }

    ui.add(
        egui::Slider::new(&mut setup.qm.run.n_samples, 500..=80_000)
            .logarithmic(true)
            .text("próbek |ψ|²"),
    );
    ui.checkbox(&mut setup.qm.run.sparkle, "iskrzenie chmury");

    let charts = qm::plot::from_config(&setup.qm);
    charts::atom_charts(ui, &charts);
}

fn mixture_text(cfg: &sm::Config) -> String {
    let parts: Vec<String> = cfg
        .spawn
        .mixture
        .iter()
        .filter(|i| i.count > 0)
        .map(|i| format!("{}×{}", i.particle, i.count))
        .collect();
    if parts.is_empty() {
        "mieszanka pusta".to_string()
    } else {
        format!("{} · {}", parts.join("  "), cfg.describe_time_scale())
    }
}

fn setup_summary(setup: &Setup) -> String {
    match setup.mode {
        Mode::Relativistic => format!(
            "{} · N={} · R={:.1} · M={:.3e} · G={:.4} · c={:.1}",
            setup.sr.spawn.geometry.label(),
            setup.sr.spawn.n_particles,
            setup.sr.spawn.radius,
            setup.sr.spawn.total_mass,
            setup.sr.physics.g,
            setup.sr.physics.c
        ),
        Mode::Particles => mixture_text(&setup.sm),
        Mode::Atoms => setup.qm.scene.label(),
        Mode::Cosmological => {
            let c = lcdm::Cosmology::planck18();
            format!(
                "Planck18  Ω_m={:.3}  Ω_b={:.3}  Ω_Λ={:.3}\nh={:.4}  n_s={:.3}  σ₈={:.3}\n\
                 wiek dziś {:.2} Gyr · N={} · próbka {:.0} Mpc/h",
                c.omega_m,
                c.omega_b,
                c.omega_l,
                c.h,
                c.n_s,
                c.sigma8,
                c.age_gyr(1.0),
                setup.lcdm.n_particles(),
                setup.lcdm.box_size
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_setup_is_startable_in_both_modes() {
        let setup = Setup::default();
        assert!(setup.lcdm.n_grid >= 16);
        assert!(setup.sr.spawn.n_particles > 0);
        assert!(sr::presets::preset(setup.sr_preset).is_some());
        assert!(setup.sm.spawn.total_count() > 0);
        assert!(sm::presets::preset(setup.sm_preset).is_some());
        assert!(qm::presets::preset(setup.qm_preset).is_some());
        assert!(setup.qm.run.n_samples > 0);
    }

    #[test]
    fn every_cosmo_preset_has_a_label_and_a_config() {
        for preset in lcdm::Preset::ALL {
            assert!(!preset.label().is_empty());
            let cfg = preset.config();
            assert!(cfg.z_start > cfg.z_end);
        }
    }

    /// Podsumowanie jest jedyną informacją przed startem, więc nie może być puste
    /// ani zawierać `NaN` — a domyślna konfiguracja ΛCDM liczy wiek wszechświata.
    #[test]
    fn summary_is_filled_for_both_modes() {
        for mode in Mode::ALL {
            let setup = Setup {
                mode,
                ..Setup::default()
            };
            let text = setup_summary(&setup);
            assert!(!text.is_empty());
            assert!(!text.contains("NaN"), "{text}");
        }
    }

    #[test]
    fn every_mode_has_a_four_line_card() {
        for mode in Mode::ALL {
            let card = mode.card();
            assert!(!card.equation.is_empty(), "{mode:?}");
            assert!(!card.scope.is_empty(), "{mode:?}");
            assert!(!card.comparison.is_empty(), "{mode:?}");
            assert!(!card.not_this.is_empty(), "{mode:?}");
        }
    }
}
