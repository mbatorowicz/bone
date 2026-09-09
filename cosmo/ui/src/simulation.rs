//! Cztery modele i odtwarzanie pod jednym interfejsem panelu i renderera.

use bone_core::session::Session;
use bone_core::sr::relativity::Kinematics;
use bone_core::vec3::Vec3;
use crate::render::PointCloud;
use crate::replay::Replay;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Odosobniona chmura; kinematyka Newton albo SR (przełącznik).
    Relativistic,
    /// Próbka wszechświata ΛCDM z parametrami Plancka 2018.
    Cosmological,
    /// Klasyczny gaz cząstek Modelu Standardowego.
    Particles,
    /// Atomy i orbitale: Schrödinger (wodoropodobne) albo Slater (wieloelektronowe).
    Atoms,
}

impl Mode {
    pub const ALL: [Mode; 4] = [
        Mode::Relativistic,
        Mode::Cosmological,
        Mode::Particles,
        Mode::Atoms,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Relativistic => "N-ciała",
            Self::Cosmological => "Kosmologia",
            Self::Particles => "Cząstki",
            Self::Atoms => "Atomy",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::Relativistic => "Newton albo kinematyka SR",
            Self::Cosmological => "ΛCDM · PM periodyczny",
            Self::Particles => "kinematyka i rozpady PDG",
            Self::Atoms => "Schrödinger · |ψ|²",
        }
    }

    /// Cztery linie karty nad suwakami: równanie, zakres, porównanie, czym to nie jest.
    pub fn card(self) -> ModelCard {
        match self {
            Self::Relativistic => ModelCard {
                equation: "F = −G m m r / r³ (Plummer)",
                scope: "izolowana chmura · kinematyka Newton albo SR",
                comparison: "wiriał, dryf E · przy małym β energie zgodne",
                not_this: "nie metryka, nie 1PN, nie fale grawitacyjne",
            },
            Self::Cosmological => ModelCard {
                equation: "p = a² ẋ, tło Planck 2018",
                scope: "CDM, bez gazu · brzegi periodyczne",
                comparison: "δ_rms / D(a)",
                not_this: "bez oscylacji barionowych, pudło ≪ 150 Mpc/h",
            },
            Self::Particles => ModelCard {
                equation: "klasyczny gaz + losowe rozpady PDG",
                scope: "nie QFT",
                comparison: "B, L, Q całkowite",
                not_this: "nie amplitudy, nie hadronizacja",
            },
            Self::Atoms => ModelCard {
                equation: "ψ_nlm = R_nl Y_lm",
                scope: "chmura to próbka |ψ|²",
                comparison: "NIST / Hα",
                not_this: "nie HF, nie cząsteczki",
            },
        }
    }
}

/// Karta N-ciał zależy od przełącznika kinematyki — względność to nie kolor.
pub fn nbody_card(kinematics: Kinematics) -> ModelCard {
    match kinematics {
        Kinematics::Newton => ModelCard {
            equation: "v = p/m, E = p²/2m",
            scope: "izolowana chmura · siła Newtona (Plummer)",
            comparison: "wiriał, dryf E · |v| może przekroczyć c",
            not_this: "nie OTW, nie 1PN, nie fale grawitacyjne",
        },
        Kinematics::Sr => ModelCard {
            equation: "p = γmv, v = pc²/E",
            scope: "izolowana chmura · siła Newtona (Plummer)",
            comparison: "wiriał, dryf E · |v| < c z definicji",
            not_this: "nie OTW, nie 1PN, nie fale grawitacyjne",
        },
    }
}

/// Karta modelu: nazwa zakładki to laboratorium, nie nazwa teorii.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelCard {
    pub equation: &'static str,
    pub scope: &'static str,
    pub comparison: &'static str,
    pub not_this: &'static str,
}

/// To, co panel i renderer widzą: żywy bieg albo nagranie.
pub enum View {
    Live(Session),
    Replay(Replay),
}

impl View {
    pub fn rows(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Live(session) => session.rows(),
            Self::Replay(replay) => replay.rows(),
        }
    }

    pub fn headline(&self) -> String {
        match self {
            Self::Live(session) => session.headline(),
            Self::Replay(replay) => replay.headline(),
        }
    }

    pub fn lcdm_growth(&self) -> Option<bone_core::lcdm::growth::Chart> {
        match self {
            Self::Live(session) => session.lcdm_growth_chart(),
            Self::Replay(_) => None,
        }
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        match self {
            Self::Live(session) => session.take_warnings(),
            Self::Replay(_) => Vec::new(),
        }
    }

    pub fn as_live_mut(&mut self) -> Option<&mut Session> {
        match self {
            Self::Live(session) => Some(session),
            Self::Replay(_) => None,
        }
    }

    pub fn as_replay_mut(&mut self) -> Option<&mut Replay> {
        match self {
            Self::Replay(replay) => Some(replay),
            Self::Live(_) => None,
        }
    }

    pub fn is_replay(&self) -> bool {
        matches!(self, Self::Replay(_))
    }

    pub fn cloud(&self) -> &dyn PointCloud {
        match self {
            Self::Live(session) => session,
            Self::Replay(replay) => replay,
        }
    }
}

impl PointCloud for Session {
    fn len(&self) -> usize {
        Session::n(self)
    }

    fn position(&self, index: usize) -> Vec3 {
        Session::position(self, index)
    }

    fn shade(&self, index: usize) -> f32 {
        Session::shade(self, index)
    }

    fn center_span(&self) -> (Vec3, f64) {
        Session::center_span(self)
    }
}

impl PointCloud for Replay {
    fn len(&self) -> usize {
        Replay::n(self)
    }

    fn position(&self, index: usize) -> Vec3 {
        Replay::position(self, index)
    }

    fn shade(&self, index: usize) -> f32 {
        Replay::shade(self, index)
    }

    fn center_span(&self) -> (Vec3, f64) {
        Replay::center_span(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::lcdm;
    use bone_core::qm;
    use bone_core::sm;
    use bone_core::sr;
    use std::path::PathBuf;

    fn small_sr() -> sr::Config {
        let mut cfg = sr::presets::galaxy();
        cfg.spawn.n_particles = 200;
        cfg.solver.backend = sr::config::BackendKind::Exact;
        cfg
    }

    fn small_lcdm() -> lcdm::RunConfig {
        lcdm::RunConfig {
            n_grid: 12,
            pm_grid: 16,
            ..lcdm::RunConfig::structure()
        }
    }

    fn small_sm() -> sm::Config {
        sm::presets::para()
    }

    fn small_qm() -> qm::Config {
        let mut cfg = qm::presets::wodor();
        cfg.run.n_samples = 400;
        cfg.run.sparkle = false;
        cfg
    }

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "bone-view-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn both_modes_have_labels_and_subtitles() {
        for mode in Mode::ALL {
            assert!(!mode.label().is_empty());
            assert!(!mode.subtitle().is_empty());
        }
    }

    #[test]
    fn tab_names_are_laboratories_not_theories() {
        assert_eq!(Mode::Relativistic.label(), "N-ciała");
        assert_eq!(Mode::Cosmological.label(), "Kosmologia");
        assert_eq!(Mode::Particles.label(), "Cząstki");
        assert_eq!(Mode::Atoms.label(), "Atomy");
        assert_eq!(Mode::Relativistic.subtitle(), "Newton albo kinematyka SR");
        assert_eq!(Mode::Cosmological.subtitle(), "ΛCDM · PM periodyczny");
        assert_eq!(Mode::Particles.subtitle(), "kinematyka i rozpady PDG");
        assert_eq!(Mode::Atoms.subtitle(), "Schrödinger · |ψ|²");
    }

    #[test]
    fn model_cards_name_the_equation_and_the_lie() {
        let nbody = Mode::Relativistic.card();
        assert!(nbody.equation.contains("Plummer"));
        assert!(nbody.scope.contains("Newton"));
        assert!(nbody.scope.contains("SR"));
        assert!(nbody.not_this.contains("1PN"));
        let newton = nbody_card(Kinematics::Newton);
        assert!(newton.equation.contains("p/m"));
        assert!(newton.comparison.contains("przekroczyć c"));
        let rel = nbody_card(Kinematics::Sr);
        assert!(rel.equation.contains("γmv"));
        assert!(rel.comparison.contains("|v| < c"));
        let cosmo = Mode::Cosmological.card();
        assert!(cosmo.equation.contains("Planck 2018"));
        assert!(cosmo.scope.contains("periodyczne"));
        assert!(cosmo.comparison.contains("D(a)"));
        assert!(cosmo.not_this.contains("barionow"));
        assert!(!cosmo.not_this.contains("izolowane"));
        let particles = Mode::Particles.card();
        assert!(particles.scope.contains("nie QFT"));
        assert!(particles.not_this.contains("hadronizacja"));
        let atoms = Mode::Atoms.card();
        assert!(atoms.equation.contains("Y_lm"));
        assert!(atoms.not_this.contains("HF"));
    }

    #[test]
    fn live_view_advances_and_reports() {
        let mut view = View::Live(
            Session::start_sr(small_sr(), scratch("sr"), false).unwrap(),
        );
        view.as_live_mut().unwrap().advance(5).unwrap();
        assert!(!view.rows().is_empty());
        assert!(view.headline().contains("krok"));
        assert_eq!(view.cloud().len(), 200);
    }

    #[test]
    fn cosmological_view_reports_redshift() {
        let mut view = View::Live(
            Session::start_lcdm(small_lcdm(), scratch("lcdm"), false).unwrap(),
        );
        view.as_live_mut().unwrap().advance(3).unwrap();
        assert!(view.headline().contains("z="));
        assert_eq!(view.cloud().len(), 12usize.pow(3));
        let rows = view.rows();
        assert!(
            rows.iter().any(|(name, _)| *name == "σ(δ)/D"),
            "brak ilorazu wzrostu w tabeli: {rows:?}"
        );
        let chart = view.lcdm_growth().expect("wykres wzrostu");
        assert!(chart.ratio.xs.len() >= 2);
        assert!(chart.ratio_now.is_finite() && chart.ratio_now > 0.0);
    }

    #[test]
    fn particles_view_reports_census() {
        let mut view = View::Live(
            Session::start_sm(small_sm(), scratch("sm"), false).unwrap(),
        );
        view.as_live_mut().unwrap().advance(3).unwrap();
        assert!(view.headline().contains("krok"));
        assert_eq!(view.cloud().len(), 2);
    }

    #[test]
    fn atoms_view_reports_energy() {
        let mut view = View::Live(
            Session::start_qm(small_qm(), scratch("qm"), false).unwrap(),
        );
        view.as_live_mut().unwrap().advance(2).unwrap();
        assert!(view.headline().contains("krok"));
        assert_eq!(view.cloud().len(), 400);
    }

    #[test]
    fn live_galaxy_paints_visible_pixels() {
        use crate::camera::Camera;
        use crate::render::render;
        use eframe::egui::Color32;

        let view = View::Live(
            Session::start_sr(small_sr(), scratch("paint-sr"), false).unwrap(),
        );
        let image = render(Some(view.cloud()), &Camera::default(), 320, 240);
        let background = Color32::from_rgb(6, 8, 14);
        let lit = image.pixels.iter().filter(|p| **p != background).count();
        let max_luma = image
            .pixels
            .iter()
            .map(|p| p.r() as u32 + p.g() as u32 + p.b() as u32)
            .max()
            .unwrap_or(0);
        assert!(lit > 20, "żywa chmura nie trafiła na obraz: {lit} pikseli");
        assert!(max_luma >= 80, "żywa chmura jest za ciemna: luma {max_luma}");
    }

    #[test]
    fn shades_of_both_models_are_normalized() {
        let sr = Session::start_sr(small_sr(), scratch("shade-sr"), false).unwrap();
        for i in 0..sr.len() {
            let s = PointCloud::shade(&sr, i);
            assert!((0.0..1.0).contains(&s), "SR: cząstka {i} ma odcień {s}");
        }
        let lcdm = Session::start_lcdm(small_lcdm(), scratch("shade-lcdm"), false).unwrap();
        for i in 0..lcdm.len() {
            let s = PointCloud::shade(&lcdm, i);
            assert!((0.0..=1.0).contains(&s), "ΛCDM: cząstka {i} ma odcień {s}");
        }
        let sm = Session::start_sm(small_sm(), scratch("shade-sm"), false).unwrap();
        for i in 0..sm.len() {
            let s = PointCloud::shade(&sm, i);
            assert!((0.0..=1.0).contains(&s), "SM: cząstka {i} ma odcień {s}");
        }
        let qm = Session::start_qm(small_qm(), scratch("shade-qm"), false).unwrap();
        for i in 0..qm.len() {
            let s = PointCloud::shade(&qm, i);
            assert!((0.0..=1.0).contains(&s), "QM: punkt {i} ma odcień {s}");
        }
    }
}
