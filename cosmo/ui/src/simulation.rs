//! Dwa modele i odtwarzanie pod jednym interfejsem panelu i renderera.

use bone_core::session::Session;
use crate::render::PointCloud;
use crate::replay::Replay;
use bone_core::vec3::Vec3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Odosobniona chmura, kinematyka szczególnej teorii względności.
    Relativistic,
    /// Próbka wszechświata ΛCDM z parametrami Plancka 2018.
    Cosmological,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::Relativistic, Mode::Cosmological];

    pub fn label(self) -> &'static str {
        match self {
            Self::Relativistic => "chmura SR",
            Self::Cosmological => "ΛCDM",
        }
    }

    pub fn subtitle(self) -> &'static str {
        match self {
            Self::Relativistic => "grawitacja newtonowska, kinematyka SR — układ izolowany",
            Self::Cosmological => "ΛCDM · Planck 2018 · PM izolowany (Hockney)",
        }
    }
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
    }
}
