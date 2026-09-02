//! Nazwane zestawy nastaw ΛCDM — jedno źródło dla CLI i panelu.
//!
//! Identyfikatory (`liniowy`, `struktury`, `struktury64`) są częścią kontraktu
//! wiersza poleceń. Etykiety są po polsku, bo panel i pomoc też są po polsku.
//! Same liczby żyją w [`RunConfig`]; ten moduł tylko je nazywa.

use crate::lcdm::engine::RunConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    LinearGrowth,
    Structure,
    Structure64,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Self::LinearGrowth, Self::Structure, Self::Structure64];

    pub fn id(self) -> &'static str {
        match self {
            Self::LinearGrowth => "liniowy",
            Self::Structure => "struktury",
            Self::Structure64 => "struktury64",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::LinearGrowth => "wzrost liniowy",
            Self::Structure => "formacja struktur",
            Self::Structure64 => "formacja 64³",
        }
    }

    pub fn config(self) -> RunConfig {
        match self {
            Self::LinearGrowth => RunConfig::linear_growth(),
            Self::Structure => RunConfig::structure(),
            Self::Structure64 => RunConfig::structure_64(),
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|preset| preset.id() == text)
    }
}

pub fn ids() -> Vec<&'static str> {
    Preset::ALL.iter().map(|preset| preset.id()).collect()
}

pub fn by_id(id: &str) -> Option<RunConfig> {
    Preset::parse(id).map(Preset::config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_has_id_label_and_sane_config() {
        for preset in Preset::ALL {
            assert!(!preset.id().is_empty());
            assert!(!preset.label().is_empty());
            let cfg = preset.config();
            assert!(cfg.z_start > cfg.z_end);
            assert!(cfg.n_grid >= 8);
            assert_eq!(by_id(preset.id()), Some(cfg));
        }
    }

    #[test]
    fn unknown_id_is_none() {
        assert!(Preset::parse("nie-ma").is_none());
        assert!(by_id("nie-ma").is_none());
    }

    #[test]
    fn ids_match_cli_contract() {
        let names = ids();
        assert_eq!(names, ["liniowy", "struktury", "struktury64"]);
    }
}
