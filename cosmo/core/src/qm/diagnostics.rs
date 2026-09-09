//! Pomiary, które panel i CLI pokazują obok chmury.
//!
//! Dla orbitalu wodoropodobnego liczby są dokładne (wzór, nie całka po próbce).
//! Hel wariacyjny podaje energię całkowitą wobec −2.904 Ha. Slaterowski atom
//! lekki stawia IE obok NIST; ciężki pokazuje konfigurację, bez energetyki.

use crate::qm::config::{Config, Scene};
use crate::qm::elements::{self, ionization_error, valence_ionization_ev};
use crate::qm::helium;
use crate::qm::hydrogen::{self, energy_hartree, mean_radius};
use crate::qm::plot::{self, Charts};
use crate::qm::units::{atomic_to_fs, bohr_to_fm, hartree_to_ev};
use crate::vec3::Vec3;

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub time: f64,
    pub step: u64,
    pub n: usize,
    pub energy_ev: f64,
    pub mean_r: f64,
    pub sample_mean_r: f64,
    pub bohr_r: f64,
    pub ionization_ev: Option<f64>,
    pub ionization_exp_ev: Option<f64>,
    pub ionization_error: Option<f64>,
    pub variational_hartree: Option<f64>,
    pub exact_hartree: Option<f64>,
    pub energy_error: Option<f64>,
    pub beat_period: Option<f64>,
    pub exact: bool,
    pub method: &'static str,
    pub charts: Charts,
}

pub fn collect(
    cfg: &Config,
    time: f64,
    step: u64,
    positions: &[Vec3],
    mu: f64,
) -> Snapshot {
    let cfg = cfg.sanitized();
    let (z, n, l, _m) = cfg.scene.primary_orbital();
    let z_f = match &cfg.scene {
        Scene::Atom { z } => {
            let el = elements::nearest(*z);
            el.subshells
                .last()
                .map(|s| elements::z_eff(el.z, s.n, s.l, el.subshells))
                .unwrap_or(*z as f64)
        }
        Scene::Helium => helium::ZETA,
        _ => z as f64,
    };
    let energy = match &cfg.scene {
        Scene::Superposition { z, terms } => terms
            .iter()
            .map(|t| t.amplitude_sq() * energy_hartree(*z as f64, t.n, mu))
            .sum::<f64>()
            / terms.iter().map(|t| t.amplitude_sq()).sum::<f64>().max(1e-18),
        Scene::Helium => helium::ENERGY_HARTREE,
        _ => energy_hartree(z_f, n, mu),
    };
    let sample_mean_r = if positions.is_empty() {
        0.0
    } else {
        positions.iter().map(|p| p.norm()).sum::<f64>() / positions.len() as f64
    };
    let (ie_model, ie_exp, ie_err) = match &cfg.scene {
        Scene::Atom { z } => {
            let el = elements::nearest(*z);
            if elements::reports_ionization(el) {
                (
                    valence_ionization_ev(el),
                    Some(el.ionization_ev),
                    ionization_error(el),
                )
            } else {
                (None, None, None)
            }
        }
        Scene::Orbital { z, n, .. } if *n == 1 => {
            let e = -hartree_to_ev(energy_hartree(*z as f64, 1, mu));
            let exp = elements::by_z(*z).map(|el| el.ionization_ev);
            let err = exp.map(|v| (e - v) / v);
            (Some(e), exp, err)
        }
        _ => (None, None, None),
    };
    let (var_ha, exact_ha, e_err) = match &cfg.scene {
        Scene::Helium => (
            Some(helium::ENERGY_HARTREE),
            Some(helium::EXACT_ENERGY_HARTREE),
            Some(helium::energy_error()),
        ),
        _ => (None, None, None),
    };
    let beat = match &cfg.scene {
        Scene::Superposition { z, terms } if terms.len() >= 2 => hydrogen::beat_period(
            energy_hartree(*z as f64, terms[0].n, mu),
            energy_hartree(*z as f64, terms[1].n, mu),
        ),
        _ => None,
    };
    let method = match &cfg.scene {
        Scene::Orbital { .. } | Scene::Superposition { .. } => "dokładny Schrödinger",
        Scene::Helium => "wariacja ζ=27/16",
        Scene::Atom { z } => {
            if elements::reports_ionization(elements::nearest(*z)) {
                "Slater Z_eff"
            } else {
                "Aufbau (konfiguracja)"
            }
        }
    };
    let exact = matches!(
        cfg.scene,
        Scene::Orbital { .. } | Scene::Superposition { .. }
    );
    Snapshot {
        time,
        step,
        n: positions.len(),
        energy_ev: hartree_to_ev(energy),
        mean_r: mean_radius(z_f, n, l),
        sample_mean_r,
        bohr_r: hydrogen::bohr_radius(z_f, n),
        ionization_ev: ie_model,
        ionization_exp_ev: ie_exp,
        ionization_error: ie_err,
        variational_hartree: var_ha,
        exact_hartree: exact_ha,
        energy_error: e_err,
        beat_period: beat,
        exact,
        method,
        charts: plot::from_config(&cfg),
    }
}

impl Snapshot {
    pub fn time_fs(&self) -> f64 {
        atomic_to_fs(self.time)
    }

    pub fn mean_r_fm(&self) -> f64 {
        bohr_to_fm(self.mean_r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm::config::Config;
    use crate::vec3::ZERO;

    #[test]
    fn hydrogen_snapshot_energy_is_minus_13_6() {
        let snap = collect(&Config::default(), 0.0, 0, &[ZERO; 8], 1.0);
        assert!((snap.energy_ev + 13.605_7).abs() < 0.01, "{}", snap.energy_ev);
        assert!(snap.exact);
        assert_eq!(snap.method, "dokładny Schrödinger");
        assert!((snap.mean_r - 1.5).abs() < 1e-12);
        assert!((snap.bohr_r - 1.0).abs() < 1e-12);
        assert!(!snap.charts.radial.xs.is_empty());
    }

    #[test]
    fn helium_snapshot_reports_two_percent_error() {
        let cfg = Config {
            scene: crate::qm::config::Scene::Helium,
            ..Config::default()
        };
        let snap = collect(&cfg, 0.0, 0, &[ZERO; 8], 1.0);
        assert!(!snap.exact);
        assert_eq!(snap.method, "wariacja ζ=27/16");
        let err = snap.energy_error.unwrap();
        assert!((0.015..0.025).contains(&err), "{err}");
        let e = snap.variational_hartree.unwrap();
        assert!((e - crate::qm::helium::ENERGY_HARTREE).abs() < 1e-12);
        assert!(snap.ionization_ev.is_none());
    }
}
