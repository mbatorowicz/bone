//! Wykresy funkcji, które panel rysuje bez silnika: `R_{nl}`, `P(r)`, poziomy `E_n`.
//!
//! Liczone z konfiguracji, nie ze stanu. Suwak `n` ma zmieniać krzywą od razu,
//! a nie dopiero po „Uruchom" — inaczej funkcje byłyby ozdobnikiem, a nie narzędziem.

use crate::qm::config::{Config, Scene};
use crate::qm::elements::{self, occupied_orbitals};
use crate::qm::hydrogen::{self, energy_hartree, mean_radius, radial, radial_probability, real_harmonic};
use crate::qm::units::{self, hartree_to_ev, wavelength_nm};

#[derive(Clone, Debug)]
pub struct Series {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct Level {
    pub n: u32,
    pub l: u32,
    pub e_ev: f64,
    pub occupied: u32,
    pub label: String,
}

#[derive(Clone, Debug)]
pub struct Transition {
    pub from_n: u32,
    pub to_n: u32,
    pub delta_ev: f64,
    pub wavelength_nm: f64,
    pub series: &'static str,
}

#[derive(Clone, Debug)]
pub struct Charts {
    /// `R_{nl}(r)` wzdłuż osi r.
    pub radial: Series,
    /// `P(r) = r² R²`.
    pub probability: Series,
    /// `|Y_{lm}(θ, 0)|²` w płaszczyźnie xz, `θ ∈ [0, π]`.
    pub angular: Series,
    pub levels: Vec<Level>,
    pub transitions: Vec<Transition>,
    pub formula: &'static str,
}

pub fn from_config(cfg: &Config) -> Charts {
    let cfg = cfg.sanitized();
    let (z, n, l, m) = cfg.scene.primary_orbital();
    let z_f = match &cfg.scene {
        Scene::Atom { z } => {
            let el = elements::nearest(*z);
            el.subshells
                .last()
                .map(|s| elements::z_eff(el.z, s.n, s.l, el.subshells))
                .unwrap_or(1.0)
        }
        _ => z as f64,
    };
    let r_max = (8.0 * mean_radius(z_f, n, l)).max(8.0);
    let radial = sample_radial(z_f, n, l, r_max, 240, false);
    let probability = sample_radial(z_f, n, l, r_max, 240, true);
    let angular = sample_angular(l, m, 180);
    let levels = match &cfg.scene {
        Scene::Atom { z } => atom_levels(elements::nearest(*z)),
        Scene::Orbital { z, .. } => hydrogen_levels(*z as f64, n, l, true),
        Scene::Superposition { z, terms } => hydrogen_levels(
            *z as f64,
            terms.iter().map(|t| t.n).max().unwrap_or(n),
            l,
            false,
        ),
    };
    let transitions = match &cfg.scene {
        Scene::Atom { .. } => Vec::new(),
        _ => rydberg_lines(z as f64, n.max(3)),
    };
    Charts {
        radial,
        probability,
        angular,
        levels,
        transitions,
        formula: formula_for(&cfg.scene, n, l),
    }
}

fn formula_for(scene: &Scene, n: u32, l: u32) -> &'static str {
    match scene {
        Scene::Atom { .. } => {
            "E ≈ −Z_eff² / (2n²)  hartree   (Slater; to nie jest pełny Hamiltonian)"
        }
        _ if l == 0 && n == 1 => "R₁₀(r) = 2 Z^{3/2} e^{−Zr}    E = −Z²/2  hartree",
        _ if l == 0 => "R_{n0}(r) ∝ L_{n−1}^{1}(2Zr/n) e^{−Zr/n}    E_n = −Z²/(2n²)",
        _ => "ψ_{nlm} = R_{nl}(r) Y_{lm}(θ,φ)    E_n = −μ Z² / (2n²)  hartree",
    }
}

fn sample_radial(z: f64, n: u32, l: u32, r_max: f64, points: usize, probability: bool) -> Series {
    let mut xs = Vec::with_capacity(points);
    let mut ys = Vec::with_capacity(points);
    for i in 0..points {
        let r = r_max * i as f64 / (points - 1).max(1) as f64;
        let y = if probability {
            radial_probability(z, n, l, r)
        } else {
            radial(z, n, l, r)
        };
        xs.push(r);
        ys.push(if y.is_finite() { y } else { 0.0 });
    }
    Series { xs, ys }
}

fn sample_angular(l: u32, m: i32, points: usize) -> Series {
    let mut xs = Vec::with_capacity(points);
    let mut ys = Vec::with_capacity(points);
    for i in 0..points {
        let theta = std::f64::consts::PI * i as f64 / (points - 1).max(1) as f64;
        let y = real_harmonic(l, m, theta, 0.0);
        xs.push(theta);
        ys.push(y * y);
    }
    Series { xs, ys }
}

fn hydrogen_levels(z: f64, n_hi: u32, highlight_l: u32, mark: bool) -> Vec<Level> {
    let n_hi = n_hi.clamp(1, 8);
    let mut out = Vec::new();
    for n in 1..=n_hi {
        for l in 0..n {
            let e = hartree_to_ev(energy_hartree(z, n, 1.0));
            out.push(Level {
                n,
                l,
                e_ev: e,
                occupied: u32::from(mark && n == n_hi && l == highlight_l),
                label: hydrogen::orbital_label(n, l, 0),
            });
        }
    }
    out
}

fn atom_levels(element: &elements::Element) -> Vec<Level> {
    occupied_orbitals(element)
        .into_iter()
        .map(|o| Level {
            n: o.n,
            l: o.l,
            e_ev: hartree_to_ev(energy_hartree(o.z_eff, o.n, element.reduced_mass())),
            occupied: 1,
            label: o.label(),
        })
        .collect()
}

fn rydberg_lines(z: f64, n_hi: u32) -> Vec<Transition> {
    let mut out = Vec::new();
    let mu = units::reduced_mass_ratio(units::PROTON_ELECTRON_MASS);
    let n_hi = n_hi.clamp(3, 8);
    for (to_n, series) in [(1, "Lyman"), (2, "Balmer"), (3, "Paschen")] {
        for from_n in (to_n + 1)..=n_hi {
            let e_from = energy_hartree(z, from_n, mu);
            let e_to = energy_hartree(z, to_n, mu);
            let delta = hartree_to_ev(e_from - e_to);
            if let Some(lambda) = wavelength_nm(delta) {
                out.push(Transition {
                    from_n,
                    to_n,
                    delta_ev: delta,
                    wavelength_nm: lambda,
                    series,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm::config::{Scene, Term};

    #[test]
    fn hydrogen_1s_radial_peaks_at_the_bohr_radius() {
        let charts = from_config(&Config::default());
        let (r_peak, _) = charts
            .probability
            .xs
            .iter()
            .zip(charts.probability.ys.iter())
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(
            (*r_peak - 1.0).abs() < 0.08,
            "maksimum P(r) przy r={r_peak}, oczekiwane a₀"
        );
        assert!(charts.radial.ys[0] > 1.0);
    }

    #[test]
    fn balmer_alpha_is_in_the_list() {
        let charts = from_config(&Config::default());
        let h_alpha = charts
            .transitions
            .iter()
            .find(|t| t.series == "Balmer" && t.from_n == 3)
            .unwrap();
        assert!((h_alpha.wavelength_nm - 656.3).abs() < 1.0, "{}", h_alpha.wavelength_nm);
    }

    #[test]
    fn superposition_charts_are_nonempty() {
        let cfg = Config {
            scene: Scene::Superposition {
                z: 1,
                terms: vec![Term::real(1, 0, 0, 1.0), Term::real(2, 1, 0, 1.0)],
            },
            ..Config::default()
        };
        let charts = from_config(&cfg);
        assert!(!charts.levels.is_empty());
        assert!(!charts.radial.xs.is_empty());
    }
}
