//! Pierwiastki, konfiguracje elektronowe i reguły Slatera.
//!
//! Hel ma osobną wariację w [`crate::qm::helium`]. Slater zostaje jako lekcja
//! ekranowania: każdy elektron w wodoropodobnym orbitalu z `Z_eff = Z − σ`
//! (Slater 1930). To jest **przybliżenie niezależnych cząstek**. Diagnostyka
//! porównuje energię orbitalu walencyjnego z pierwszą jonizacją NIST i
//! **pokazuje błąd** — albo milczy, gdy atom jest za ciężki na tę reklamę.

use crate::qm::hydrogen::{orbital_label, quantum_ok};
use crate::qm::units::{self, PROTON_ELECTRON_MASS};

/// Jeden orbital obsadzony `occupancy` elektronami.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subshell {
    pub n: u32,
    pub l: u32,
    pub occupancy: u32,
}

/// Elektron w konkretnym orbitalu rzeczywistym, z własnym `Z_eff`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Occupied {
    pub n: u32,
    pub l: u32,
    pub m: i32,
    pub z_eff: f64,
}

impl Occupied {
    pub fn label(self) -> String {
        orbital_label(self.n, self.l, self.m)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Element {
    pub z: u32,
    pub symbol: &'static str,
    pub name: &'static str,
    /// Pierwsza energia jonizacji (eV), NIST ASD.
    pub ionization_ev: f64,
    /// Masa jądra w jednostkach `m_e` — do zredukowanej masy elektronu.
    pub nucleus_over_electron: f64,
    pub subshells: &'static [Subshell],
}

const fn s(n: u32, l: u32, occupancy: u32) -> Subshell {
    Subshell { n, l, occupancy }
}

/// Masa jądra ≈ `A · m_p / m_e`. Wystarcza do korekty zredukowanej masy;
/// defekt masy jądra jest tu zaniedbywalny wobec błędu Slatera.
const fn nucleus(a: f64) -> f64 {
    a * PROTON_ELECTRON_MASS
}

const H: &[Subshell] = &[s(1, 0, 1)];
const HE: &[Subshell] = &[s(1, 0, 2)];
const LI: &[Subshell] = &[s(1, 0, 2), s(2, 0, 1)];
const BE: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2)];
const B: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 1)];
const C: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 2)];
const N: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 3)];
const O: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 4)];
const F: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 5)];
const NE: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6)];
const NA: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 1)];
const MG: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2)];
const AL: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 1)];
const SI: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 2)];
const P: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 3)];
const S: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 4)];
const CL: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 5)];
const AR: &[Subshell] = &[s(1, 0, 2), s(2, 0, 2), s(2, 1, 6), s(3, 0, 2), s(3, 1, 6)];
const FE: &[Subshell] = &[
    s(1, 0, 2),
    s(2, 0, 2),
    s(2, 1, 6),
    s(3, 0, 2),
    s(3, 1, 6),
    s(3, 2, 6),
    s(4, 0, 2),
];
const CU: &[Subshell] = &[
    s(1, 0, 2),
    s(2, 0, 2),
    s(2, 1, 6),
    s(3, 0, 2),
    s(3, 1, 6),
    s(3, 2, 10),
    s(4, 0, 1),
];
const AU: &[Subshell] = &[
    s(1, 0, 2),
    s(2, 0, 2),
    s(2, 1, 6),
    s(3, 0, 2),
    s(3, 1, 6),
    s(3, 2, 10),
    s(4, 0, 2),
    s(4, 1, 6),
    s(4, 2, 10),
    s(4, 3, 14),
    s(5, 0, 2),
    s(5, 1, 6),
    s(5, 2, 10),
    s(6, 0, 1),
];
const U: &[Subshell] = &[
    s(1, 0, 2),
    s(2, 0, 2),
    s(2, 1, 6),
    s(3, 0, 2),
    s(3, 1, 6),
    s(3, 2, 10),
    s(4, 0, 2),
    s(4, 1, 6),
    s(4, 2, 10),
    s(4, 3, 14),
    s(5, 0, 2),
    s(5, 1, 6),
    s(5, 2, 10),
    s(5, 3, 3),
    s(6, 0, 2),
    s(6, 1, 6),
    s(6, 2, 1),
    s(7, 0, 2),
];

pub const TABLE: &[Element] = &[
    Element { z: 1, symbol: "H", name: "wodór", ionization_ev: 13.59844, nucleus_over_electron: nucleus(1.0), subshells: H },
    Element { z: 2, symbol: "He", name: "hel", ionization_ev: 24.58739, nucleus_over_electron: nucleus(4.0), subshells: HE },
    Element { z: 3, symbol: "Li", name: "lit", ionization_ev: 5.39172, nucleus_over_electron: nucleus(7.0), subshells: LI },
    Element { z: 4, symbol: "Be", name: "beryl", ionization_ev: 9.32270, nucleus_over_electron: nucleus(9.0), subshells: BE },
    Element { z: 5, symbol: "B", name: "bor", ionization_ev: 8.29803, nucleus_over_electron: nucleus(11.0), subshells: B },
    Element { z: 6, symbol: "C", name: "węgiel", ionization_ev: 11.26030, nucleus_over_electron: nucleus(12.0), subshells: C },
    Element { z: 7, symbol: "N", name: "azot", ionization_ev: 14.53414, nucleus_over_electron: nucleus(14.0), subshells: N },
    Element { z: 8, symbol: "O", name: "tlen", ionization_ev: 13.61806, nucleus_over_electron: nucleus(16.0), subshells: O },
    Element { z: 9, symbol: "F", name: "fluor", ionization_ev: 17.42282, nucleus_over_electron: nucleus(19.0), subshells: F },
    Element { z: 10, symbol: "Ne", name: "neon", ionization_ev: 21.56454, nucleus_over_electron: nucleus(20.0), subshells: NE },
    Element { z: 11, symbol: "Na", name: "sód", ionization_ev: 5.13908, nucleus_over_electron: nucleus(23.0), subshells: NA },
    Element { z: 12, symbol: "Mg", name: "magnez", ionization_ev: 7.64624, nucleus_over_electron: nucleus(24.0), subshells: MG },
    Element { z: 13, symbol: "Al", name: "glin", ionization_ev: 5.98577, nucleus_over_electron: nucleus(27.0), subshells: AL },
    Element { z: 14, symbol: "Si", name: "krzem", ionization_ev: 8.15169, nucleus_over_electron: nucleus(28.0), subshells: SI },
    Element { z: 15, symbol: "P", name: "fosfor", ionization_ev: 10.48669, nucleus_over_electron: nucleus(31.0), subshells: P },
    Element { z: 16, symbol: "S", name: "siarka", ionization_ev: 10.36001, nucleus_over_electron: nucleus(32.0), subshells: S },
    Element { z: 17, symbol: "Cl", name: "chlor", ionization_ev: 12.96764, nucleus_over_electron: nucleus(35.0), subshells: CL },
    Element { z: 18, symbol: "Ar", name: "argon", ionization_ev: 15.75962, nucleus_over_electron: nucleus(40.0), subshells: AR },
    Element { z: 26, symbol: "Fe", name: "żelazo", ionization_ev: 7.9024, nucleus_over_electron: nucleus(56.0), subshells: FE },
    Element { z: 29, symbol: "Cu", name: "miedź", ionization_ev: 7.7264, nucleus_over_electron: nucleus(63.0), subshells: CU },
    Element { z: 79, symbol: "Au", name: "złoto", ionization_ev: 9.2255, nucleus_over_electron: nucleus(197.0), subshells: AU },
    Element { z: 92, symbol: "U", name: "uran", ionization_ev: 6.1941, nucleus_over_electron: nucleus(238.0), subshells: U },
];

pub fn by_z(z: u32) -> Option<&'static Element> {
    TABLE.iter().find(|e| e.z == z)
}

pub fn nearest(z: u32) -> &'static Element {
    TABLE
        .iter()
        .min_by_key(|e| e.z.abs_diff(z.max(1)))
        .unwrap_or(&TABLE[0])
}

impl Element {
    pub fn electron_count(&self) -> u32 {
        self.subshells.iter().map(|s| s.occupancy).sum()
    }

    pub fn configuration_text(&self) -> String {
        self.subshells
            .iter()
            .map(|s| {
                format!(
                    "{}{}{}",
                    s.n,
                    crate::qm::hydrogen::spectroscopic(s.l),
                    superscript(s.occupancy)
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn reduced_mass(&self) -> f64 {
        units::reduced_mass_ratio(self.nucleus_over_electron)
    }
}

fn superscript(n: u32) -> String {
    const DIGITS: &[char] = &['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    if n < 10 {
        return DIGITS[n as usize].to_string();
    }
    n.to_string()
        .chars()
        .map(|c| DIGITS[c.to_digit(10).unwrap_or(0) as usize])
        .collect()
}

/// Grupa ekranowania Slatera: (1s), (2s 2p), (3s 3p), (3d), (4s 4p), …
fn slater_group(n: u32, l: u32) -> u32 {
    match (n, l) {
        (1, _) => 0,
        (2, _) => 1,
        (3, 0 | 1) => 2,
        (3, _) => 3,
        (4, 0 | 1) => 4,
        (4, 2) => 5,
        (4, _) => 6,
        (5, 0 | 1) => 7,
        (5, 2) => 8,
        (5, _) => 9,
        (6, 0 | 1) => 10,
        (6, _) => 11,
        (7, _) => 12,
        _ => 13,
    }
}

fn principal(n: u32, l: u32) -> u32 {
    let _ = l;
    n
}

/// Stała ekranowania `σ` dla elektronu w powłoce `(n, l)`.
pub fn slater_sigma(_z: u32, n: u32, l: u32, subshells: &[Subshell]) -> f64 {
    let group = slater_group(n, l);
    let mut sigma = 0.0;
    for shell in subshells {
        let g = slater_group(shell.n, shell.l);
        let count = shell.occupancy as f64;
        if g > group {
            continue;
        }
        if g == group {
            let others = if shell.n == n && shell.l == l {
                (count - 1.0).max(0.0)
            } else {
                count
            };
            sigma += others * if n == 1 { 0.30 } else { 0.35 };
            continue;
        }
        if principal(shell.n, shell.l) + 1 == n && l <= 1 {
            sigma += count * 0.85;
        } else {
            sigma += count;
        }
    }
    sigma
}

pub fn z_eff(z: u32, n: u32, l: u32, subshells: &[Subshell]) -> f64 {
    (z as f64 - slater_sigma(z, n, l, subshells)).max(0.15)
}

/// Wypełnianie magnetycznych `m` regułą Hunda: najpierw po jednym, potem pary.
fn hund_m_values(l: u32, occupancy: u32) -> Vec<i32> {
    let l = l as i32;
    let mut order: Vec<i32> = (0..=l).chain((-l..0).rev()).collect();
    if l == 1 {
        order = vec![1, -1, 0];
    }
    if l == 2 {
        order = vec![2, 1, 0, -1, -2];
    }
    let slots = (2 * l + 1) as u32;
    let mut out = Vec::with_capacity(occupancy as usize);
    let first = occupancy.min(slots);
    for i in 0..first {
        out.push(order[i as usize]);
    }
    let rest = occupancy.saturating_sub(slots);
    for i in 0..rest {
        out.push(order[i as usize]);
    }
    out
}

/// Rozwiń konfigurację na listę elektronów z `Z_eff` i `m`.
pub fn occupied_orbitals(element: &Element) -> Vec<Occupied> {
    let mut out = Vec::new();
    for shell in element.subshells {
        if !quantum_ok(shell.n, shell.l, 0) {
            continue;
        }
        let z_eff = z_eff(element.z, shell.n, shell.l, element.subshells);
        for m in hund_m_values(shell.l, shell.occupancy) {
            out.push(Occupied {
                n: shell.n,
                l: shell.l,
                m,
                z_eff,
            });
        }
    }
    out
}

/// Energia orbitalu walencyjnego (hartree) — przybliżenie Koopmansa dla IE.
pub fn valence_energy_hartree(element: &Element) -> Option<f64> {
    let last = element.subshells.last()?;
    let z_eff = z_eff(element.z, last.n, last.l, element.subshells);
    Some(crate::qm::hydrogen::energy_hartree(
        z_eff,
        last.n,
        element.reduced_mass(),
    ))
}

pub fn valence_ionization_ev(element: &Element) -> Option<f64> {
    valence_energy_hartree(element).map(|e| -units::hartree_to_ev(e))
}

/// Slater pokazuje IE tylko dla lekkich atomów. Fe/Cu/Au/U to Aufbau:
/// konfiguracja, nie energia.
pub fn reports_ionization(element: &Element) -> bool {
    element.z <= 18
}

/// Względny błąd IE: `(model − doświadczenie) / doświadczenie`.
/// `None` gdy model nie reklamuje jonizacji (ciężkie atomy).
pub fn ionization_error(element: &Element) -> Option<f64> {
    if !reports_ionization(element) {
        return None;
    }
    let model = valence_ionization_ev(element)?;
    if element.ionization_ev == 0.0 {
        return None;
    }
    Some((model - element.ionization_ev) / element.ionization_ev)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_unique_z_and_matches_electron_count() {
        let mut seen = std::collections::BTreeSet::new();
        for el in TABLE {
            assert!(seen.insert(el.z), "powtórzony Z={}", el.z);
            assert_eq!(el.electron_count(), el.z, "{}", el.symbol);
            assert!(el.ionization_ev > 0.0);
            assert!(!el.name.is_empty());
        }
        assert_eq!(by_z(6).unwrap().symbol, "C");
        assert!(by_z(99).is_none());
        assert_eq!(nearest(25).symbol, "Fe");
    }

    #[test]
    fn carbon_is_1s2_2s2_2p2() {
        let c = by_z(6).unwrap();
        assert_eq!(c.configuration_text(), "1s² 2s² 2p²");
        let occ = occupied_orbitals(c);
        assert_eq!(occ.len(), 6);
        assert_eq!(occ.iter().filter(|o| o.n == 1).count(), 2);
        assert_eq!(occ.iter().filter(|o| o.l == 1).count(), 2);
        let p_m: Vec<i32> = occ.iter().filter(|o| o.l == 1).map(|o| o.m).collect();
        assert_eq!(p_m, vec![1, -1], "Hund: dwa 2p niesparowane w p_x i p_y");
    }

    #[test]
    fn slater_he_1s_is_1_70() {
        let he = by_z(2).unwrap();
        let z = z_eff(he.z, 1, 0, he.subshells);
        assert!((z - 1.70).abs() < 1e-12, "{z}");
    }

    #[test]
    fn slater_lithium_2s_is_1_30() {
        let li = by_z(3).unwrap();
        // σ = 2·0,85 = 1,70; Z_eff = 3 − 1,70 = 1,30.
        let z = z_eff(li.z, 2, 0, li.subshells);
        assert!((z - 1.30).abs() < 1e-12, "{z}");
    }

    #[test]
    fn hydrogen_ionization_error_is_tiny() {
        let h = by_z(1).unwrap();
        let err = ionization_error(h).unwrap();
        assert!(err.abs() < 1e-3, "{err}");
    }

    #[test]
    fn slater_ionization_is_wrong_and_reported() {
        let he = by_z(2).unwrap();
        let model = valence_ionization_ev(he).unwrap();
        // Koopmans+Slater na He daje ~39 eV wobec 24,6 eV. To ma być WIDOCZNE.
        assert!(model > 30.0, "{model}");
        let err = ionization_error(he).unwrap();
        assert!(err.abs() > 0.2, "błąd He powinien być duży, jest {err}");
    }

    #[test]
    fn iron_does_not_advertise_ionization() {
        let fe = by_z(26).unwrap();
        assert!(!reports_ionization(fe));
        assert!(ionization_error(fe).is_none());
        assert!(valence_ionization_ev(fe).is_some());
    }

    #[test]
    fn copper_uses_the_4s1_3d10_exception() {
        let cu = by_z(29).unwrap();
        let last = cu.subshells.last().unwrap();
        assert_eq!((last.n, last.l, last.occupancy), (4, 0, 1));
        assert_eq!(cu.electron_count(), 29);
    }
}
