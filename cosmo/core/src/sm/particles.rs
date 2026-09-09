//! Tablica cząstek Modelu Standardowego — jedno źródło mas i liczb kwantowych.
//!
//! Siedemnaście cząstek elementarnych (sześć kwarków, sześć leptonów, cztery bozony
//! cechowania, bozon Higgsa) plus cztery hadrony złożone, bez których nie dałoby się
//! policzyć niczego obserwowalnego: proton, neutron i piony.
//!
//! # Ładunek jest liczbą całkowitą
//!
//! Ładunek jest przechowywany w **trzecich `e`** jako `i32`, a nie jako ułamek
//! zmiennoprzecinkowy. Kwark górny ma `+2`, dolny `−1`, elektron `−3`. Powód jest
//! jeden i wystarczający: rozpad tworzy i niszczy cząstki, a suma ładunków przed
//! rozpadem i po nim musi się zgadzać **dokładnie**. Z `f64` wyszłoby „zgadza się do
//! 10⁻¹⁵" i nie dałoby się odróżnić poprawnego kanału od kanału z pomyloną cząstką,
//! którego błąd też jest mały. Ta sama zasada obejmuje liczbę barionową (też
//! w trzecich, bo kwark ma `1/3`) i liczby leptonowe.
//!
//! # Antycząstki nie mają własnych wpisów
//!
//! [`Particle`] to gatunek plus znacznik `anti`. Antycząstka ma tę samą masę, ten sam
//! spin i przeciwne wszystkie ładunki addytywne. Zdublowanie tablicy dawałoby
//! czterdzieści wpisów, z których połowa musiałaby ręcznie powtarzać masy — a przy
//! pierwszej poprawce z PDG rozjechałyby się połówki tej samej pary.
//!
//! # Czego w tej tablicy NIE ma
//!
//! Nie ma szerokości rozpadu jako liczby zespolonej, mieszania (macierze CKM i PMNS),
//! sprzężeń Yukawy ani stanów związanych innych niż wypisane hadrony. To jest tablica
//! liczb kwantowych do symulacji klasycznej, a nie wejście do generatora zdarzeń.

use crate::sm::units;

/// Do której rodziny należy cząstka.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Family {
    Quark,
    Lepton,
    /// Bozon cechowania: nośnik oddziaływania, spin 1.
    Gauge,
    /// Bozon Higgsa: jedyny skalar, spin 0.
    Scalar,
    /// Cząstka złożona z kwarków — nie jest elementarna.
    Hadron,
}

/// Ładunek kolorowy. Rozstrzyga, czy cząstka bierze udział w oddziaływaniu silnym.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Color {
    /// Bez koloru — leptony, fotony, hadrony jako całość (są białe).
    Neutral,
    /// Trójka koloru: kwarki.
    Triplet,
    /// Ósemka: gluony. Niosą kolor, więc oddziałują same ze sobą.
    Octet,
}

/// Zapach leptonowy — osobno zachowywana liczba dla każdego pokolenia.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Flavour {
    Electron,
    Muon,
    Tau,
}

/// Gatunek cząstki. Kolejność wariantów jest kolejnością wierszy w [`TABLE`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Species {
    Up,
    Down,
    Strange,
    Charm,
    Bottom,
    Top,
    Electron,
    Muon,
    Tau,
    NeutrinoE,
    NeutrinoMu,
    NeutrinoTau,
    Photon,
    Gluon,
    WBoson,
    ZBoson,
    Higgs,
    Proton,
    Neutron,
    PionCharged,
    PionNeutral,
}

/// Jeden wiersz tablicy. Wartości z PDG (wydanie 2024) i CODATA 2018.
#[derive(Clone, Copy, Debug)]
pub struct Data {
    /// Identyfikator w wierszu poleceń i w plikach konfiguracji.
    pub id: &'static str,
    /// Symbol dla cząstki (nie antycząstki).
    pub symbol: &'static str,
    /// Nazwa polska.
    pub name: &'static str,
    /// Masa spoczynkowa w MeV.
    pub mass: f64,
    /// Ładunek elektryczny w trzecich `e`.
    pub charge_thirds: i32,
    /// Podwojony spin: `1` znaczy `½`, `2` znaczy `1`, `0` znaczy `0`.
    pub spin_doubled: u32,
    pub color: Color,
    pub family: Family,
    /// Pokolenie 1–3; `0` dla bozonów i hadronów.
    pub generation: u8,
    /// Średni czas życia w sekundach. `None` znaczy „trwała w granicach pomiaru".
    pub lifetime_s: Option<f64>,
    /// Liczba barionowa w trzecich (kwark ma `1`, proton `3`).
    pub baryon_thirds: i32,
    /// Liczba leptonowa.
    pub lepton: i32,
    pub flavour: Option<Flavour>,
    /// Czy cząstka jest swoją własną antycząstką.
    pub self_conjugate: bool,
}

/// Masa neutrin przyjęta jako zero.
///
/// Neutrina mają masę — to najpewniejszy znany wyłom w pierwotnym Modelu
/// Standardowym. Ale znane są tylko różnice kwadratów mas i górne ograniczenie
/// (`Σm < 0,12 eV` z kosmologii, `m_νe < 0,8 eV` z KATRIN-a), więc każda liczba
/// wpisana w to pole byłaby wymyślona. Zero jest jawnym przybliżeniem: neutrino
//  porusza się wtedy dokładnie z `c` i nie rozpada się. Skutkiem jest to, że
/// oscylacje neutrin są poza tym modelem, i to jest powiedziane wprost.
pub const NEUTRINO_MASS: f64 = 0.0;

const TABLE: [Data; 21] = [
    Data {
        id: "u",
        symbol: "u",
        name: "kwark górny",
        mass: 2.16,
        charge_thirds: 2,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 1,
        lifetime_s: None,
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "d",
        symbol: "d",
        name: "kwark dolny",
        mass: 4.67,
        charge_thirds: -1,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 1,
        lifetime_s: None,
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "s",
        symbol: "s",
        name: "kwark dziwny",
        mass: 93.4,
        charge_thirds: -1,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 2,
        lifetime_s: None,
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "c",
        symbol: "c",
        name: "kwark powabny",
        mass: 1_270.0,
        charge_thirds: 2,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 2,
        lifetime_s: None,
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "b",
        symbol: "b",
        name: "kwark piękny",
        mass: 4_180.0,
        charge_thirds: -1,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 3,
        lifetime_s: None,
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    // Kwark szczytowy jako jedyny ma czas życia, mimo że niesie kolor: rozpada się
    // szybciej (~5·10⁻²⁵ s), niż zdąży się zhadronizować (~3·10⁻²⁴ s). To nie jest
    // niekonsekwencja tablicy, tylko powód, dla którego szczytowy jest jedynym
    // kwarkiem obserwowanym „gołym".
    Data {
        id: "t",
        symbol: "t",
        name: "kwark szczytowy",
        mass: 172_690.0,
        charge_thirds: 2,
        spin_doubled: 1,
        color: Color::Triplet,
        family: Family::Quark,
        generation: 3,
        lifetime_s: Some(4.6e-25),
        baryon_thirds: 1,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "e",
        symbol: "e⁻",
        name: "elektron",
        mass: 0.510_998_950_0,
        charge_thirds: -3,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 1,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Electron),
        self_conjugate: false,
    },
    Data {
        id: "mu",
        symbol: "μ⁻",
        name: "mion",
        mass: 105.658_375_5,
        charge_thirds: -3,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 2,
        lifetime_s: Some(2.196_981_1e-6),
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Muon),
        self_conjugate: false,
    },
    Data {
        id: "tau",
        symbol: "τ⁻",
        name: "taon",
        mass: 1_776.86,
        charge_thirds: -3,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 3,
        lifetime_s: Some(2.903e-13),
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Tau),
        self_conjugate: false,
    },
    Data {
        id: "nu_e",
        symbol: "νₑ",
        name: "neutrino elektronowe",
        mass: NEUTRINO_MASS,
        charge_thirds: 0,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 1,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Electron),
        self_conjugate: false,
    },
    Data {
        id: "nu_mu",
        symbol: "ν_μ",
        name: "neutrino mionowe",
        mass: NEUTRINO_MASS,
        charge_thirds: 0,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 2,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Muon),
        self_conjugate: false,
    },
    Data {
        id: "nu_tau",
        symbol: "ν_τ",
        name: "neutrino taonowe",
        mass: NEUTRINO_MASS,
        charge_thirds: 0,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Lepton,
        generation: 3,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 1,
        flavour: Some(Flavour::Tau),
        self_conjugate: false,
    },
    Data {
        id: "gamma",
        symbol: "γ",
        name: "foton",
        mass: 0.0,
        charge_thirds: 0,
        spin_doubled: 2,
        color: Color::Neutral,
        family: Family::Gauge,
        generation: 0,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: true,
    },
    Data {
        id: "g",
        symbol: "g",
        name: "gluon",
        mass: 0.0,
        charge_thirds: 0,
        spin_doubled: 2,
        color: Color::Octet,
        family: Family::Gauge,
        generation: 0,
        lifetime_s: None,
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: true,
    },
    Data {
        id: "W",
        symbol: "W⁺",
        name: "bozon W",
        mass: 80_377.0,
        charge_thirds: 3,
        spin_doubled: 2,
        color: Color::Neutral,
        family: Family::Gauge,
        generation: 0,
        lifetime_s: Some(3.157e-25),
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "Z",
        symbol: "Z",
        name: "bozon Z",
        mass: 91_187.6,
        charge_thirds: 0,
        spin_doubled: 2,
        color: Color::Neutral,
        family: Family::Gauge,
        generation: 0,
        lifetime_s: Some(2.638e-25),
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: true,
    },
    Data {
        id: "H",
        symbol: "H",
        name: "bozon Higgsa",
        mass: 125_250.0,
        charge_thirds: 0,
        spin_doubled: 0,
        color: Color::Neutral,
        family: Family::Scalar,
        generation: 0,
        lifetime_s: Some(2.06e-22),
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: true,
    },
    Data {
        id: "p",
        symbol: "p",
        name: "proton",
        mass: 938.272_088_16,
        charge_thirds: 3,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Hadron,
        generation: 0,
        lifetime_s: None,
        baryon_thirds: 3,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "n",
        symbol: "n",
        name: "neutron",
        mass: 939.565_420_52,
        charge_thirds: 0,
        spin_doubled: 1,
        color: Color::Neutral,
        family: Family::Hadron,
        generation: 0,
        lifetime_s: Some(878.4),
        baryon_thirds: 3,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "pi",
        symbol: "π⁺",
        name: "pion naładowany",
        mass: 139.570_39,
        charge_thirds: 3,
        spin_doubled: 0,
        color: Color::Neutral,
        family: Family::Hadron,
        generation: 0,
        lifetime_s: Some(2.6033e-8),
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: false,
    },
    Data {
        id: "pi0",
        symbol: "π⁰",
        name: "pion obojętny",
        mass: 134.9768,
        charge_thirds: 0,
        spin_doubled: 0,
        color: Color::Neutral,
        family: Family::Hadron,
        generation: 0,
        lifetime_s: Some(8.43e-17),
        baryon_thirds: 0,
        lepton: 0,
        flavour: None,
        self_conjugate: true,
    },
];

impl Species {
    pub const ALL: [Species; 21] = [
        Self::Up,
        Self::Down,
        Self::Strange,
        Self::Charm,
        Self::Bottom,
        Self::Top,
        Self::Electron,
        Self::Muon,
        Self::Tau,
        Self::NeutrinoE,
        Self::NeutrinoMu,
        Self::NeutrinoTau,
        Self::Photon,
        Self::Gluon,
        Self::WBoson,
        Self::ZBoson,
        Self::Higgs,
        Self::Proton,
        Self::Neutron,
        Self::PionCharged,
        Self::PionNeutral,
    ];

    /// Siedemnaście cząstek elementarnych — bez hadronów złożonych.
    pub fn elementary() -> impl Iterator<Item = Species> {
        Self::ALL
            .into_iter()
            .filter(|s| s.data().family != Family::Hadron)
    }

    pub fn data(self) -> &'static Data {
        &TABLE[self as usize]
    }

    pub fn mass(self) -> f64 {
        self.data().mass
    }

    pub fn id(self) -> &'static str {
        self.data().id
    }

    /// Gatunek po identyfikatorze, bez znaku ładunku (`"mu"`, nie `"mu-"`).
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.id() == text)
    }

    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    /// Czy cząstka niesie kolor, czyli czy bierze udział w oddziaływaniu silnym.
    pub fn colored(self) -> bool {
        self.data().color != Color::Neutral
    }

    /// Czas życia w fm/c; `None` dla cząstek trwałych.
    pub fn lifetime_fm(self) -> Option<f64> {
        self.data().lifetime_s.map(units::lifetime_to_fm)
    }
}

/// Konkretna cząstka: gatunek i to, czy jest antycząstką.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Particle {
    pub species: Species,
    pub anti: bool,
}

impl Particle {
    /// Cząstka danego gatunku. Dla cząstek samosprzężonych `anti` jest wyciszane
    /// do `false`: „antyfoton" nie jest innym obiektem, tylko innym zapisem tego
    /// samego, a dwa zapisy jednego stanu psułyby porównania i zliczanie.
    pub fn new(species: Species, anti: bool) -> Self {
        Self {
            species,
            anti: anti && !species.data().self_conjugate,
        }
    }

    pub fn of(species: Species) -> Self {
        Self::new(species, false)
    }

    pub fn anti_of(species: Species) -> Self {
        Self::new(species, true)
    }

    pub fn data(self) -> &'static Data {
        self.species.data()
    }

    /// Antycząstka. Dla samosprzężonych zwraca to samo.
    pub fn conjugate(self) -> Self {
        Self::new(self.species, !self.anti)
    }

    pub fn mass(self) -> f64 {
        self.data().mass
    }

    /// Ładunek w trzecich `e` — liczba całkowita, więc suma jest dokładna.
    pub fn charge_thirds(self) -> i32 {
        self.signed(self.data().charge_thirds)
    }

    /// Ładunek w jednostkach `e`.
    pub fn charge(self) -> f64 {
        self.charge_thirds() as f64 / 3.0
    }

    pub fn baryon_thirds(self) -> i32 {
        self.signed(self.data().baryon_thirds)
    }

    pub fn lepton(self) -> i32 {
        self.signed(self.data().lepton)
    }

    pub fn flavour(self) -> Option<Flavour> {
        self.data().flavour
    }

    /// Liczba leptonowa danego zapachu — osobno zachowywana w każdym rozpadzie.
    pub fn lepton_of(self, flavour: Flavour) -> i32 {
        match self.flavour() {
            Some(f) if f == flavour => self.lepton(),
            _ => 0,
        }
    }

    pub fn colored(self) -> bool {
        self.species.colored()
    }

    pub fn massless(self) -> bool {
        self.mass() == 0.0
    }

    pub fn lifetime_fm(self) -> Option<f64> {
        self.species.lifetime_fm()
    }

    /// Symbol z właściwym znakiem: `μ⁻` kontra `μ⁺`.
    pub fn symbol(self) -> String {
        let base = self.data().symbol;
        if !self.anti {
            return base.to_string();
        }
        match self.data().charge_thirds {
            // Antycząstka naładowana: odwracamy znak w symbolu.
            _ if base.ends_with('⁻') => format!("{}⁺", base.trim_end_matches('⁻')),
            _ if base.ends_with('⁺') => format!("{}⁻", base.trim_end_matches('⁺')),
            // Antycząstka obojętna: kreska nad symbolem (ν̄, n̄, ū).
            _ => format!("{base}\u{0304}"),
        }
    }

    /// Identyfikator do wiersza poleceń: `e-`, `e+`, `nu_mu~`, `p`, `pi+`, `u~`.
    ///
    /// Znakiem `+`/`−` oznaczamy tylko cząstki o **całkowitym** ładunku. Kwark ma
    /// ładunek ułamkowy, więc `u+` sugerowałoby jednostkowy ładunek dodatni, którego
    /// kwark nie ma; antykwark zapisujemy więc kreską `u~`, tak jak w tekście `ū`.
    pub fn id(self) -> String {
        let base = self.data().id;
        if self.data().self_conjugate {
            return base.to_string();
        }
        let sign = match self.charge_thirds() {
            3 => "+",
            -3 => "-",
            _ if self.anti => "~",
            _ => "",
        };
        format!("{base}{sign}")
    }

    /// Odwrotność [`Particle::id`]. Przyjmuje `e-`, `e+`, `mu+`, `nu_e~`, `p`, `u~`.
    pub fn parse(text: &str) -> Option<Self> {
        let (stem, suffix) = match text.chars().last() {
            Some(last @ ('+' | '-' | '~')) => (&text[..text.len() - 1], Some(last)),
            _ => (text, None),
        };
        let species = Species::parse(stem)?;
        let base_charge = species.data().charge_thirds;
        let anti = match suffix {
            None => false,
            Some('~') => true,
            // Znak w identyfikatorze niesie znaczenie, więc musi zgadzać się
            // z ładunkiem gatunku. Bez tego `pi-` znaczyłoby „antypion" tylko przez
            // przypadkową zgodność, a `gamma+` przeszłoby jako foton.
            Some('+') if base_charge == 3 => false,
            Some('+') if base_charge == -3 => true,
            Some('-') if base_charge == -3 => false,
            Some('-') if base_charge == 3 => true,
            Some(_) => return None,
        };
        Some(Self::new(species, anti))
    }

    fn signed(self, value: i32) -> i32 {
        if self.anti {
            -value
        } else {
            value
        }
    }

    /// Gatunek w dolnych 16 bitach, znacznik antycząstki w bicie 16.
    ///
    /// Checkpoint zapisuje tożsamość, nie masę: masa wynika z tablicy i po
    /// wznowieniu musi być taka sama jak przy starcie, a nie ta, którą ktoś
    /// wpisał ręcznie do pliku binarnego.
    pub fn pack(self) -> u32 {
        (self.species as u32) | (u32::from(self.anti) << 16)
    }

    pub fn unpack(code: u32) -> Option<Self> {
        let species = Species::from_index((code & 0xffff) as usize)?;
        Some(Self::new(species, (code >> 16) & 1 == 1))
    }
}

impl std::fmt::Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.symbol())
    }
}

/// Cząstka zapisuje się w JSON-ie jako `"e-"`, a nie jako obiekt z dwoma polami.
///
/// `config.json` jest czytany i poprawiany ręcznie — to jest jego jedyny powód
/// istnienia obok pliku binarnego. Zapis wyprowadzony automatycznie dałby
/// `{"species":"Electron","anti":false}`: poprawny, dwukrotnie dłuższy i przy
/// kilkunastu składnikach mieszanki nie do ogarnięcia okiem.
impl serde::Serialize for Particle {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.id())
    }
}

impl<'de> serde::Deserialize<'de> for Particle {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Particle::parse(&text).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "nie znam cząstki „{text}”; przykłady: e-, e+, mu-, p, n, pi+, gamma, u~"
            ))
        })
    }
}

/// Spin zapisany po ludzku: `½`, `1`, `0`.
pub fn spin_text(spin_doubled: u32) -> String {
    match spin_doubled % 2 {
        0 => (spin_doubled / 2).to_string(),
        _ if spin_doubled == 1 => "½".to_string(),
        _ => format!("{}½", spin_doubled / 2),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wiersz tablicy MUSI odpowiadać wariantowi enuma o tym samym indeksie.
    /// Cała reszta modułu polega na `TABLE[self as usize]`, więc przestawienie
    /// jednego wariantu bez przestawienia wiersza dałoby elektronowi masę taonu —
    /// i nic by nie zaprotestowało.
    #[test]
    fn table_rows_line_up_with_the_enum() {
        assert_eq!(TABLE.len(), Species::ALL.len());
        for (index, species) in Species::ALL.into_iter().enumerate() {
            assert_eq!(species as usize, index, "{species:?} jest nie na swoim miejscu");
        }
        assert_eq!(Species::Electron.id(), "e");
        assert_eq!(Species::Higgs.id(), "H");
        assert_eq!(Species::PionNeutral.id(), "pi0");
    }

    #[test]
    fn identifiers_are_unique() {
        let mut ids: Vec<&str> = Species::ALL.iter().map(|s| s.id()).collect();
        ids.sort_unstable();
        let count = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), count, "zdublowany identyfikator w tablicy");
    }

    /// Model Standardowy ma dokładnie siedemnaście cząstek elementarnych.
    #[test]
    fn there_are_seventeen_elementary_particles() {
        let elementary: Vec<Species> = Species::elementary().collect();
        assert_eq!(elementary.len(), 17);
        let quarks = elementary
            .iter()
            .filter(|s| s.data().family == Family::Quark)
            .count();
        let leptons = elementary
            .iter()
            .filter(|s| s.data().family == Family::Lepton)
            .count();
        let gauge = elementary
            .iter()
            .filter(|s| s.data().family == Family::Gauge)
            .count();
        let scalar = elementary
            .iter()
            .filter(|s| s.data().family == Family::Scalar)
            .count();
        assert_eq!((quarks, leptons, gauge, scalar), (6, 6, 4, 1));
    }

    /// Każde pokolenie fermionów ma dwa kwarki i dwa leptony.
    #[test]
    fn every_generation_is_complete() {
        for generation in 1..=3u8 {
            let quarks = Species::ALL
                .iter()
                .filter(|s| s.data().family == Family::Quark && s.data().generation == generation)
                .count();
            let leptons = Species::ALL
                .iter()
                .filter(|s| s.data().family == Family::Lepton && s.data().generation == generation)
                .count();
            assert_eq!((quarks, leptons), (2, 2), "pokolenie {generation}");
        }
    }

    /// Ładunki kwarków w trzecich muszą składać się na całkowite ładunki hadronów:
    /// `uud` daje proton `+1`, `udd` neutron `0`, `ud̄` pion `+1`. To sprawdza
    /// jednocześnie tablicę kwarków i tablicę hadronów, jednym rachunkiem.
    #[test]
    fn quark_content_reproduces_hadron_charges() {
        let u = Particle::of(Species::Up).charge_thirds();
        let d = Particle::of(Species::Down).charge_thirds();
        let anti_d = Particle::anti_of(Species::Down).charge_thirds();

        assert_eq!(u + u + d, Particle::of(Species::Proton).charge_thirds());
        assert_eq!(u + d + d, Particle::of(Species::Neutron).charge_thirds());
        assert_eq!(u + anti_d, Particle::of(Species::PionCharged).charge_thirds());
    }

    /// To samo dla liczby barionowej: trzy kwarki dają barion, kwark z antykwarkiem
    /// mezon o liczbie barionowej zero.
    #[test]
    fn quark_content_reproduces_baryon_numbers() {
        let q = Particle::of(Species::Up).baryon_thirds();
        assert_eq!(3 * q, Particle::of(Species::Proton).baryon_thirds());
        assert_eq!(
            Particle::of(Species::Up).baryon_thirds()
                + Particle::anti_of(Species::Down).baryon_thirds(),
            Particle::of(Species::PionCharged).baryon_thirds()
        );
    }

    #[test]
    fn antiparticles_flip_every_additive_charge() {
        for species in Species::ALL {
            let p = Particle::of(species);
            let a = p.conjugate();
            assert_eq!(a.mass(), p.mass(), "{species:?}: masa się zmieniła");
            if species.data().self_conjugate {
                assert_eq!(a, p, "{species:?} jest samosprzężona");
                continue;
            }
            assert_eq!(a.charge_thirds(), -p.charge_thirds(), "{species:?}");
            assert_eq!(a.baryon_thirds(), -p.baryon_thirds(), "{species:?}");
            assert_eq!(a.lepton(), -p.lepton(), "{species:?}");
        }
    }

    /// Podwójne sprzężenie musi wracać do punktu wyjścia — dla wszystkich, także
    /// dla samosprzężonych.
    #[test]
    fn conjugation_is_an_involution() {
        for species in Species::ALL {
            for anti in [false, true] {
                let p = Particle::new(species, anti);
                assert_eq!(p.conjugate().conjugate(), p, "{species:?} anti={anti}");
            }
        }
    }

    /// Cząstka samosprzężona nie ma prawa nieść żadnego ładunku addytywnego —
    /// inaczej „jest swoją antycząstką" byłoby sprzeczne z „antycząstka ma
    /// przeciwny ładunek".
    #[test]
    fn self_conjugate_particles_carry_no_additive_charge() {
        for species in Species::ALL {
            let d = species.data();
            if !d.self_conjugate {
                continue;
            }
            assert_eq!(d.charge_thirds, 0, "{species:?}");
            assert_eq!(d.baryon_thirds, 0, "{species:?}");
            assert_eq!(d.lepton, 0, "{species:?}");
        }
    }

    #[test]
    fn identifiers_survive_a_round_trip() {
        for species in Species::ALL {
            for anti in [false, true] {
                let p = Particle::new(species, anti);
                let text = p.id();
                assert_eq!(Particle::parse(&text), Some(p), "identyfikator „{text}”");
            }
        }
    }

    #[test]
    fn packed_identity_survives_a_round_trip() {
        for species in Species::ALL {
            for anti in [false, true] {
                let p = Particle::new(species, anti);
                assert_eq!(Particle::unpack(p.pack()), Some(p), "{species:?} anti={anti}");
            }
        }
        assert!(Particle::unpack(0xffff).is_none());
    }

    /// Znak w identyfikatorze niesie znaczenie, więc `e+` i `e-` muszą być różnymi
    /// cząstkami, a nie tym samym wpisem z ozdobnikiem.
    #[test]
    fn charge_sign_in_identifiers_is_meaningful() {
        let electron = Particle::parse("e-").unwrap();
        let positron = Particle::parse("e+").unwrap();
        assert_eq!(electron.charge_thirds(), -3);
        assert_eq!(positron.charge_thirds(), 3);
        assert!(!electron.anti && positron.anti);

        // Bez znaku: gatunek w postaci podstawowej.
        assert_eq!(Particle::parse("mu"), Some(Particle::of(Species::Muon)));
        // Znak przy cząstce obojętnej jest błędem, a nie życzeniem do zignorowania.
        assert!(Particle::parse("gamma+").is_none());
        assert!(Particle::parse("nie-ma-takiej").is_none());
    }

    #[test]
    fn symbols_show_the_right_sign() {
        assert_eq!(Particle::of(Species::Electron).symbol(), "e⁻");
        assert_eq!(Particle::anti_of(Species::Electron).symbol(), "e⁺");
        assert_eq!(Particle::of(Species::Photon).symbol(), "γ");
        assert_eq!(Particle::anti_of(Species::Photon).symbol(), "γ");
        assert_eq!(Particle::anti_of(Species::NeutrinoE).symbol(), "νₑ\u{0304}");
    }

    /// Masy muszą rosnąć z pokoleniem — to jest wzorzec, którego Model Standardowy
    /// nie tłumaczy, ale który tablica ma odtwarzać.
    #[test]
    fn charged_fermion_masses_grow_with_generation() {
        let charged = |family: Family| -> Vec<f64> {
            let mut rows: Vec<(u8, f64)> = Species::ALL
                .iter()
                .map(|s| s.data())
                .filter(|d| d.family == family && d.charge_thirds != 0)
                .map(|d| (d.generation, d.mass))
                .collect();
            rows.sort_by_key(|(g, _)| *g);
            rows.into_iter().map(|(_, m)| m).collect()
        };
        for masses in [charged(Family::Lepton), charged(Family::Quark)] {
            assert!(
                masses.windows(2).all(|w| w[0] < w[1]),
                "masy nie rosną: {masses:?}"
            );
        }
    }

    /// Kolor niesie tylko to, co ma go nieść: kwarki i gluony.
    #[test]
    fn only_quarks_and_gluons_are_colored() {
        for species in Species::ALL {
            let expected = matches!(species.data().family, Family::Quark) || species == Species::Gluon;
            assert_eq!(species.colored(), expected, "{species:?}");
        }
    }

    /// Neutron żyje kwadrans, mion mikrosekundę, a foton nie rozpada się wcale.
    /// Test na to, że czasy życia nie pomyliły się o rzędy wielkości przy przejściu
    /// na fm/c.
    #[test]
    fn lifetimes_land_in_the_right_decades() {
        assert!(Species::Photon.lifetime_fm().is_none());
        assert!(Species::Electron.lifetime_fm().is_none());
        assert!(Species::Proton.lifetime_fm().is_none());

        let muon = Species::Muon.lifetime_fm().unwrap();
        assert!((1e17..1e18).contains(&muon), "τ_μ = {muon} fm/c");
        // Neutron: kwadrans to ~2,6·10²⁶ fm/c.
        let neutron = Species::Neutron.lifetime_fm().unwrap();
        assert!(neutron > muon * 1e8, "neutron żyje za krótko: {neutron}");
        // Pion obojętny rozpada się na dystansie dziesiątków nanometrów.
        let pi0 = Species::PionNeutral.lifetime_fm().unwrap();
        assert!((1e7..1e8).contains(&pi0), "τ_π⁰ = {pi0} fm/c");
    }

    /// Bezmasowe są dokładnie te i tylko te. Lista jest wypisana wprost, bo
    /// przypadkowe wyzerowanie masy w tablicy zamieniłoby cząstkę w coś, co porusza
    /// się z prędkością światła — a to zmiana, której nie widać po wykresie.
    #[test]
    fn massless_particles_are_exactly_the_expected_ones() {
        let massless: Vec<&str> = Species::ALL
            .iter()
            .filter(|s| Particle::of(**s).massless())
            .map(|s| s.id())
            .collect();
        assert_eq!(massless, ["nu_e", "nu_mu", "nu_tau", "gamma", "g"]);
    }

    #[test]
    fn spin_reads_naturally() {
        assert_eq!(spin_text(0), "0");
        assert_eq!(spin_text(1), "½");
        assert_eq!(spin_text(2), "1");
        assert_eq!(spin_text(3), "1½");
    }

    /// Fermiony mają spin połówkowy, bozony całkowity. Podział na rodziny musi się
    /// z tym zgadzać, bo na nim opiera się cała statystyka.
    #[test]
    fn families_agree_with_spin_statistics() {
        for species in Species::ALL {
            let d = species.data();
            let half_integer = d.spin_doubled % 2 == 1;
            let fermion = matches!(d.family, Family::Quark | Family::Lepton)
                || (d.family == Family::Hadron && d.baryon_thirds != 0);
            assert_eq!(half_integer, fermion, "{species:?} ma spin niezgodny z rodziną");
        }
    }
}
