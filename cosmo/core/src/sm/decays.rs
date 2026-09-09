//! Rozpady i anihilacja — warstwa stochastyczna nad całkowaniem.
//!
//! To tutaj oddziaływanie słabe robi coś obserwowalnego. Nie jest potencjałem
//! statycznym ani checkboxem w panelu — wymiana ciężkiego bozonu zmienia zapach,
//! więc skutkiem jest rozpad, nie przekaz pędu.
//!
//! # Co jest liczone dokładnie
//!
//! - **Zachowanie ładunku, liczby barionowej i liczb leptonowych.** Są liczbami
//!   całkowitymi, więc rozpad albo je zachowuje co do bitu, albo tablica kanałów
//!   jest błędna. Test [`tests::every_channel_conserves_every_quantum_number`]
//!   sprawdza to dla **każdego** kanału w tablicy, a nie dla wylosowanej próbki.
//! - **Zachowanie czteropędu.** Produkty powstają w układzie spoczynkowym matki,
//!   z dokładnego wzoru na `p*`, i są pchnięte do laboratorium jednym wspólnym
//!   pchnięciem Lorentza.
//! - **Dylatacja czasu.** Prawdopodobieństwo rozpadu w kroku `dt` liczone jest
//!   z czasu **własnego** `dτ = dt/γ`. Szybki mion żyje dłużej i to nie jest tu
//!   dołożone ręcznie — wychodzi z tego, że `γ` bierze się z pędu, który już jest
//!   zmienną stanu.
//!
//! # Co jest przybliżone i dlaczego
//!
//! - **Rozkłady kątowe są izotropowe.** Prawdziwy rozpad ma rozkład zależny od
//!   spinu (widmo Michela w rozpadzie mionu, korelacje spinowe, naruszenie
//!   parzystości). Ten model nie śledzi spinu, więc udawanie tych rozkładów byłoby
//!   wymyślaniem liczb. Rozkład płaski po przestrzeni fazowej jest jedynym
//!   wyborem, który niczego nie zmyśla.
//! - **Kanały wielociałowe (>3 produkty) są zastąpione najbliższym kanałem
//!   trzyciałowym** o tych samych liczbach kwantowych. Dotyczy to głównie
//!   hadronowych rozpadów taonu.
//! - **Kanały hadronowe bozonów są reprezentowane jedną parą kwark-antykwark.**
//!   Ładunek i liczba barionowa się zgadzają, skład zapachowy nie jest rozstrzygany.
//! - **Kanały z cząstką wirtualną nie istnieją.** `H → WW*` wymaga bozonu poza
//!   powłoką masy, a ten model zna tylko cząstki rzeczywiste. Takie kanały są
//!   odsiewane jako niedozwolone energetycznie, a ich prawdopodobieństwo rozkłada
//!   się na kanały dostępne — patrz [`Decays::choose_channel`].

use crate::rng::Rng;
use crate::sm::config::DecayConfig;
use crate::sm::kinematics::{self, FourMomentum};
use crate::sm::particles::{Particle, Species};
use crate::sm::state::{Birth, State};
use crate::sm::units;
use crate::vec3::Vec3;

/// Produkt rozpadu, zapisany **względem** cząstki macierzystej.
///
/// `anti` znaczy „antycząstka, gdy matka jest cząstką". Dla matki będącej
/// antycząstką cały kanał jest sprzęgany. Dzięki temu `μ⁻ → e⁻ν̄ₑν_μ` i
/// `μ⁺ → e⁺νₑν̄_μ` to jeden wiersz tablicy, a nie dwa, które mogą się rozjechać.
#[derive(Clone, Copy, Debug)]
pub struct Product {
    pub species: Species,
    pub anti: bool,
}

const fn p(species: Species) -> Product {
    Product {
        species,
        anti: false,
    }
}

const fn a(species: Species) -> Product {
    Product {
        species,
        anti: true,
    }
}

/// Jeden kanał rozpadu: współczynnik rozgałęzienia i lista produktów.
#[derive(Clone, Copy, Debug)]
pub struct Channel {
    pub branching: f64,
    pub products: &'static [Product],
}

const MUON: [Channel; 1] = [Channel {
    // μ⁻ → e⁻ ν̄ₑ ν_μ — praktycznie jedyny kanał (pozostałe to rzędu 10⁻².
    // z dodatkowym fotonem, czyli ten sam stan końcowy plus promieniowanie).
    branching: 1.0,
    products: &[p(Species::Electron), a(Species::NeutrinoE), p(Species::NeutrinoMu)],
}];

const TAU: [Channel; 4] = [
    Channel {
        branching: 0.1785,
        products: &[p(Species::Electron), a(Species::NeutrinoE), p(Species::NeutrinoTau)],
    },
    Channel {
        branching: 0.1737,
        products: &[p(Species::Muon), a(Species::NeutrinoMu), p(Species::NeutrinoTau)],
    },
    Channel {
        branching: 0.1082,
        products: &[a(Species::PionCharged), p(Species::NeutrinoTau)],
    },
    // 25,5% to π⁻π⁰ν, a kolejne ~28% to kanały z trzema i więcej pionami. Tych
    // ostatnich ten model nie umie rozegrać (przestrzeń fazowa jest tu policzona
    // do trzech ciał), więc ich prawdopodobieństwo dostaje kanał dwupionowy.
    // Skutek: pionów jest za mało, a ich energie za wysokie.
    Channel {
        branching: 0.5396,
        products: &[
            a(Species::PionCharged),
            p(Species::PionNeutral),
            p(Species::NeutrinoTau),
        ],
    },
];

const NEUTRON: [Channel; 1] = [Channel {
    branching: 1.0,
    products: &[p(Species::Proton), p(Species::Electron), a(Species::NeutrinoE)],
}];

const PION_CHARGED: [Channel; 2] = [
    Channel {
        branching: 0.999_877,
        products: &[a(Species::Muon), p(Species::NeutrinoMu)],
    },
    // Kanał elektronowy jest 10⁴ razy rzadszy MIMO większej przestrzeni fazowej —
    // to jest podręcznikowy skutek tego, że oddziaływanie słabe sprzęga się do
    // skrętności. Tablica go zna, choć model nie zna spinu.
    Channel {
        branching: 0.000_123,
        products: &[a(Species::Electron), p(Species::NeutrinoE)],
    },
];

const PION_NEUTRAL: [Channel; 2] = [
    Channel {
        branching: 0.988_23,
        products: &[p(Species::Photon), p(Species::Photon)],
    },
    Channel {
        branching: 0.011_77,
        products: &[p(Species::Electron), a(Species::Electron), p(Species::Photon)],
    },
];

const W_BOSON: [Channel; 4] = [
    Channel {
        branching: 0.1071,
        products: &[a(Species::Electron), p(Species::NeutrinoE)],
    },
    Channel {
        branching: 0.1063,
        products: &[a(Species::Muon), p(Species::NeutrinoMu)],
    },
    Channel {
        branching: 0.1138,
        products: &[a(Species::Tau), p(Species::NeutrinoTau)],
    },
    // Kanał hadronowy jako jedna para kwark-antykwark. Ładunek się zgadza
    // (`+2/3 + 1/3 = +1`), skład zapachowy nie jest rozstrzygany.
    Channel {
        branching: 0.6728,
        products: &[p(Species::Up), a(Species::Down)],
    },
];

const Z_BOSON: [Channel; 5] = [
    Channel {
        branching: 0.033_63,
        products: &[p(Species::Electron), a(Species::Electron)],
    },
    Channel {
        branching: 0.033_66,
        products: &[p(Species::Muon), a(Species::Muon)],
    },
    Channel {
        branching: 0.033_70,
        products: &[p(Species::Tau), a(Species::Tau)],
    },
    // Kanał niewidzialny: 20% szerokości Z to neutrina. Trzy zapachy są tu
    // reprezentowane jednym — model nie zna oscylacji, więc rozróżnienie i tak
    // nie miałoby dalszych skutków.
    Channel {
        branching: 0.2000,
        products: &[p(Species::NeutrinoE), a(Species::NeutrinoE)],
    },
    Channel {
        branching: 0.6991,
        products: &[p(Species::Bottom), a(Species::Bottom)],
    },
];

const HIGGS: [Channel; 7] = [
    // 58,09% z PDG plus 0,19% kanałów pominiętych (Zγ, μ⁺μ⁻) — dosypane tutaj,
    // żeby suma wyszła dokładnie jeden bez zmyślania wartości pozostałych kanałów.
    Channel {
        branching: 0.5828,
        products: &[p(Species::Bottom), a(Species::Bottom)],
    },
    // H → WW i H → ZZ mają razem ~24% szerokości, ale wymagają bozonu poza powłoką
    // masy: `2·m_W = 160,7 GeV > m_H = 125,2 GeV`. Ten model zna tylko cząstki
    // rzeczywiste, więc oba kanały zostaną ODSIANE jako niedozwolone. Wypisujemy je
    // mimo to, żeby powód ich nieobecności był widoczny w kodzie, a nie zgadywany
    // z tego, czego w tablicy brakuje.
    Channel {
        branching: 0.2152,
        products: &[p(Species::WBoson), a(Species::WBoson)],
    },
    Channel {
        branching: 0.0264,
        products: &[p(Species::ZBoson), p(Species::ZBoson)],
    },
    Channel {
        branching: 0.0818,
        products: &[p(Species::Gluon), p(Species::Gluon)],
    },
    Channel {
        branching: 0.0627,
        products: &[p(Species::Tau), a(Species::Tau)],
    },
    Channel {
        branching: 0.0288,
        products: &[p(Species::Charm), a(Species::Charm)],
    },
    Channel {
        branching: 0.0023,
        products: &[p(Species::Photon), p(Species::Photon)],
    },
];

const TOP: [Channel; 1] = [Channel {
    branching: 1.0,
    products: &[p(Species::WBoson), p(Species::Bottom)],
}];

/// Kanały rozpadu gatunku. Pusta lista znaczy „nie rozpada się".
pub fn channels(species: Species) -> &'static [Channel] {
    match species {
        Species::Muon => &MUON,
        Species::Tau => &TAU,
        Species::Neutron => &NEUTRON,
        Species::PionCharged => &PION_CHARGED,
        Species::PionNeutral => &PION_NEUTRAL,
        Species::WBoson => &W_BOSON,
        Species::ZBoson => &Z_BOSON,
        Species::Higgs => &HIGGS,
        Species::Top => &TOP,
        _ => &[],
    }
}

/// Produkty kanału dla konkretnej matki — ze sprzężeniem, gdy matka jest antycząstką.
pub fn products_for(parent: Particle, channel: &Channel) -> Vec<Particle> {
    channel
        .products
        .iter()
        .map(|product| {
            let base = Particle::new(product.species, product.anti);
            if parent.anti {
                base.conjugate()
            } else {
                base
            }
        })
        .collect()
}

/// Czy kanał jest dozwolony energetycznie: suma mas produktów nie przekracza matki.
pub fn channel_allowed(parent: Particle, channel: &Channel) -> bool {
    let sum: f64 = channel
        .products
        .iter()
        .map(|product| product.species.mass())
        .sum();
    // Kanał dwuciałowy dokładnie na progu jest dozwolony (produkty spoczywają),
    // trzyciałowy też — dlatego nierówność jest nieostra.
    sum <= parent.mass()
}

/// Ile cząstek zmieniło stan w jednym kroku.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub decayed: usize,
    pub annihilated: usize,
    pub born: usize,
    /// Czy limit liczby cząstek powstrzymał choć jeden rozpad.
    pub capped: bool,
}

impl Outcome {
    pub fn happened(self) -> bool {
        self.decayed > 0 || self.annihilated > 0
    }
}

/// Sumaryczny licznik przez cały bieg.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tally {
    pub decayed: u64,
    pub annihilated: u64,
    pub born: u64,
    pub capped_steps: u64,
}

impl Tally {
    fn absorb(&mut self, outcome: Outcome) {
        self.decayed += outcome.decayed as u64;
        self.annihilated += outcome.annihilated as u64;
        self.born += outcome.born as u64;
        self.capped_steps += u64::from(outcome.capped);
    }
}

pub struct Decays {
    cfg: DecayConfig,
    rng: Rng,
    tally: Tally,
}

impl Decays {
    pub fn new(cfg: DecayConfig) -> Self {
        Self {
            rng: Rng::seeded(cfg.seed),
            cfg,
            tally: Tally::default(),
        }
    }

    pub fn tally(&self) -> Tally {
        self.tally
    }

    pub fn config(&self) -> &DecayConfig {
        &self.cfg
    }

    pub fn apply_runtime(&mut self, live: &DecayConfig) {
        self.cfg.enabled = live.enabled;
        self.cfg.annihilation = live.annihilation;
        self.cfg.pair_radius = live.pair_radius;
    }

    /// Rozegraj jeden krok warstwy stochastycznej.
    ///
    /// `dt` jest w fm/c i jest czasem **laboratoryjnym**; przeliczenie na czas
    /// własny każdej cząstki dzieje się wewnątrz.
    pub fn step(&mut self, state: &mut State, dt: f64, max_particles: usize) -> Outcome {
        let mut outcome = Outcome::default();
        if !self.cfg.enabled || state.is_empty() || dt <= 0.0 {
            return outcome;
        }

        let n = state.n();
        let mut removed = vec![false; n];
        let mut born: Vec<Birth> = Vec::new();
        let mut budget = max_particles.saturating_sub(n);

        for (i, gone) in removed.iter_mut().enumerate() {
            let parent = state.kinds[i];
            let Some(lifetime) = parent.lifetime_fm() else {
                continue;
            };
            if !self.decides_to_decay(state, i, lifetime, dt) {
                continue;
            }
            let Some(channel) = self.choose_channel(parent) else {
                continue;
            };
            let daughters = products_for(parent, channel);
            // Rozpad dokłada `len − 1` cząstek. Przy wyczerpanym budżecie cząstka
            // zostaje nierozpadnięta zamiast zniknąć — zniknięcie łamałoby bilans
            // energii, a limit jest ograniczeniem pamięci, nie fizyki.
            if daughters.len() > budget + 1 {
                outcome.capped = true;
                continue;
            }
            let momenta = self.decay_momenta(state.four_momentum(i), &daughters);
            let Some(momenta) = momenta else {
                continue;
            };
            *gone = true;
            budget = budget + 1 - daughters.len();
            for (particle, momentum) in daughters.into_iter().zip(momenta) {
                born.push(Birth {
                    particle,
                    position: state.positions[i],
                    momentum,
                });
            }
            outcome.decayed += 1;
        }

        if self.cfg.annihilation {
            let mut capped = outcome.capped;
            outcome.annihilated =
                self.annihilate(state, dt, &mut removed, &mut born, &mut budget, &mut capped);
            outcome.capped = capped;
        }

        outcome.born = born.len();
        state.replace(&removed, &born);
        self.tally.absorb(outcome);
        outcome
    }

    /// Czy cząstka rozpada się w tym kroku.
    ///
    /// `P = 1 − exp(−dτ/τ)`, gdzie `dτ = dt/γ`. Postać wykładnicza, a nie `dt/τ`,
    /// bo przy dużym kroku (a bieg nastawiony na rozpady ma krok rzędu czasu życia)
    /// liniowe przybliżenie dawałoby prawdopodobieństwo powyżej jedności — czyli
    /// rozpad wszystkiego w pierwszym kroku zamiast rozkładu wykładniczego.
    fn decides_to_decay(&mut self, state: &State, index: usize, lifetime: f64, dt: f64) -> bool {
        let mass = state.kinds[index].mass();
        let gamma = kinematics::gamma(mass, state.momenta[index]).unwrap_or(1.0);
        let proper_dt = dt / gamma;
        let probability = -(-proper_dt / lifetime).exp_m1();
        self.rng.unit() < probability
    }

    /// Wylosuj kanał spośród **dozwolonych energetycznie**.
    ///
    /// Kanały niedozwolone (`H → WW`) są odsiewane, a losowanie odbywa się po sumie
    /// pozostałych. Alternatywą byłoby losowanie po pełnej tablicy i porzucanie
    /// rozpadu przy trafieniu w kanał zabroniony — co wydłużałoby zmierzony czas
    /// życia cząstki o czynnik zależny od tego, ile kanałów akurat odpada. Byłby to
    /// błąd cichy i systematyczny.
    fn choose_channel(&mut self, parent: Particle) -> Option<&'static Channel> {
        let all = channels(parent.species);
        let total: f64 = all
            .iter()
            .filter(|c| channel_allowed(parent, c))
            .map(|c| c.branching)
            .sum();
        if total <= 0.0 {
            return None;
        }
        let mut draw = self.rng.unit() * total;
        for channel in all.iter().filter(|c| channel_allowed(parent, c)) {
            draw -= channel.branching;
            if draw <= 0.0 {
                return Some(channel);
            }
        }
        // Zaokrąglenia mogą zostawić resztkę — ostatni dozwolony kanał zamyka sumę.
        all.iter().rfind(|c| channel_allowed(parent, c))
    }

    /// Pędy produktów w układzie laboratorium.
    fn decay_momenta(&mut self, parent: FourMomentum, daughters: &[Particle]) -> Option<Vec<Vec3>> {
        let mass = parent.invariant_mass();
        let masses: Vec<f64> = daughters.iter().map(|d| d.mass()).collect();
        let rest_frame = match masses.len() {
            2 => self.two_body(mass, masses[0], masses[1])?,
            3 => self.three_body(mass, &masses)?,
            _ => return None,
        };
        let boost = parent.boost_velocity();
        Some(
            rest_frame
                .into_iter()
                .zip(masses.iter())
                .map(|(momentum, m)| FourMomentum::on_shell(*m, momentum).boost(boost).momentum)
                .collect(),
        )
    }

    /// Rozpad dwuciałowy w układzie spoczynkowym: dwa przeciwne pędy o module `p*`.
    fn two_body(&mut self, parent: f64, m1: f64, m2: f64) -> Option<Vec<Vec3>> {
        let p_star = kinematics::two_body_momentum(parent, m1, m2)?;
        let direction = self.rng.unit_vector();
        Some(vec![direction * p_star, direction * -p_star])
    }

    /// Rozpad trzyciałowy: płaska przestrzeń fazowa, metodą rekurencyjną.
    ///
    /// Rozkładamy `M → 1 + X`, gdzie `X` jest fikcyjną cząstką o masie `m₂₃`
    /// wylosowanej z właściwą wagą, a potem `X → 2 + 3` w jej własnym układzie.
    /// Waga `p*(M, m₁, m₂₃)·p*(m₂₃, m₂, m₃)` jest dokładnie jakobianem płaskiej
    /// przestrzeni fazowej trzech ciał, więc wynik nie jest przybliżeniem tego
    /// rozkładu — jest tym rozkładem.
    ///
    /// Losowanie z odrzuceniem, a nie transformacja odwrotna, bo dystrybuanta tej
    /// wagi nie ma postaci zamkniętej. Kres górny znajdujemy przeglądem, z zapasem.
    fn three_body(&mut self, parent: f64, masses: &[f64]) -> Option<Vec<Vec3>> {
        let (m1, m2, m3) = (masses[0], masses[1], masses[2]);
        if parent < m1 + m2 + m3 {
            return None;
        }
        let low = m2 + m3;
        let high = parent - m1;
        if high <= low {
            // Przestrzeń fazowa jest punktem: wszystko spoczywa względem siebie.
            return Some(vec![crate::vec3::ZERO; 3]);
        }

        let weight = |m23: f64| -> f64 {
            let outer = kinematics::two_body_momentum(parent, m1, m23).unwrap_or(0.0);
            let inner = kinematics::two_body_momentum(m23, m2, m3).unwrap_or(0.0);
            outer * inner
        };

        const SCAN: usize = 96;
        let mut ceiling = 0.0f64;
        for k in 0..=SCAN {
            let m23 = low + (high - low) * k as f64 / SCAN as f64;
            ceiling = ceiling.max(weight(m23));
        }
        if ceiling <= 0.0 {
            return Some(vec![crate::vec3::ZERO; 3]);
        }
        ceiling *= 1.05;

        let mut m23 = low;
        // Przegląd znalazł kres z zapasem 5%, więc odrzucenie kończy się szybko;
        // limit prób chroni tylko przed patologicznym wejściem, a nie przed
        // typowym przebiegiem.
        for _ in 0..1_000 {
            let candidate = self.rng.uniform(low, high);
            if self.rng.unit() * ceiling <= weight(candidate) {
                m23 = candidate;
                break;
            }
        }

        // M → 1 + X
        let p_star = kinematics::two_body_momentum(parent, m1, m23)?;
        let axis = self.rng.unit_vector();
        let first = axis * p_star;
        let compound = FourMomentum::on_shell(m23, axis * -p_star);

        // X → 2 + 3 w układzie X, potem pchnięcie do układu matki.
        let q = kinematics::two_body_momentum(m23, m2, m3)?;
        let inner_axis = self.rng.unit_vector();
        let boost = compound.boost_velocity();
        let second = FourMomentum::on_shell(m2, inner_axis * q).boost(boost);
        let third = FourMomentum::on_shell(m3, inner_axis * -q).boost(boost);

        Some(vec![first, second.momentum, third.momentum])
    }

    /// Anihilacja par lepton-antylepton na dwa fotony.
    ///
    /// Prawdopodobieństwo bierze się z przekroju czynnego Diraca, a nie z kryterium
    /// „są bliżej niż X". Różnica jest istotna: przekrój zależy od energii jak `1/β`,
    /// więc wolna para anihiluje o rzędy wielkości chętniej niż szybka, a kryterium
    /// geometryczne tego nie odda.
    ///
    /// `pair_radius` wyznacza tylko objętość `V`, w której para uchodzi za
    /// sąsiadującą; `P = σ·v·dt/V`. Wynik nie powinien od niej zależeć — i to jest
    /// sprawdzalne przez jej zmianę.
    fn annihilate(
        &mut self,
        state: &State,
        dt: f64,
        removed: &mut [bool],
        born: &mut Vec<Birth>,
        budget: &mut usize,
        capped: &mut bool,
    ) -> usize {
        let n = state.n();
        let volume = (4.0 / 3.0) * std::f64::consts::PI * self.cfg.pair_radius.powi(3);
        if volume <= 0.0 {
            return 0;
        }
        let radius2 = self.cfg.pair_radius * self.cfg.pair_radius;
        let mut count = 0;

        for i in 0..n {
            if removed[i] || !annihilates(state.kinds[i]) || state.kinds[i].anti {
                continue;
            }
            for j in 0..n {
                if removed[j] || state.kinds[j] != state.kinds[i].conjugate() {
                    continue;
                }
                if (state.positions[i] - state.positions[j]).norm_squared() > radius2 {
                    continue;
                }
                if *budget < 1 {
                    *capped = true;
                    break;
                }
                let pair = state.four_momentum(i) + state.four_momentum(j);
                let mass = state.kinds[i].mass();
                let sigma = annihilation_cross_section(mass, pair.invariant_mass());
                let speed = moller_velocity(state.velocity(i), state.velocity(j));
                let probability = (sigma * speed * dt / volume).min(1.0);
                if self.rng.unit() >= probability {
                    continue;
                }

                // Dwa fotony w układzie środka masy pary, pchnięte do laboratorium.
                let half = pair.invariant_mass() / 2.0;
                let axis = self.rng.unit_vector();
                let boost = pair.boost_velocity();
                let position = (state.positions[i] + state.positions[j]) * 0.5;
                for sign in [1.0, -1.0] {
                    let photon = FourMomentum::on_shell(0.0, axis * (half * sign)).boost(boost);
                    born.push(Birth {
                        particle: Particle::of(Species::Photon),
                        position,
                        momentum: photon.momentum,
                    });
                }
                removed[i] = true;
                removed[j] = true;
                *budget -= 1;
                count += 1;
                break;
            }
        }
        count
    }
}

/// Które cząstki umieją anihilować w tym modelu: naładowane leptony.
///
/// Kwarki i hadrony też anihilują, ale ich przekrój czynny nie ma postaci
/// zamkniętej i zależy od struktury, której ten model nie zna. Wpisanie tam
/// czegokolwiek byłoby zmyślaniem; brak jest ograniczeniem wypowiedzianym wprost.
fn annihilates(particle: Particle) -> bool {
    matches!(
        particle.species,
        Species::Electron | Species::Muon | Species::Tau
    )
}

/// Przekrój czynny Diraca na `l⁺l⁻ → γγ`, w fm².
///
/// ```text
/// σ = πr²/(γ+1) · [ (γ²+4γ+1)/(γ²−1) · ln(γ+√(γ²−1)) − (γ+3)/√(γ²−1) ]
/// ```
///
/// `γ` jest czynnikiem Lorentza jednej cząstki w układzie spoczynkowym drugiej,
/// a `r = αħc/m` jej promieniem klasycznym.
///
/// W granicy `γ → 1` licznik i mianownik dążą do zera jednocześnie, więc poniżej
/// progu, przy którym `f64` przestaje odróżniać te dwa zera, wracamy do granicy
/// analitycznej `σ → πr²/β`. Bez tego wynik dla wolnej pary — czyli tej, która
/// anihiluje najchętniej — byłby szumem.
pub fn annihilation_cross_section(mass: f64, invariant_mass: f64) -> f64 {
    if mass <= 0.0 || invariant_mass < 2.0 * mass {
        return 0.0;
    }
    let radius = units::COULOMB / mass;
    let area = std::f64::consts::PI * radius * radius;

    // s = (2m²)(1+γ) dla pary, więc γ = s/(2m²) − 1.
    let s = invariant_mass * invariant_mass;
    let gamma = s / (2.0 * mass * mass) - 1.0;
    if gamma <= 1.0 {
        return 0.0;
    }

    let excess = gamma - 1.0;
    if excess < 1e-6 {
        // β pary w układzie spoczynkowym jednej z cząstek.
        let beta = (2.0 * excess).sqrt();
        return area / beta;
    }

    let root = (gamma * gamma - 1.0).sqrt();
    let bracket =
        (gamma * gamma + 4.0 * gamma + 1.0) / (gamma * gamma - 1.0) * (gamma + root).ln()
            - (gamma + 3.0) / root;
    (area / (gamma + 1.0) * bracket).max(0.0)
}

/// Prędkość Møllera — czynnik strumienia dla pary.
///
/// ```text
/// v = √(|v₁ − v₂|² − |v₁ × v₂|²)
/// ```
///
/// **To nie jest prędkość niczego.** Dla pary lecącej naprzeciw siebie dąży do `2c`
/// i to jest poprawne: wielkość ta mierzy, jak często cząstki się spotykają
/// w danym układzie odniesienia, a nie jak szybko coś się porusza. Nic tu nie
/// przekracza `c`, bo nic tu nie jest ruchem.
///
/// Powodem, dla którego nie wystarczy zwykłe `|v₁ − v₂|`, jest złożenie prędkości:
/// dla cząstek lecących prostopadle z `β = 0,9` różnica daje `1,27`, a poprawny
/// czynnik `0,98`. Człon z iloczynem wektorowym jest dokładnie tą poprawką i to on
/// sprawia, że `σ·v·n₁n₂` jest niezmiennikiem, a nie liczbą zależną od układu.
pub fn moller_velocity(v1: Vec3, v2: Vec3) -> f64 {
    let difference = (v1 - v2).norm_squared();
    let cross = v1.cross(v2).norm_squared();
    (difference - cross).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sm::config::DecayConfig;
    use crate::sm::particles::Flavour;
    use crate::vec3::{vec3, ZERO};

    fn all_decaying() -> Vec<Particle> {
        let mut out = Vec::new();
        for species in Species::ALL {
            if channels(species).is_empty() {
                continue;
            }
            out.push(Particle::of(species));
            out.push(Particle::anti_of(species));
        }
        out
    }

    /// Sedno całej tablicy. Każdy kanał, dla cząstki i dla antycząstki, musi
    /// zachowywać ładunek, liczbę barionową i **każdą z trzech** liczb leptonowych.
    /// Wszystkie są całkowite, więc porównanie jest dokładne — kanał z pomyloną
    /// cząstką nie ma jak się prześlizgnąć.
    #[test]
    fn every_channel_conserves_every_quantum_number() {
        for parent in all_decaying() {
            for channel in channels(parent.species) {
                let products = products_for(parent, channel);
                let charge: i32 = products.iter().map(|p| p.charge_thirds()).sum();
                let baryon: i32 = products.iter().map(|p| p.baryon_thirds()).sum();
                assert_eq!(
                    charge,
                    parent.charge_thirds(),
                    "{parent}: ładunek w kanale {:?}",
                    products.iter().map(|p| p.symbol()).collect::<Vec<_>>()
                );
                assert_eq!(baryon, parent.baryon_thirds(), "{parent}: liczba barionowa");
                for flavour in [Flavour::Electron, Flavour::Muon, Flavour::Tau] {
                    let lepton: i32 = products.iter().map(|p| p.lepton_of(flavour)).sum();
                    assert_eq!(
                        lepton,
                        parent.lepton_of(flavour),
                        "{parent}: liczba leptonowa {flavour:?}"
                    );
                }
            }
        }
    }

    /// Współczynniki rozgałęzienia muszą sumować się do jedności. Suma 0,7
    /// oznaczałaby, że 30% rozpadów jest wymyślone przez normalizację, i nikt by
    /// tego nie zobaczył na wykresie.
    #[test]
    fn branching_ratios_sum_to_one() {
        for parent in all_decaying() {
            let total: f64 = channels(parent.species).iter().map(|c| c.branching).sum();
            assert!(
                (total - 1.0).abs() < 1e-3,
                "{parent}: suma współczynników {total}"
            );
        }
    }

    /// Każda cząstka z czasem życia musi mieć kanały, i odwrotnie. Rozejście się
    /// tych dwóch tablic dałoby cząstkę, która „rozpada się" donikąd.
    #[test]
    fn lifetimes_and_channels_agree() {
        for species in Species::ALL {
            let has_channels = !channels(species).is_empty();
            let has_lifetime = species.lifetime_fm().is_some();
            assert_eq!(has_channels, has_lifetime, "{species:?}");
        }
    }

    /// Kanały wirtualne muszą zostać odsiane, a reszta zostać.
    #[test]
    fn kinematically_forbidden_channels_are_filtered_out() {
        let higgs = Particle::of(Species::Higgs);
        let allowed: Vec<bool> = HIGGS.iter().map(|c| channel_allowed(higgs, c)).collect();
        assert!(!allowed[1], "H → WW nie ma prawa być dozwolone");
        assert!(!allowed[2], "H → ZZ nie ma prawa być dozwolone");
        assert!(allowed[0] && allowed[3] && allowed[4] && allowed[5] && allowed[6]);

        // Losowanie nie ma prawa wybrać kanału zabronionego — nigdy.
        let mut decays = Decays::new(DecayConfig::default());
        for _ in 0..2_000 {
            let channel = decays.choose_channel(higgs).expect("Higgs ma dokąd się rozpaść");
            assert!(channel_allowed(higgs, channel));
        }
    }

    /// Rozpad musi zachować czteropęd co do zaokrągleń — dla matki spoczywającej
    /// i dla matki w locie. To sprawdza jednocześnie przestrzeń fazową i pchnięcie.
    #[test]
    fn decays_conserve_energy_and_momentum() {
        let mut decays = Decays::new(DecayConfig::default());
        for parent in all_decaying() {
            for momentum in [ZERO, vec3(500.0, -200.0, 50.0), vec3(0.0, 0.0, 1e5)] {
                let before = FourMomentum::on_shell(parent.mass(), momentum);
                for channel in channels(parent.species) {
                    if !channel_allowed(parent, channel) {
                        continue;
                    }
                    let daughters = products_for(parent, channel);
                    let Some(momenta) = decays.decay_momenta(before, &daughters) else {
                        continue;
                    };
                    let after: FourMomentum = daughters
                        .iter()
                        .zip(momenta.iter())
                        .map(|(d, p)| FourMomentum::on_shell(d.mass(), *p))
                        .sum();
                    let scale = before.energy;
                    assert!(
                        (after.energy - before.energy).abs() / scale < 1e-9,
                        "{parent}: energia {} → {}",
                        before.energy,
                        after.energy
                    );
                    assert!(
                        (after.momentum - before.momentum).norm() / scale < 1e-9,
                        "{parent}: pęd {:?} → {:?}",
                        before.momentum,
                        after.momentum
                    );
                }
            }
        }
    }

    /// Masa niezmiennicza produktów musi odtworzyć masę matki — to jest ten sam
    /// bilans, ale wyrażony wielkością niezależną od układu odniesienia.
    #[test]
    fn products_reconstruct_the_parent_mass() {
        let mut decays = Decays::new(DecayConfig::default());
        let parent = Particle::of(Species::Tau);
        let before = FourMomentum::on_shell(parent.mass(), vec3(3_000.0, 0.0, 0.0));
        for channel in channels(parent.species) {
            let daughters = products_for(parent, channel);
            let momenta = decays.decay_momenta(before, &daughters).unwrap();
            let after: FourMomentum = daughters
                .iter()
                .zip(momenta.iter())
                .map(|(d, p)| FourMomentum::on_shell(d.mass(), *p))
                .sum();
            let mass = after.invariant_mass();
            assert!(
                (mass - parent.mass()).abs() / parent.mass() < 1e-6,
                "masa odtworzona {mass} zamiast {}",
                parent.mass()
            );
        }
    }

    /// Rozpad dwuciałowy pionu obojętnego: dwa fotony po `m/2`, dokładnie
    /// przeciwne w układzie spoczynkowym.
    #[test]
    fn a_pion_at_rest_makes_two_back_to_back_photons() {
        let mut decays = Decays::new(DecayConfig::default());
        let pion = Particle::of(Species::PionNeutral);
        let momenta = decays
            .decay_momenta(FourMomentum::at_rest(pion.mass()), &[
                Particle::of(Species::Photon),
                Particle::of(Species::Photon),
            ])
            .unwrap();
        assert!((momenta[0] + momenta[1]).norm() < 1e-12, "fotony nie są przeciwne");
        let e1 = momenta[0].norm();
        let e2 = momenta[1].norm();
        assert!((e1 + e2 - pion.mass()).abs() < 1e-12, "E₁+E₂ = {} ≠ m = {}", e1 + e2, pion.mass());
        for photon in momenta {
            assert!((photon.norm() - pion.mass() / 2.0).abs() < 1e-12);
        }
    }

    /// Czas życia zmierzony na próbce musi zgadzać się z tablicowym. To jest test
    /// całego łańcucha: prawdopodobieństwa, losowania i pętli po krokach.
    #[test]
    fn the_measured_lifetime_matches_the_table() {
        let muon = Particle::of(Species::Muon);
        let lifetime = muon.lifetime_fm().unwrap();
        let mut decays = Decays::new(DecayConfig {
            annihilation: false,
            ..DecayConfig::default()
        });
        let n = 4_000;
        let mut state = State::new(
            vec![ZERO; n],
            vec![ZERO; n],
            vec![muon; n],
        )
        .unwrap();

        // Krok równy dziesiątej części czasu życia; po jednym czasie życia ma
        // zostać 1/e próbki.
        let dt = lifetime / 10.0;
        for _ in 0..10 {
            decays.step(&mut state, dt, 10 * n);
        }
        let survivors = state.count_of(muon) as f64 / n as f64;
        let expected = std::f64::consts::E.recip();
        assert!(
            (survivors - expected).abs() < 0.03,
            "przeżyło {survivors}, a ma przeżyć {expected}"
        );
    }

    /// Dylatacja czasu: szybki mion żyje `γ` razy dłużej. To jest jedyny test,
    /// w którym szczególna teoria względności ma obserwowalny skutek dla rozpadu —
    /// i nie jest tu dołożona ręcznie, tylko wynika z pędu jako zmiennej stanu.
    #[test]
    fn a_fast_muon_lives_longer_by_exactly_gamma() {
        let muon = Particle::of(Species::Muon);
        let lifetime = muon.lifetime_fm().unwrap();
        let gamma = 10.0;
        let momentum = vec3(0.0, 0.0, muon.mass() * (gamma * gamma - 1.0f64).sqrt());

        let survival = |momenta: Vec<Vec3>, dt: f64| -> f64 {
            let n = momenta.len();
            let mut decays = Decays::new(DecayConfig {
                annihilation: false,
                ..DecayConfig::default()
            });
            let mut state = State::new(vec![ZERO; n], momenta, vec![muon; n]).unwrap();
            for _ in 0..10 {
                decays.step(&mut state, dt / 10.0, 10 * n);
            }
            state.count_of(muon) as f64 / n as f64
        };

        let n = 4_000;
        let slow = survival(vec![ZERO; n], lifetime);
        // τ_lab = γτ: przez γ razy dłuższy czas laboratoryjny szybki mion
        // przeżywa tyle samo, co spoczywający przez τ — ułamek 1/e.
        let fast = survival(vec![momentum; n], lifetime * gamma);
        let expected = std::f64::consts::E.recip();
        assert!(
            (slow - expected).abs() < 0.04,
            "spoczywający po τ: {slow}, oczekiwane {expected}"
        );
        assert!(
            (fast - expected).abs() < 0.04,
            "lecący z γ=10 po τ_lab=γτ: {fast}, oczekiwane {expected}"
        );
        assert!(
            (slow - fast).abs() < 0.04,
            "spoczywający: {slow}, lecący z γ=10: {fast}"
        );
        // A przez ten sam czas laboratoryjny — znacznie więcej.
        let fast_short = survival(vec![momentum; n], lifetime);
        assert!(fast_short > slow + 0.25, "brak dylatacji: {fast_short} vs {slow}");
    }

    /// Rozpad w silniku musi zachować liczby kwantowe DOKŁADNIE, przez cały bieg
    /// z kaskadą (taon → mion → elektron, z pionami po drodze).
    #[test]
    fn a_full_cascade_conserves_the_integer_charges() {
        let tau = Particle::of(Species::Tau);
        let n = 200;
        let mut state = State::new(vec![ZERO; n], vec![ZERO; n], vec![tau; n]).unwrap();
        let charge = state.total_charge_thirds();
        let baryon = state.total_baryon_thirds();
        let lepton = state.total_lepton();

        let mut decays = Decays::new(DecayConfig {
            annihilation: false,
            ..DecayConfig::default()
        });
        let dt = tau.lifetime_fm().unwrap();
        for _ in 0..40 {
            decays.step(&mut state, dt, 100_000);
        }

        assert!(state.count_of(tau) < n / 10, "taony się nie rozpadły");
        assert_eq!(state.total_charge_thirds(), charge, "ładunek");
        assert_eq!(state.total_baryon_thirds(), baryon, "liczba barionowa");
        assert_eq!(state.total_lepton(), lepton, "liczba leptonowa");
    }

    /// Kaskada musi też zachować energię — sumaryczną, przez wszystkie pokolenia.
    #[test]
    fn a_full_cascade_conserves_energy() {
        let n = 300;
        let pion = Particle::of(Species::PionCharged);
        let mut state = State::new(vec![ZERO; n], vec![ZERO; n], vec![pion; n]).unwrap();
        let before = state.total_four_momentum();

        let mut decays = Decays::new(DecayConfig {
            annihilation: false,
            ..DecayConfig::default()
        });
        let dt = pion.lifetime_fm().unwrap() * 3.0;
        for _ in 0..30 {
            decays.step(&mut state, dt, 100_000);
        }

        let after = state.total_four_momentum();
        assert!(
            (after.energy - before.energy).abs() / before.energy < 1e-9,
            "energia {} → {}",
            before.energy,
            after.energy
        );
        assert!(after.momentum.norm() / before.energy < 1e-9, "pęd się urodził");
    }

    /// Limit liczby cząstek ma POWSTRZYMAĆ rozpad, a nie zjeść cząstkę. Zniknięcie
    /// matki bez produktów łamałoby bilans energii i wyglądałoby jak usterka fizyki.
    #[test]
    fn the_particle_cap_stops_decays_instead_of_eating_particles() {
        let n = 50;
        let pion = Particle::of(Species::PionNeutral);
        let mut state = State::new(vec![ZERO; n], vec![ZERO; n], vec![pion; n]).unwrap();
        let before = state.total_four_momentum();

        let mut decays = Decays::new(DecayConfig {
            annihilation: false,
            ..DecayConfig::default()
        });
        // Miejsca starczy na kilka rozpadów, nie na wszystkie.
        let outcome = decays.step(&mut state, pion.lifetime_fm().unwrap() * 10.0, n + 10);

        assert!(outcome.capped, "limit nie zadziałał");
        assert!(state.n() <= n + 10);
        let after = state.total_four_momentum();
        assert!(
            (after.energy - before.energy).abs() / before.energy < 1e-9,
            "limit zjadł energię: {} → {}",
            before.energy,
            after.energy
        );
    }

    /// Granica nierelatywistyczna przekroju czynnego: `σβ → πr_e²`. Liczba znana
    /// z podręcznika, więc sprawdza wzór niezależnie od tego kodu.
    #[test]
    fn the_annihilation_cross_section_matches_the_low_energy_limit() {
        let m = Species::Electron.mass();
        let expected = std::f64::consts::PI * units::CLASSICAL_ELECTRON_RADIUS.powi(2);
        for gamma in [1.0 + 1e-9, 1.0 + 1e-7, 1.0 + 1e-5, 1.0 + 1e-3] {
            let s = 2.0 * m * m * (1.0 + gamma);
            let sigma = annihilation_cross_section(m, s.sqrt());
            let beta = (2.0 * (gamma - 1.0)).sqrt();
            assert!(
                (sigma * beta / expected - 1.0).abs() < 0.02,
                "γ−1={:e}: σβ/πr² = {}",
                gamma - 1.0,
                sigma * beta / expected
            );
        }
    }

    /// Przekrój maleje z energią — szybka para anihiluje niechętnie. To jest powód,
    /// dla którego kryterium geometryczne („bliżej niż X") byłoby złym modelem.
    #[test]
    fn the_cross_section_falls_with_energy() {
        let m = Species::Electron.mass();
        let at = |gamma: f64| annihilation_cross_section(m, (2.0 * m * m * (1.0 + gamma)).sqrt());
        let slow = at(1.01);
        let fast = at(100.0);
        assert!(slow > 100.0 * fast, "σ(1,01) = {slow}, σ(100) = {fast}");
        // Poniżej progu przekrój jest zerem, a nie liczbą ujemną.
        assert_eq!(annihilation_cross_section(m, m), 0.0);
        assert_eq!(annihilation_cross_section(0.0, 10.0), 0.0);
    }

    /// Czynnik strumienia: dla pary czołowej sięga `2c` (i tak ma być, bo nie jest
    /// prędkością), dla pary prostopadłej jest MNIEJSZY niż zwykła różnica prędkości.
    /// Ta druga własność odróżnia poprawny wzór od naiwnego `|v₁ − v₂|`.
    #[test]
    fn moller_velocity_is_a_flux_factor_not_a_speed() {
        let head_on = moller_velocity(vec3(0.9, 0.0, 0.0), vec3(-0.9, 0.0, 0.0));
        assert!((head_on - 1.8).abs() < 1e-12, "czołowo: {head_on}");

        let perpendicular = moller_velocity(vec3(0.9, 0.0, 0.0), vec3(0.0, 0.9, 0.0));
        let naive = (vec3(0.9, 0.0, 0.0) - vec3(0.0, 0.9, 0.0)).norm();
        assert!(
            perpendicular < naive - 0.2,
            "prostopadle: {perpendicular} wobec naiwnego {naive}"
        );
        assert!((perpendicular - 0.9819).abs() < 1e-3, "{perpendicular}");

        // Cząstki lecące tak samo nigdy się nie spotkają.
        let together = moller_velocity(vec3(0.5, 0.0, 0.0), vec3(0.5, 0.0, 0.0));
        assert!(together < 1e-12, "v = {together}");
    }

    /// Anihilacja zamienia parę na dwa fotony i nie gubi ani energii, ani pędu.
    #[test]
    fn annihilation_turns_a_pair_into_two_photons() {
        let mut decays = Decays::new(DecayConfig {
            enabled: true,
            annihilation: true,
            pair_radius: 1.0,
            seed: 11,
        });
        let electron = Particle::of(Species::Electron);
        let mut state = State::new(
            vec![ZERO, vec3(0.1, 0.0, 0.0)],
            vec![vec3(0.0, 0.001, 0.0), vec3(0.0, -0.001, 0.0)],
            vec![electron, electron.conjugate()],
        )
        .unwrap();
        let before = state.total_four_momentum();

        // Wolna para w małej objętości: przekrój jest ogromny, więc jeden duży krok
        // wystarczy, żeby prawdopodobieństwo nasyciło się do jedności.
        let outcome = decays.step(&mut state, 1.0e6, 100);

        assert_eq!(outcome.annihilated, 1);
        assert_eq!(state.n(), 2);
        assert_eq!(state.count_of(Particle::of(Species::Photon)), 2);
        assert_eq!(state.total_charge_thirds(), 0);
        assert_eq!(state.total_lepton(), 0);

        let after = state.total_four_momentum();
        assert!((after.energy - before.energy).abs() / before.energy < 1e-12);
        assert!((after.momentum - before.momentum).norm() / before.energy < 1e-12);
    }

    /// Wyłączona warstwa nie ma prawa niczego ruszyć.
    #[test]
    fn a_disabled_layer_changes_nothing() {
        let mut decays = Decays::new(DecayConfig {
            enabled: false,
            ..DecayConfig::default()
        });
        let muon = Particle::of(Species::Muon);
        let mut state = State::new(vec![ZERO; 10], vec![ZERO; 10], vec![muon; 10]).unwrap();
        let outcome = decays.step(&mut state, 1e30, 1_000);
        assert_eq!(outcome, Outcome::default());
        assert_eq!(state.count_of(muon), 10);
    }

    /// Ten sam zarodek musi dać ten sam bieg. Bez tego porównywanie dwóch
    /// przebiegów przestaje cokolwiek znaczyć.
    #[test]
    fn the_same_seed_gives_the_same_run() {
        let run = || {
            let mut decays = Decays::new(DecayConfig::default());
            let n = 100;
            let muon = Particle::of(Species::Muon);
            let mut state = State::new(vec![ZERO; n], vec![ZERO; n], vec![muon; n]).unwrap();
            for _ in 0..5 {
                decays.step(&mut state, muon.lifetime_fm().unwrap() / 4.0, 10_000);
            }
            (state.n(), state.census())
        };
        assert_eq!(run().0, run().0);
        assert_eq!(run().1, run().1);
    }

    /// Cząstki trwałe nie rozpadają się nawet po nieprzyzwoicie długim czasie.
    #[test]
    fn stable_particles_never_decay() {
        let mut decays = Decays::new(DecayConfig::default());
        let stable = [Species::Electron, Species::Proton, Species::Photon];
        let kinds: Vec<Particle> = stable.iter().map(|s| Particle::of(*s)).collect();
        let n = kinds.len();
        let mut state = State::new(vec![ZERO; n], vec![ZERO; n], kinds).unwrap();
        for _ in 0..20 {
            decays.step(&mut state, 1e40, 1_000);
        }
        assert_eq!(state.n(), n);
    }
}
