//! Nastawy modelu cząstek: skład początkowy, oddziaływania, rozpady, bieg.

use serde::{Deserialize, Serialize};

use crate::sm::particles::{Particle, Species};
use crate::sm::units;
use crate::sr::config::BackendKind;

/// Kształt rozkładu początkowego.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Shape {
    /// Kula wypełniona jednorodnie, pędy izotropowe — gaz w pudle bez ścian.
    Ball,
    /// Wiązka: cząstki lecą wzdłuż osi `z` z zadaną energią kinetyczną.
    Beam,
    /// Dwie cząstki naprzeciw siebie — najprostszy układ, jaki cokolwiek pokazuje.
    Pair,
}

impl Shape {
    pub const ALL: [Shape; 3] = [Self::Ball, Self::Beam, Self::Pair];

    pub fn label(self) -> &'static str {
        match self {
            Self::Ball => "kula",
            Self::Beam => "wiązka",
            Self::Pair => "para",
        }
    }
}

/// Jeden składnik mieszanki początkowej.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ingredient {
    pub particle: Particle,
    pub count: usize,
}

impl Ingredient {
    pub fn new(particle: Particle, count: usize) -> Self {
        Self { particle, count }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpawnConfig {
    pub shape: Shape,
    /// Skład: ile cząstek którego rodzaju.
    pub mixture: Vec<Ingredient>,
    /// Promień chmury (fm) — dla wiązki promień przekroju poprzecznego.
    pub radius: f64,
    /// Skala energii kinetycznej losowanej izotropowo (MeV na cząstkę).
    pub temperature: f64,
    /// Energia kinetyczna wiązki (MeV na cząstkę), używana przez [`Shape::Beam`]
    /// i [`Shape::Pair`].
    pub beam_energy: f64,
    pub seed: u64,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            shape: Shape::Ball,
            mixture: vec![
                Ingredient::new(Particle::of(Species::Electron), 400),
                Ingredient::new(Particle::of(Species::Proton), 400),
            ],
            radius: 5.0e4,
            temperature: 1.0e-3,
            beam_energy: 0.0,
            seed: 20_240_901,
        }
    }
}

impl SpawnConfig {
    pub fn total_count(&self) -> usize {
        self.mixture.iter().map(|i| i.count).sum()
    }

    /// Wypadkowy ładunek mieszanki w trzecich `e`. Zero znaczy „quasi-neutralna",
    /// czyli taka, w której ekranowanie Debye'a ma sens.
    pub fn net_charge_thirds(&self) -> i64 {
        self.mixture
            .iter()
            .map(|i| i.particle.charge_thirds() as i64 * i.count as i64)
            .sum()
    }

    /// Przeskaluj mieszankę do zadanej liczby cząstek, zachowując proporcje.
    ///
    /// Reszta z dzielenia całkowitego idzie do ostatniego składnika, żeby suma
    /// była dokładnie `n`. Przy dziwnych `n` ładunek wypadkowy może o jeden
    /// odjechać — i [`Config::warnings`] to zapowie.
    pub fn scale_to(&mut self, n: usize) {
        let n = n.max(1);
        let total = self.total_count();
        if total == 0 || self.mixture.is_empty() {
            return;
        }
        let last = self.mixture.len() - 1;
        let mut remaining = n;
        for (i, ingredient) in self.mixture.iter_mut().enumerate() {
            if i == last {
                ingredient.count = remaining;
                break;
            }
            let count = (ingredient.count as u128 * n as u128 / total as u128) as usize;
            ingredient.count = count;
            remaining = remaining.saturating_sub(count);
        }
    }
}

/// Które oddziaływania są włączone i jak liczone.
///
/// Coulomb i Cornell zostają w kodzie, bo laboratoria ich potrzebują.
/// Grawitacja nie jest tu flagą: jej udział to odczyt `F_g/F_EM` z mas i
/// ładunków. Słabe nie jest tu flagą: to rozpady, nie potencjał.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ForceConfig {
    pub coulomb: bool,
    pub strong: bool,
    /// Zmiękczenie Plummera `ε` w fm — wspólne dla oddziaływań dalekozasięgowych.
    ///
    /// **To jest proteza za mechanikę kwantową, nie parametr fizyczny.** Dwa
    /// klasyczne ładunki przeciwnego znaku nie mają stanu podstawowego: spadają na
    /// siebie, uwalniając nieskończoną energię. `ε` jest odległością, poniżej której
    /// przestajemy udawać, że opis klasyczny obowiązuje. Diagnostyka podaje, jaka
    /// energia z tego wynika, żeby dało się ocenić, czy wynik od niej zależy.
    pub softening: f64,
    pub backend: BackendKind,
    pub grid: usize,
    pub box_margin: f64,
}

impl Default for ForceConfig {
    fn default() -> Self {
        Self {
            coulomb: true,
            strong: false,
            softening: 1.0e-2,
            backend: BackendKind::Auto,
            grid: 48,
            box_margin: 0.15,
        }
    }
}

impl ForceConfig {
    pub fn any_long_range(&self) -> bool {
        self.coulomb
    }

    pub fn any_short_range(&self) -> bool {
        self.strong
    }

    pub fn none_enabled(&self) -> bool {
        !self.any_long_range() && !self.any_short_range()
    }
}

/// Nastawy warstwy stochastycznej.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DecayConfig {
    pub enabled: bool,
    pub annihilation: bool,
    /// Promień, w którym para cząstka-antycząstka jest w ogóle rozważana (fm).
    ///
    /// Nie jest to „promień zderzenia": prawdopodobieństwo anihilacji liczone jest
    /// z przekroju czynnego Diraca, a ten promień wyznacza tylko objętość, w której
    /// para jest uznana za sąsiadującą. Wynik nie powinien od niego zależeć,
    /// dopóki jest znacznie większy od promienia oddziaływania — i to jest
    /// sprawdzalne przez zmianę tej liczby.
    pub pair_radius: f64,
    pub seed: u64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            annihilation: true,
            pair_radius: 50.0,
            seed: 7_707_707,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    /// Krok całkowania w fm/c.
    pub dt: f64,
    /// Czy przycinać krok, gdy siły są duże. Przy biegu nastawionym na rozpady krok
    /// jest o rzędy wielkości większy od skali sił i przycinanie zatrzymałoby bieg —
    /// dlatego jest to wybór, a nie zachowanie domyślne.
    pub adaptive: bool,
    pub steps: u64,
    pub diagnostics_every: u32,
    pub trajectory_every: u32,
    pub point_stride: usize,
    /// Twardy limit liczby cząstek. Rozpad `H → WW → ...` mnoży cząstki, więc bez
    /// limitu bieg potrafi wyczerpać pamięć zamiast się skończyć.
    pub max_particles: usize,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            dt: 1.0e-2,
            adaptive: true,
            steps: 2_000,
            diagnostics_every: 20,
            trajectory_every: 10,
            point_stride: 1,
            max_particles: 200_000,
        }
    }
}

/// `#[serde(default)]` na każdym poziomie: plik konfiguracji ma dać się skrócić do
/// tych pól, które faktycznie się zmienia. Bez tego dopisanie jednej nastawy do
/// [`RunConfig`] unieważniałoby wszystkie zapisane wcześniej `config.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub spawn: SpawnConfig,
    pub forces: ForceConfig,
    pub decay: DecayConfig,
    pub run: RunConfig,
}

impl Config {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("konfiguracja jest serializowalna")
    }

    /// # Errors
    /// Gdy tekst nie jest poprawnym JSON-em albo brakuje w nim pól.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    /// Zastrzeżenia, o których użytkownik ma usłyszeć **przed** biegiem, a nie po.
    ///
    /// Żadne z nich nie jest błędem — każde jest sytuacją, w której wynik będzie
    /// poprawnie policzony, ale nie będzie pokazywał tego, czego się spodziewano.
    pub fn warnings(&self) -> Vec<String> {
        let mut out = Vec::new();

        if self.spawn.total_count() == 0 {
            out.push("mieszanka jest pusta — nie ma czego liczyć".to_string());
        }
        if self.forces.none_enabled() {
            out.push(
                "wszystkie oddziaływania wyłączone — cząstki lecą po prostych".to_string(),
            );
        }

        let net = self.spawn.net_charge_thirds();
        if self.forces.coulomb && net != 0 {
            out.push(format!(
                "mieszanka ma wypadkowy ładunek {:+.2} e — chmura rozleci się odpychaniem, \
                 zamiast się ekranować",
                net as f64 / 3.0
            ));
        }

        if self.forces.strong
            && !self
                .spawn
                .mixture
                .iter()
                .any(|i| i.particle.colored() && i.count > 0)
        {
            out.push(
                "oddziaływanie silne włączone, ale żadna cząstka nie niesie koloru".to_string(),
            );
        }

        // Sedno problemu dwóch skal czasu: krok albo rozdziela siły, albo dożywa
        // rozpadu — jedno i drugie naraz nie jest możliwe.
        if let Some(shortest) = self.shortest_lifetime_fm() {
            let horizon = self.run.dt * self.run.steps.max(1) as f64;
            if self.decay.enabled && horizon < shortest * 1e-3 {
                out.push(format!(
                    "bieg trwa {:.2e} fm/c, a najkrótszy czas życia w mieszance to {:.2e} fm/c \
                     — rozpadów praktycznie nie będzie widać",
                    horizon, shortest
                ));
            }
        }

        if self.forces.any_long_range() && self.run.dt > 1.0 && self.run.adaptive {
            out.push(format!(
                "krok {:.2e} fm/c jest znacznie większy od skali sił — adaptacja przytnie go \
                 i bieg będzie posuwał się wolniej, niż wynika z liczby kroków",
                self.run.dt
            ));
        }

        out
    }

    /// Najkrótszy czas życia w mieszance (fm/c); `None`, gdy wszystko jest trwałe.
    pub fn shortest_lifetime_fm(&self) -> Option<f64> {
        self.spawn
            .mixture
            .iter()
            .filter(|i| i.count > 0)
            .filter_map(|i| i.particle.lifetime_fm())
            .min_by(f64::total_cmp)
    }

    /// Opis skali czasu biegu dla nagłówka panelu.
    pub fn describe_time_scale(&self) -> String {
        let horizon = self.run.dt * self.run.steps.max(1) as f64;
        format!(
            "krok {:.2e} fm/c, widnokrąg {:.2e} fm/c ({:.2e} s)",
            self.run.dt,
            horizon,
            units::fm_to_seconds(horizon)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_mixture_is_neutral_and_nonempty() {
        let cfg = Config::default();
        assert!(cfg.spawn.total_count() > 0);
        assert_eq!(cfg.spawn.net_charge_thirds(), 0);
        assert!(cfg.warnings().is_empty(), "{:?}", cfg.warnings());
    }

    #[test]
    fn config_survives_a_json_round_trip() {
        let cfg = Config::default();
        let back = Config::from_json(&cfg.to_json()).unwrap();
        assert_eq!(back, cfg);
    }

    /// Cząstka w pliku ma być czytelna dla człowieka. Test pilnuje kontraktu,
    /// bo derive na `Particle` przywróciłby zapis obiektowy bez żadnego sygnału.
    #[test]
    fn particles_serialize_as_readable_identifiers() {
        let cfg = Config::default();
        let text = cfg.to_json();
        assert!(text.contains("\"e-\""), "brak czytelnego elektronu w:\n{text}");
        // Cząstka o całkowitym ładunku zawsze nosi znak, także proton: `p+`, nie `p`.
        // Zapis bez znaku jest przyjmowany przy czytaniu, ale zapisujemy kanoniczny.
        assert!(text.contains("\"p+\""), "brak czytelnego protonu w:\n{text}");
        assert!(
            !text.contains("Electron"),
            "cząstka zapisała się jako obiekt:\n{text}"
        );
    }

    #[test]
    fn unknown_particle_in_json_is_an_error_with_a_hint() {
        let text = r#"{"spawn":{"mixture":[{"particle":"kwarkoid","count":1}]}}"#;
        let err = Config::from_json(text).unwrap_err().to_string();
        assert!(err.contains("kwarkoid"), "{err}");
        assert!(err.contains("e-"), "brak podpowiedzi w komunikacie: {err}");
    }

    /// Skrócony plik ma się wczytać, a brakujące pola przyjąć wartości domyślne.
    /// Bez tego każde dopisanie nastawy unieważniałoby zapisane biegi.
    #[test]
    fn a_partial_file_fills_in_the_defaults() {
        let cfg = Config::from_json(r#"{"run":{"dt":0.5}}"#).unwrap();
        assert_eq!(cfg.run.dt, 0.5);
        assert_eq!(cfg.run.steps, RunConfig::default().steps);
        assert_eq!(cfg.spawn, SpawnConfig::default());
    }

    /// Mieszanka z niezerowym ładunkiem jest legalna, ale musi być zapowiedziana:
    /// chmura samych elektronów rozleci się odpychaniem i nie pokaże ekranowania,
    /// po które zwykle się ją uruchamia.
    #[test]
    fn a_charged_mixture_is_announced() {
        let mut cfg = Config::default();
        cfg.spawn.mixture = vec![Ingredient::new(Particle::of(Species::Electron), 100)];
        let warnings = cfg.warnings();
        assert!(
            warnings.iter().any(|w| w.contains("wypadkowy ładunek")),
            "{warnings:?}"
        );
    }

    #[test]
    fn strong_force_without_color_is_announced() {
        let mut cfg = Config::default();
        cfg.forces.strong = true;
        assert!(cfg.warnings().iter().any(|w| w.contains("koloru")));

        cfg.spawn.mixture = vec![
            Ingredient::new(Particle::of(Species::Up), 2),
            Ingredient::new(Particle::anti_of(Species::Up), 2),
        ];
        assert!(!cfg.warnings().iter().any(|w| w.contains("koloru")));
    }

    /// Bieg krótszy od czasu życia o trzy rzędy wielkości nie pokaże rozpadów.
    /// Milczenie w tej sytuacji dawałoby „rozpady nie działają" zamiast
    /// „patrzysz przez zbyt krótką lunetę".
    #[test]
    fn a_run_too_short_to_see_decays_is_announced() {
        let mut cfg = Config::default();
        cfg.spawn.mixture = vec![Ingredient::new(Particle::of(Species::Muon), 10)];
        cfg.forces.coulomb = false;
        cfg.run.dt = 1.0;
        cfg.run.steps = 100;
        assert!(
            cfg.warnings().iter().any(|w| w.contains("rozpadów")),
            "{:?}",
            cfg.warnings()
        );

        // Krok dobrany do czasu życia mionu ostrzeżenia już nie wywołuje.
        cfg.run.dt = 1.0e16;
        cfg.run.adaptive = false;
        assert!(!cfg.warnings().iter().any(|w| w.contains("rozpadów")));
    }

    #[test]
    fn shortest_lifetime_ignores_stable_and_empty_entries() {
        let mut cfg = Config::default();
        assert_eq!(cfg.shortest_lifetime_fm(), None, "elektron i proton są trwałe");

        cfg.spawn.mixture.push(Ingredient::new(
            Particle::of(Species::PionNeutral),
            0,
        ));
        assert_eq!(cfg.shortest_lifetime_fm(), None, "pusty składnik się nie liczy");

        cfg.spawn
            .mixture
            .push(Ingredient::new(Particle::of(Species::Muon), 5));
        cfg.spawn
            .mixture
            .push(Ingredient::new(Particle::of(Species::PionNeutral), 5));
        let shortest = cfg.shortest_lifetime_fm().unwrap();
        assert_eq!(shortest, Species::PionNeutral.lifetime_fm().unwrap());
    }

    #[test]
    fn scaling_keeps_the_ratio_and_the_total() {
        let mut spawn = SpawnConfig::default();
        spawn.scale_to(80);
        assert_eq!(spawn.total_count(), 80);
        assert_eq!(spawn.mixture[0].count, 40);
        assert_eq!(spawn.mixture[1].count, 40);
        assert_eq!(spawn.net_charge_thirds(), 0);
    }

    #[test]
    fn disabled_forces_are_announced() {
        let mut cfg = Config::default();
        cfg.forces = ForceConfig {
            coulomb: false,
            strong: false,
            ..cfg.forces
        };
        assert!(cfg.forces.none_enabled());
        assert!(cfg.warnings().iter().any(|w| w.contains("po prostych")));
    }
}
