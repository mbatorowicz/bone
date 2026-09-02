//! Konfiguracja biegu SR: kształt, materia, prawa fizyki, solver, zapis.
//!
//! Jedno źródło prawdy. Podział na cztery sekcje idzie po PYTANIU, na które
//! odpowiadają: co budujemy, jakie prawa nim rządzą, czym to liczymy, co z tego
//! zapisujemy.
//!
//! # Runtime kontra startup
//!
//! Część parametrów wolno ruszać w trakcie biegu (G, ε, tempo chłodzenia), część
//! nie (liczba cząstek, geometria, ziarno). Granica nie jest zapisana jako lista
//! nazw pól — jest zapisana w [`Config::with_runtime_from`], która składa nową
//! konfigurację z runtime'owych pól jednej i startowych drugiej. Lista nazw
//! rozjechałaby się z polami przy pierwszej zmianie; funkcja się nie skompiluje.

use serde::{Deserialize, Serialize};

/// Kształt chmury początkowej.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Geometry {
    Ball,
    Cube,
    Cylinder,
    Disk,
    Torus,
    SphereShell,
    Filament,
    Gaussian,
    TwoClumps,
    Plummer,
}

impl Geometry {
    /// Kolejność jak w interfejsie: bryły zwarte, kształty o wyróżnionej osi,
    /// rozkłady nieostre, układy złożone.
    pub const ALL: [Geometry; 10] = [
        Geometry::Ball,
        Geometry::Cube,
        Geometry::Cylinder,
        Geometry::Disk,
        Geometry::Torus,
        Geometry::SphereShell,
        Geometry::Filament,
        Geometry::Gaussian,
        Geometry::TwoClumps,
        Geometry::Plummer,
    ];

    /// Etykieta wraz z tym, co dla danego kształtu znaczy „grubość" — bez tego
    /// suwak grubości jest dla większości kształtów zagadką.
    pub fn label(self) -> &'static str {
        match self {
            Self::Ball => "Kula (jednorodna)",
            Self::Cube => "Kostka",
            Self::Cylinder => "Walec",
            Self::Disk => "Dysk — grubość = σ w pionie",
            Self::Torus => "Torus — grubość = przekrój",
            Self::SphereShell => "Powłoka sferyczna",
            Self::Filament => "Włókno — grubość = σ przekroju",
            Self::Gaussian => "Chmura gaussowska",
            Self::TwoClumps => "Dwie gromady",
            Self::Plummer => "Sfera Plummera (równowaga)",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Ball => "ball",
            Self::Cube => "cube",
            Self::Cylinder => "cylinder",
            Self::Disk => "disk",
            Self::Torus => "torus",
            Self::SphereShell => "sphere_shell",
            Self::Filament => "filament",
            Self::Gaussian => "gaussian",
            Self::TwoClumps => "two_clumps",
            Self::Plummer => "plummer",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.slug() == text)
    }

    /// Czy kształt ma własny przekrój poprzeczny i czyta `thickness`.
    pub fn uses_thickness(self) -> bool {
        matches!(self, Self::Filament | Self::Torus | Self::Disk)
    }
}

/// Wybór solvera sił.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Dokładne O(N²), dopóki mieści się w budżecie czasu; wyżej siatka.
    Auto,
    Exact,
    Mesh,
}

impl BackendKind {
    pub const ALL: [BackendKind; 3] = [Self::Auto, Self::Exact, Self::Mesh];

    pub fn slug(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Exact => "exact",
            Self::Mesh => "mesh",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.slug() == text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpawnConfig {
    pub n_particles: usize,
    pub geometry: Geometry,
    pub radius: f64,
    pub seed: u64,
    /// Masa CAŁEGO układu, nie jednej cząstki. Dzięki temu suwak liczby cząstek
    /// zmienia rozdzielczość, a nie fizykę: ten sam obiekt próbkowany gęściej.
    /// Gdyby parametrem była masa cząstki, przejście z 4 na 100 tys. cząstek
    /// zwiększyłoby masę układu 25-krotnie i rozwaliło warunek startowy.
    pub total_mass: f64,
    pub mass_spread: f64,
    /// Grubość poprzeczna jako ułamek promienia; czytają ją kształty o wyróżnionym
    /// przekroju. To ona, a nie promień, wyznacza długość fali fragmentacji
    /// (λ ≈ 3,6·σ) — bez tego pokrętła te kształty były samopodobne i zawsze
    /// mieściły tyle samo długości Jeansa, niezależnie od promienia.
    pub thickness: f64,
    /// Spłaszczenie osi z, mnożnik stosowany do KAŻDEGO kształtu po wylosowaniu.
    /// Rozdziela rozmiar od proporcji, więc zmiana kształtu nie zmienia przy okazji
    /// skali i porównania między biegami pozostają uczciwe.
    pub flatten: f64,
    /// Ułamek prędkości okrężnej nadawany na starcie (0 = zimny start).
    pub rotation: f64,
    /// Izotropowy rozrzut prędkości jako ułamek c.
    pub temperature: f64,
    /// Docelowy stosunek wirialny 2K/|U|; > 0 przejmuje kontrolę nad `temperature`.
    ///
    /// Istnieje, bo `temperature` NIE JEST porównywalna między kształtami: ta sama
    /// dyspersja daje 2K/|U| = 1,34 dla kostki i 0,49 dla włókna. Bieg z ustaloną
    /// temperaturą miesza wpływ geometrii z wpływem tego, jak daleko od równowagi
    /// kształt wystartował. Przy zadanym wiriale każdy kształt startuje z równowagi
    /// liczonej dla NIEGO.
    pub virial: f64,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            n_particles: 4_000,
            geometry: Geometry::Plummer,
            radius: 8.0,
            seed: 42,
            total_mass: 4_000.0,
            mass_spread: 0.15,
            thickness: 0.08,
            flatten: 1.0,
            rotation: 0.6,
            temperature: 0.0,
            virial: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PhysicsConfig {
    /// Dobrane tak, aby przy domyślnym starcie prędkość okrężna na brzegu wynosiła
    /// ≈0,3 c — patrz [`gravity_for_beta`].
    pub g: f64,
    pub c: f64,
    /// Softening Plummera; ma wymiar DŁUGOŚCI (nie ε²).
    pub softening: f64,
    /// Górny limit kroku; faktyczny krok wybiera kryterium adaptacyjne.
    pub dt_max: f64,
    /// Dokładność kroku adaptacyjnego: dt ≤ η·√(ε/a_max).
    pub accuracy: f64,
    pub adaptive_dt: bool,

    /// Tempo tłumienia dyspersji prędkości przy ŚREDNIEJ gęstości układu, w 1/czas.
    /// Odwrotność jest czasem chłodzenia. Domyślne zero, bo dyssypacja zmienia
    /// klasę modelu i nie powinna się włączać niepostrzeżenie.
    pub cooling_rate: f64,
    /// Wykładnik zależności tempa od gęstości. 1 odpowiada emisyjności ∝ n² na
    /// jednostkę objętości — dzięki temu gęste obszary zapadają się wybiórczo,
    /// zamiast żeby cały układ opadał równomiernie.
    pub cooling_density_power: f64,
    /// Bezwzględna podłoga dyspersji jako ułamek c — odpowiednik temperatury,
    /// poniżej której ośrodek przestaje promieniować. Ustala największą skalę
    /// fragmentu; 0 wyłącza.
    pub cooling_floor: f64,
    /// Podłoga NUMERYCZNA: ile cząstek musi liczyć masa Jeansa.
    ///
    /// To ona wykonuje właściwą pracę, bo podłoga nałożona na samą dyspersję nie
    /// umie obronić rozdzielczości — λ = σ√(π/Gρ) zależy też od gęstości, więc przy
    /// zapadaniu λ schodzi pod oczko siatki przy niezmienionym σ. Kryterium na masę
    /// Jeansa zależy od gęstości jak ρ^(1/6), a kryterium na λ jak √ρ, czyli
    /// dziedziczy błąd zgrubnej siatki niemal w całości.
    pub cooling_min_particles: f64,
    /// Bok siatki, na której mierzony jest LOKALNY przepływ masowy; 0 = automat.
    /// Ustala skalę słowa „lokalny": za gęsta daje kilka cząstek na komórkę i
    /// dyspersję z szumu, za zgrubna wlicza do przepływu ruch osobnych zgęstek.
    pub cooling_grid: usize,
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            g: 0.16,
            c: 30.0,
            softening: 0.25,
            dt_max: 0.02,
            accuracy: 0.03,
            adaptive_dt: true,
            cooling_rate: 0.0,
            cooling_density_power: 1.0,
            cooling_floor: 0.0,
            cooling_min_particles: 1_000.0,
            cooling_grid: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverConfig {
    /// Wybór solvera; `auto` porównuje koszt obu — patrz
    /// [`crate::sr::backends::prefer_exact`].
    ///
    /// Nastawy „powyżej tylu cząstek idź na siatkę" tu nie ma celowo: koszt siatki
    /// zależy od `grid`, a nie od liczby cząstek, więc jedna liczba nie może być dobra
    /// dla wszystkich siatek. Kto chce rozstrzygnąć sam, wybiera solver wprost.
    pub backend: BackendKind,
    /// Bok siatki PM; koszt rośnie jak (2·grid)³·log, nie z liczbą par.
    pub grid: usize,
    /// Margines pudła siatki względem rozciągłości chmury.
    pub box_margin: f64,
    /// Co ile kroków mierzyć błąd siły względem dokładnego O(N²); 0 = nigdy.
    pub error_check_every: u32,
    pub error_check_sample: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            backend: BackendKind::Auto,
            grid: 64,
            box_margin: 0.15,
            error_check_every: 0,
            error_check_sample: 512,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RunConfig {
    /// 0 = bez limitu.
    pub steps: u64,
    pub out_dir: String,
    pub diagnostics_every: u32,
    pub trajectory_every: u32,
    pub point_stride: usize,
    /// Ile kroków symulacji na jedną iterację pętli.
    pub time_scale: u32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            steps: 0,
            out_dir: "runs/latest".to_string(),
            diagnostics_every: 20,
            trajectory_every: 20,
            point_stride: 1,
            time_scale: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub spawn: SpawnConfig,
    pub physics: PhysicsConfig,
    pub solver: SolverConfig,
    pub run: RunConfig,
}

impl Config {
    /// Złóż konfigurację z runtime'owych pól `live` i startowych z `self`.
    ///
    /// Wymienione są pola, które WOLNO zmienić w locie; wszystko poza nimi zostaje
    /// z biegu. Dzięki temu panel może wysyłać cały swój stan, a bieg nie przestawi
    /// sobie liczby cząstek ani ziarna w połowie.
    pub fn with_runtime_from(&self, live: &Config) -> Config {
        let mut out = self.clone();
        out.physics.g = live.physics.g;
        out.physics.c = live.physics.c;
        out.physics.softening = live.physics.softening;
        out.physics.dt_max = live.physics.dt_max;
        out.physics.accuracy = live.physics.accuracy;
        out.physics.adaptive_dt = live.physics.adaptive_dt;
        out.physics.cooling_rate = live.physics.cooling_rate;
        out.physics.cooling_density_power = live.physics.cooling_density_power;
        out.physics.cooling_floor = live.physics.cooling_floor;
        out.physics.cooling_min_particles = live.physics.cooling_min_particles;
        out.solver.box_margin = live.solver.box_margin;
        out.solver.error_check_every = live.solver.error_check_every;
        out.run.time_scale = live.run.time_scale;
        out.run.diagnostics_every = live.run.diagnostics_every;
        out.run.trajectory_every = live.run.trajectory_every;
        out.run.point_stride = live.run.point_stride;
        out
    }

    /// Czy zmiana wymaga przebudowania solvera.
    pub fn solver_changed(&self, other: &Config) -> bool {
        self.solver != other.solver
    }

    /// Czy zmiana unieważnia policzone pole sił.
    pub fn field_invalidated(&self, other: &Config) -> bool {
        self.physics.g != other.physics.g || self.physics.softening != other.physics.softening
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("konfiguracja jest zawsze serializowalna")
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

/// `G` takie, że prędkość okrężna na brzegu chmury wynosi `beta·c`.
///
/// Bez tego dobór G jest zgadywaniem: przy G = 1 i masie układu 20 000 prędkość
/// okrężna wynosi √(GM/R) ≈ 45, więc dla c = 30 warunek orbity kołowej jest
/// NIESPEŁNIALNY. Prędkości trafiają wtedy na limit 0,95 c, układ startuje
/// z energią dodatnią i po prostu się rozlatuje — co wygląda jak błąd fizyki,
/// a jest błędem doboru parametrów.
pub fn gravity_for_beta(total_mass: f64, radius: f64, c: f64, beta: f64) -> f64 {
    (beta * c).powi(2) * radius / total_mass.max(1e-30)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_keeps_startup_fields() {
        let start = Config::default();
        let mut live = start.clone();
        live.spawn.n_particles = 99;
        live.spawn.seed = 7;
        live.solver.backend = BackendKind::Mesh;
        live.solver.grid = 128;
        live.physics.g = 0.5;
        live.physics.cooling_rate = 1.0;
        live.physics.cooling_grid = 32;
        let next = start.with_runtime_from(&live);
        assert_eq!(next.spawn.n_particles, start.spawn.n_particles);
        assert_eq!(next.spawn.seed, start.spawn.seed);
        assert_eq!(next.solver.backend, start.solver.backend);
        assert_eq!(next.solver.grid, start.solver.grid);
        assert_eq!(next.physics.cooling_grid, start.physics.cooling_grid);
        assert_eq!(next.physics.g, 0.5);
        assert_eq!(next.physics.cooling_rate, 1.0);
    }

    #[test]
    fn json_round_trip_preserves_config() {
        let cfg = Config::default();
        let back = Config::from_json(&cfg.to_json()).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn unknown_json_fields_are_ignored() {
        let text = r#"{ "spawn": { "n_particles": 12, "nieznane": true } }"#;
        let cfg = Config::from_json(text).unwrap();
        assert_eq!(cfg.spawn.n_particles, 12);
        assert_eq!(cfg.physics.c, PhysicsConfig::default().c);
    }
}
