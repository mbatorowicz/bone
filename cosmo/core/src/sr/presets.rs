//! Nazwane zestawy parametrów wraz z uzasadnieniem liczb.
//!
//! Każdy preset to hipoteza fizyczna, nie zestaw ładnych domyślnych. Komentarze
//! podają, skąd wzięła się każda liczba — bez tego zmiana jednej z nich wygląda na
//! nieszkodliwą, a potrafi unieważnić cały wynik.
//!
//! Softening w presetach siatkowych jest dobrany do oczka, które wyjdzie
//! z rozciągłości danego kształtu. Gdyby był mniejszy, solver i tak podniósłby go do
//! rozmiaru komórki — lepiej poprosić o to, co jest osiągalne.

use crate::sr::config::{
    gravity_for_beta, Config, Geometry, PhysicsConfig, SolverConfig, SpawnConfig,
};

/// Nazwany zestaw parametrów.
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub build: fn() -> Config,
}

/// Presety w kolejności wyświetlania.
pub const PRESETS: &[Preset] = &[
    Preset {
        id: "galaxy",
        label: "Galaktyka",
        build: galaxy,
    },
    Preset {
        id: "collapse",
        label: "Kolaps",
        build: collapse,
    },
    Preset {
        id: "relativistic",
        label: "Relatywistyczny",
        build: relativistic,
    },
    Preset {
        id: "merger",
        label: "Zderzenie",
        build: merger,
    },
    Preset {
        id: "fragmentation",
        label: "Fragmentacja",
        build: fragmentation,
    },
    Preset {
        id: "dissipation",
        label: "Dyssypacja",
        build: dissipation,
    },
    // presety porównawcze kształtów: identyczne liczby, różna geometria
    Preset {
        id: "shape_cube",
        label: "Kostka",
        build: shape_cube,
    },
    Preset {
        id: "shape_cylinder",
        label: "Walec",
        build: shape_cylinder,
    },
    Preset {
        id: "shape_torus",
        label: "Torus",
        build: shape_torus,
    },
    Preset {
        id: "shape_slab",
        label: "Placek",
        build: shape_slab,
    },
    Preset {
        id: "precision",
        label: "Precyzja",
        build: precision,
    },
];

pub fn preset(id: &str) -> Option<Config> {
    PRESETS.iter().find(|p| p.id == id).map(|p| (p.build)())
}

pub fn ids() -> Vec<&'static str> {
    PRESETS.iter().map(|p| p.id).collect()
}

/// Wirujący dysk — rotacja równoważy grawitację, struktura się utrzymuje.
pub fn galaxy() -> Config {
    let (mass, radius, c) = (4_000.0, 10.0, 30.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Disk,
            n_particles: 20_000,
            radius,
            total_mass: mass,
            // grubość podana wprost: domyślne 0,08 dałoby dysk dwa razy grubszy niż
            // ten, na którym preset był dobrany
            thickness: 0.04,
            rotation: 1.0,
            temperature: 0.004,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.25),
            softening: 0.3,
            dt_max: 0.01,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 96,
            ..base.solver
        },
        run: base.run,
    }
}

/// Zimna kula — kolaps grawitacyjny, wzrost γ, potem wirializacja.
pub fn collapse() -> Config {
    let (mass, radius, c) = (4_000.0, 10.0, 25.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Ball,
            n_particles: 20_000,
            radius,
            total_mass: mass,
            rotation: 0.0,
            temperature: 0.0,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.3),
            softening: 0.3,
            dt_max: 0.01,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 96,
            ..base.solver
        },
        run: base.run,
    }
}

/// Wysokie β — efekty relatywistyczne widoczne gołym okiem.
pub fn relativistic() -> Config {
    let (mass, radius, c) = (4_000.0, 6.0, 10.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Plummer,
            n_particles: 8_000,
            radius,
            total_mass: mass,
            rotation: 0.5,
            temperature: 0.1,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.7),
            softening: 0.5,
            dt_max: 0.004,
            accuracy: 0.02,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 64,
            ..base.solver
        },
        run: base.run,
    }
}

/// Dwie gromady na kursie kolizyjnym.
pub fn merger() -> Config {
    let (mass, radius, c) = (4_000.0, 12.0, 30.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::TwoClumps,
            n_particles: 24_000,
            radius,
            total_mass: mass,
            rotation: 0.35,
            temperature: 0.01,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.25),
            softening: 0.4,
            dt_max: 0.01,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 96,
            ..base.solver
        },
        run: base.run,
    }
}

/// Wirujący cienki torus rozpadający się na osobne zgęstki.
///
/// Fragmentacja wymaga, żeby najbardziej niestabilna długość fali była wyraźnie
/// mniejsza od obiektu ORAZ rosła szybciej niż jakikolwiek mod globalny. Drugi
/// warunek eliminuje po kolei niemal wszystkie kształty:
///
/// * zwirializowana kula — λ_Jeansa jest u niej rzędu promienia, brak okna;
/// * samograwitujący dysk — λ_kryt ≈ 4πR, czyli więcej niż on sam, więc wychodzi
///   jeden bar, a nie rój grudek;
/// * włókno — ma swobodne końce, a te zapadają się do środka w tempie ~1/grubość,
///   czyli DOKŁADNIE tak samo, jak rośnie fragmentacja. Pocienianie przyspiesza oba
///   mody równo i nie rozdziela ich (zmierzone dla grubości 0,64 i 0,28: za każdym
///   razem włókno zbiera się w jeden obiekt).
///
/// Cienki torus usuwa oba globalne mody naraz: nie ma końców, więc nie ma efektu
/// brzegowego, a rotacja wyłącza zapadanie promieniowe. Zostaje wyłącznie
/// fragmentacja podłużna.
pub fn fragmentation() -> Config {
    let (mass, radius, c) = (4_000.0, 8.0, 30.0);
    // Przekrój musi być stabilny, czyli masa liniowa poniżej krytycznej
    // M_lin < 2σ_v²/G: przy obwodzie 2π·8 = 50 i masie 4000 mamy M_lin = 79,5,
    // co wymaga σ_v > 1,69 — stąd 1,7, na granicy. Wtedy λ = σ_v√(π/Gρ) ≈ 1,5
    // mieści się 33 razy na obwodzie i wypada na 15 oczkach siatki.
    let sigma_1d = 1.7;
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Torus,
            n_particles: 120_000,
            radius,
            total_mass: mass,
            // przekrój pierścienia: promień mniejszy 0,085·8 = 0,68, czyli σ_rms ≈ 0,48
            thickness: 0.085,
            // rotacja lekko ponad 1, bo prędkość okrężna jest liczona z masy zamkniętej
            // w promieniu, a cienki pierścień przyciąga swoje elementy mocniej niż masa
            // punktowa w środku — potrzebuje o kilka procent więcej
            rotation: 1.05,
            // `temperature` jest ułamkiem c, a spawn rozdziela ją na trzy składowe,
            // więc σ na składową to temperature·c/√3 — stąd ten przelicznik
            temperature: sigma_1d * 3.0_f64.sqrt() / c,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.2),
            softening: 0.03,
            dt_max: 0.004,
            accuracy: 0.025,
            ..base.physics
        },
        solver: SolverConfig {
            // pudło dopasowuje się do rozciągłości pierścienia (17,4), a istotna jest
            // grubość przekroju (0,48) — stąd siatka musi być gęsta, mimo że sam obiekt
            // jest cienki: oczko 0,098 daje 15 komórek na długość fali
            grid: 192,
            box_margin: 0.05,
            error_check_every: 100,
            ..base.solver
        },
        run: crate::sr::config::RunConfig {
            // zgęstki rosną przez ~2,5 jednostki czasu, a potem zaczynają się łączyć;
            // gęsty zapis klatek pozwala analizować właściwą chwilę, nie tylko koniec
            trajectory_every: 10,
            ..base.run
        },
    }
}

/// Zwirializowana kula, która się chłodzi — i dlatego fragmentuje.
///
/// Kontrprzykład do presetu [`fragmentation`]. Tam trzeba było uciekać od kuli do
/// pierścienia, bo w modelu w pełni zachowawczym zwirializowana chmura jest
/// zamknięta: jej rozkład prędkości podtrzymuje ją przeciw zapadaniu, a energii nie
/// ma gdzie podziać. Tutaj energia ma gdzie się podziać, więc ta sama kula robi to,
/// czego przedtem zrobić nie mogła.
///
/// Długość Jeansa λ = σ√(π/Gρ) jest w zwirializowanej kuli rzędu jej promienia,
/// czyli mieści się w niej JEDNA masa Jeansa — stąd zapadanie monolityczne albo
/// żadne. Chłodzenie obniża σ, więc λ maleje, a liczba mas Jeansa rośnie jak
/// (R/λ)³: spadek σ trzykrotny daje kilka fragmentów, pięciokrotny kilkadziesiąt.
///
/// Ograniczenie, które trzeba wypowiedzieć wprost: masa fragmentu jest tu ustalona
/// przez rozdzielczość, a nie przez fizykę. Prawdziwa masa wynikałaby z funkcji
/// chłodzenia i równania stanu gazu, których ten model nie ma. Wynik mówi więc
/// „chłodzenie prowadzi do fragmentacji", a nie „fragmenty mają taką masę".
///
/// Kula JEDNORODNA, nie profil Plummera, i to nie jest kwestia gustu: Plummer
/// rozciąga cząstki do promienia 6a, mając połowę masy w 1,3a, więc pudło siatki
/// dopasowuje się do rozmiaru dwadzieścia razy większego od interesującego obszaru
/// (zmierzone: oczko 0,288, ε podniesione trzykrotnie, błąd siły 3,7%).
pub fn dissipation() -> Config {
    let (mass, radius, c) = (4_000.0, 8.0, 30.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Ball,
            n_particles: 120_000,
            radius,
            total_mass: mass,
            // warunek wirialny 2K = |U| dla kuli jednorodnej daje σ² = GM/(5R);
            // przy G = 0,072 wychodzi σ ≈ 2,68, czyli temperature = σ√3/c ≈ 0,155.
            // Rotacja przejmuje część podparcia, więc dyspersja jest odrobinę niższa.
            rotation: 0.2,
            temperature: 0.15,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.2),
            // ε ledwo nad oczkiem siatki (0,144), żeby nie było po cichu podnoszone
            softening: 0.15,
            dt_max: 0.01,
            accuracy: 0.03,
            // λ = 1 przy czasie dynamicznym 2,7: chłodzenie szybsze od zapadania, ale
            // nie o rzędy wielkości. Przy λ = 2 dryf energii dobijał 6%, bo zgęstki
            // zapadały się poniżej oczka szybciej, niż krok zdążył zmaleć.
            cooling_rate: 1.0,
            cooling_density_power: 1.0,
            // Podłoga wyłącznie numeryczna: 2000 cząstek na masę Jeansa daje λ ≈ 3,3,
            // czyli 22 długości softeningu, i ~60 fragmentów w kuli. Bezwzględna
            // podłoga dyspersji jest zbędna, bo nie ma fizycznej temperatury, którą
            // miałaby reprezentować.
            cooling_floor: 0.0,
            cooling_min_particles: 2_000.0,
            // automat wybierze 31³, czyli ~9 cząstek na zajętą komórkę — tyle trzeba,
            // żeby dyspersja nie była zdominowana przez szum próbkowania
            cooling_grid: 0,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 128,
            box_margin: 0.1,
            error_check_every: 200,
            ..base.solver
        },
        run: crate::sr::config::RunConfig {
            trajectory_every: 10,
            ..base.run
        },
    }
}

/// Wspólna podstawa presetów kształtu — jedyną różnicą jest geometria.
///
/// Sens tych presetów jest porównawczy, więc masa, promień, G, c i rozdzielczość
/// MUSZĄ być identyczne. Gdyby każdy kształt miał własne liczby, różnice w wyniku
/// dałoby się przypisać czemukolwiek, a nie kształtowi.
///
/// Warunek startowy jest zadany WIRIAŁEM, nie temperaturą, i to jest tu rzecz
/// najważniejsza. Ustalona dyspersja nie jest porównywalna między kształtami: ta
/// sama liczba daje 2K/|U| = 1,34 dla kostki i 0,49 dla włókna, czyli różnicę
/// 2,7-krotną. Przy 2K/|U| = 1 każdy kształt startuje z własnej równowagi i jedyną
/// zmienną pozostaje geometria.
fn shape_probe(geometry: Geometry) -> Config {
    let (mass, radius, c) = (4_000.0, 8.0, 30.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry,
            n_particles: 40_000,
            radius,
            total_mass: mass,
            rotation: 0.0,
            virial: 1.0,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.2),
            softening: 0.15,
            dt_max: 0.01,
            accuracy: 0.03,
            ..base.physics
        },
        solver: SolverConfig {
            grid: 128,
            box_margin: 0.1,
            error_check_every: 200,
            ..base.solver
        },
        run: crate::sr::config::RunConfig {
            trajectory_every: 20,
            ..base.run
        },
    }
}

/// Kostka zwirializowana — brzeg płaski, więc narożniki opadają pierwsze.
pub fn shape_cube() -> Config {
    shape_probe(Geometry::Cube)
}

/// Walec zwirializowany — jedna wyróżniona oś, dwa swobodne końce.
pub fn shape_cylinder() -> Config {
    shape_probe(Geometry::Cylinder)
}

/// Gruby torus bez rotacji — dziura w środku zamyka się albo nie.
pub fn shape_torus() -> Config {
    let mut cfg = shape_probe(Geometry::Torus);
    cfg.spawn.thickness = 0.35;
    cfg
}

/// Kula spłaszczona do placka — ten sam materiał, inne proporcje.
///
/// Preset istnieje po to, żeby pokazać, co robi samo spłaszczenie: masa, promień
/// i dyspersja są takie jak w pozostałych presetach kształtu, zmienia się tylko
/// proporcja pionowa. Placek zapada się najpierw w cieńszym kierunku, bo tam droga
/// swobodnego spadku jest krótsza — i to jest widoczne od pierwszych kroków.
pub fn shape_slab() -> Config {
    let mut cfg = shape_probe(Geometry::Ball);
    cfg.spawn.flatten = 0.15;
    cfg
}

/// Mało cząstek, backend dokładny — do sprawdzania zachowania energii.
pub fn precision() -> Config {
    let (mass, radius, c) = (4_000.0, 8.0, 30.0);
    let base = Config::default();
    Config {
        spawn: SpawnConfig {
            geometry: Geometry::Plummer,
            n_particles: 2_000,
            radius,
            total_mass: mass,
            rotation: 0.5,
            temperature: 0.0,
            ..base.spawn
        },
        physics: PhysicsConfig {
            c,
            g: gravity_for_beta(mass, radius, c, 0.2),
            softening: 0.25,
            dt_max: 0.005,
            accuracy: 0.015,
            ..base.physics
        },
        solver: SolverConfig {
            backend: crate::sr::config::BackendKind::Exact,
            ..base.solver
        },
        run: crate::sr::config::RunConfig {
            diagnostics_every: 25,
            ..base.run
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_is_reachable_by_id() {
        for p in PRESETS {
            assert!(preset(p.id).is_some(), "preset {} nieosiągalny", p.id);
        }
        assert!(preset("nie ma takiego").is_none());
    }

    #[test]
    fn preset_ids_are_unique() {
        let mut seen = std::collections::BTreeSet::new();
        for p in PRESETS {
            assert!(seen.insert(p.id), "zduplikowany identyfikator {}", p.id);
        }
    }

    /// Presety kształtu mają sens tylko wtedy, gdy różnią się WYŁĄCZNIE geometrią
    /// (i parametrami, które opisują ten kształt). Gdyby któryś dostał inną masę
    /// albo inną siatkę, porównanie między nimi przestałoby cokolwiek znaczyć.
    #[test]
    fn shape_presets_differ_only_in_geometry() {
        let reference = shape_cube();
        for build in [shape_cylinder, shape_torus, shape_slab] {
            let cfg = build();
            assert_eq!(cfg.spawn.n_particles, reference.spawn.n_particles);
            assert_eq!(cfg.spawn.total_mass, reference.spawn.total_mass);
            assert_eq!(cfg.spawn.radius, reference.spawn.radius);
            assert_eq!(cfg.spawn.virial, reference.spawn.virial);
            assert_eq!(cfg.physics, reference.physics);
            assert_eq!(cfg.solver, reference.solver);
        }
    }

    #[test]
    fn presets_are_physically_startable() {
        for p in PRESETS {
            let cfg = (p.build)();
            assert!(cfg.physics.c > 0.0, "{}: c ≤ 0", p.id);
            assert!(cfg.physics.g > 0.0, "{}: G ≤ 0", p.id);
            assert!(cfg.physics.softening > 0.0, "{}: ε ≤ 0", p.id);
            assert!(cfg.spawn.n_particles >= 2, "{}: za mało cząstek", p.id);
            assert!(cfg.spawn.total_mass > 0.0, "{}: masa ≤ 0", p.id);
        }
    }

    /// `gravity_for_beta` musi dawać dokładnie zamawianą prędkość na brzegu,
    /// inaczej cały mechanizm doboru G jest pozorny.
    #[test]
    fn gravity_for_beta_hits_the_requested_speed() {
        let (mass, radius, c, beta) = (4_000.0, 8.0, 30.0, 0.25);
        let g = gravity_for_beta(mass, radius, c, beta);
        let v_circ = (g * mass / radius).sqrt();
        assert!((v_circ / c - beta).abs() < 1e-12, "β={}", v_circ / c);
    }

    /// Preset dyssypacji BEZ chłodzenia byłby zwykłą zwirializowaną kulą, czyli
    /// dokładnie tym, czego ma być kontrprzykładem.
    #[test]
    fn dissipation_actually_dissipates() {
        let cfg = dissipation();
        assert!(cfg.physics.cooling_rate > 0.0);
        assert!(cfg.physics.cooling_min_particles > 0.0);
    }

    #[test]
    fn config_survives_json_round_trip() {
        for p in PRESETS {
            let cfg = (p.build)();
            let back = Config::from_json(&cfg.to_json()).expect("poprawny JSON");
            assert_eq!(cfg, back, "preset {}", p.id);
        }
    }
}
